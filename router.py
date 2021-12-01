import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from typing import List
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from model_util import *
from model_builder import ModelBuilder
from model_builders.rl_model_builder import RLModelBuilder


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Size of each hidden vector of a head
        self.depth = self.args.hidden_size // self.args.num_heads

        self.W_q = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.W_k = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.W_v = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        # W for multi-head attention
        self.mha_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def split_heads(self, x):
        # (batch size, num tokens,  num heads, depth)
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1, self.args.num_heads, self.depth))
        
        # (batch size, num heads, num tokens, depth)
        return x.transpose(2,1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        
        qk = torch.matmul(q, k.transpose(3,2))
        dk = torch.tensor(q.shape[-1], dtype=torch.float32)
        
        scaled_att_logits = qk / torch.sqrt(dk)
        if mask is not None:
            scaled_att_logits += (mask * -1e9)
            
        att_weights = F.softmax(scaled_att_logits, dim=-1)
        scaled_att = torch.matmul(att_weights, v)

        # (batch_size, mbs+hidden+enc_state, num_heads, depth)
        scaled_att = scaled_att.transpose(2,1)

        # Squeeze MHA together for each embedding
        scaled_att = torch.reshape(scaled_att, (batch_size, -1, self.args.hidden_size))

        # Combine the MHA for each embedding
        
        out = self.mha_W(scaled_att)
        return out, att_weights

class PointWiseFFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.hidden_size, self.args.dff)
        self.fc2 = nn.Linear(self.args.dff, self.args.hidden_size)

        self.act1 = nn.GELU()

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)

       
    def forward(self, x):
        pw_out = self.act1(self.fc1(x))
        pw_out = self.dropout1(pw_out)
        pw_out = self.fc2(pw_out)
        pw_out = self.dropout2(pw_out)
       
        return pw_out


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mha = MultiHeadAttention(self.args)

        self.pw_ffn = PointWiseFFN(self.args)

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)

        self.norm1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)

    def forward(self, x, mask=None):
        # Perform MHA attention
        x = self.norm1(x)
        att_out, att_weights = self.mha(x, x, x, mask)
        x = self.norm2(att_out + x)

        # Plug through Point-wise FFN
        x = self.pw_ffn(x) + x        
        return x, att_weights


class Encoder(nn.Module):
    def __init__(self, args, num_enc_layers=1):
        super().__init__()
        self.args = args
        
        self.enc_layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)

        # Create learned positional encodings
        num_patches = (self.args.img_size**2) // self.args.patch_size**2
        if self.args.rl:
            # class emb + task_emb + num_patches + max_depth
            max_inp = 3 + self.args.max_depth
        else:
            # class emb + task_emb + num_patches + max_depth
            max_inp = 2 + num_patches + self.args.max_depth
        self.pos_enc = nn.Parameter(torch.randn(1, max_inp, self.args.hidden_size))


    def forward(self, x, mask=None):
        """
            x: task emb, example emb, used mbs embs
        """
        x = x * torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32)).cuda()
        x = x + self.pos_enc[:, :x.shape[1], :]
        x = self.dropout(x)

        # Run through all encoder mb
        for enc_layer in self.enc_layers:
            x, att_weights = enc_layer(x, mask)
        
        # Return output of last encoder mb
        return x, att_weights




class TransformerRouter(nn.Module):
    def __init__(self, args, num_mod):
        super().__init__()
        self.args = args
        self.num_mod = num_mod
        
        # Token for prediction the actor prediction and critic
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.args.hidden_size))
        
        
        self.encoder = Encoder(self.args, self.args.num_enc_layers)
        
        # Maps each embedding to distr value
        self.actor_out1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.actor_out2 = nn.Linear(self.args.hidden_size, self.num_mod+1)

        self.critic_out1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.critic_out2 = nn.Linear(self.args.hidden_size, 1)
        
        #self.actor_out2 = nn.Linear(self.args.hidden_size, 1)

    def forward(self, state, mb_mask=None, action_mask=None):
       
        # Add prediction token to state
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = state.shape[0])
        state = torch.cat((cls_tokens, state), 1)
        
        enc_output, att_weights = self.encoder(state, mb_mask)
        

        
        # dec_output, att_weights = self.decoder(
        #     mb_embs, enc_output, mb_mask)
        
        # batch size, len(mb_embs)

        # Get the class prediction
        actor_out = F.gelu(self.actor_out1(enc_output[:, 0]))
        actor_out = self.actor_out2(actor_out)


        val_preds = F.gelu(self.critic_out1(enc_output[:, 0]))
        val_preds = self.critic_out2(val_preds)

        # Apply masks over invalid actions
        if action_mask is not None:
            actor_out += action_mask * -1e9

        # Get probabilities over modules
        mb_probs = F.softmax(actor_out, dim=1)

        return mb_probs, val_preds, att_weights



class MuJoCoEncoder(nn.Module):
    """Encoder for MuJoCo Observation."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.mw_obs_size, self.args.hidden_size//2)
        self.fc2 = nn.Linear(self.args.hidden_size//2, self.args.hidden_size)
        self.fc3 =  nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        return self.fc3(obs)

class ImgEncoder(nn.Module):
    """Create patch embeddings for a given image."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_dim = self.args.patch_size ** 2 * self.args.in_channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.args.patch_size, p2 = self.args.patch_size),
            nn.Linear(self.patch_dim, self.args.hidden_size),
        )

    def forward(self, imgs):
        # Extract patches and embed them
        patches = self.to_patch_embedding(imgs)
        return patches

class ALEStateEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc_1 = nn.Linear(self.args.in_channels, self.args.hidden_size)
        self.act_1 = nn.GELU()
        self.fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
    
    def forward(self, x):
        x = self.act_1(self.fc_1(x))
        x = self.fc_2(x)
        return x

class Router(nn.Module):
    def __init__(self, args, num_tasks, get_mb_fn, num_mbs):
        super().__init__()
        self.args = args
        self.get_mb_fn = get_mb_fn
        
        
        self.num_mbs = num_mbs
        if self.args.rl:
            self.num_mbs = self.num_mbs - 1
        self.num_tasks = num_tasks
        # self._valid_mbs_dict = {}

        self.trans_router = TransformerRouter(self.args, self.num_mbs)

        self.mb_embs = nn.Embedding(self.num_mbs+1, self.args.hidden_size)

        self.task_embs = nn.Embedding(self.num_tasks, self.args.hidden_size)
        
        if self.args.metaworld:
            self.state_encoder = MuJoCoEncoder(self.args)
        elif self.args.rl:
            self.state_encoder = ALEStateEncoder(self.args)
        else:
            self.state_encoder = ImgEncoder(self.args)
                
        # self.critic = Critic(self.args)


    def forward(self, imgs, task_id, hidden_state=None, argmax=False, should_drop_mbs=True):
        batch_size = imgs.shape[0]

        if self.args.rl:
            builders = [RLModelBuilder(self.args) for _ in range(batch_size)]
        else:
            builders = [ModelBuilder(self.args) for _ in range(batch_size)]

        mb_emb_indices = [list(range(self.num_mbs+1)) for _ in range(batch_size) ]

        critic_outputs = torch.zeros(batch_size, self.args.max_depth).cuda()
        
        # Store the state of the actions every timestep
        actions = [[] for _ in range(batch_size)]

        # mb_mask = torch.zeros(self.num_mbs+1).cuda()
        cur_loss = None
        action_probs = torch.zeros(batch_size, self.args.max_depth, self.num_mbs+1).cuda()
        
        att_weights_list = [[] for _ in range(batch_size)]
        valid_mbs_ep = [[] for _ in range(batch_size)]
        
        mb_count = torch.zeros(self.num_mbs  + 1).cuda()

        # Mask used for invalid actions
        actions_masks = torch.ones(
            batch_size,
            self.args.max_depth,
            self.num_mbs+1).cuda()

        # Build the model architecture 
        for i in range(self.args.max_depth):
            # Update the valid mbs over the batch
            valid_mbs_batch = []
            max_emb_len = 0

            for b_i in range(batch_size): 
                if i  == 0:
                    valid_mbs = self._get_valid_mbs(builders[b_i], mb_emb_indices[b_i], i, None)
                else:
                    valid_mbs = self._get_valid_mbs(builders[b_i], mb_emb_indices[b_i], i, tuple(actions[b_i]))
                # Perform mb dropout
                if should_drop_mbs:
                    valid_mbs = self.mb_dropout(valid_mbs)
                valid_mbs_batch.append(torch.tensor(valid_mbs))
                valid_mbs_ep[b_i].append(valid_mbs)
                
                # Update module mask to only mask out invalid actions
                actions_masks[b_i, i, valid_mbs] = 0
            
            
            # Encode the state
            enc_state = self.state_encoder(imgs)

            # Get task embedding
            task_emb = self.task_embs(task_id)
            
            # Get the embeddings for the model currently getting constructed
            if hidden_state is not None:
                if i == 0:
                    state = torch.cat((task_emb, enc_state, hidden_state), 1)                
                else:
                    mb_embs = self.mb_embs(torch.tensor(actions).cuda())
                    state = torch.cat((task_emb, enc_state, hidden_state, mb_embs), 1)
            else:
                if i == 0:
                    state = torch.cat((task_emb, enc_state), 1)                
                else:
                    mb_embs = self.mb_embs(torch.tensor(actions).cuda())
                    state = torch.cat((task_emb, enc_state, mb_embs), 1)

            # Get router prediction
            mb_probs, val_preds, att_weights = self.trans_router(
                state, action_mask=actions_masks[:, i])
            
            
            # Iterate over choices in a batch
            mb_choices = []
            #bonus = torch.zeros_like(mb_probs)
            #bonus = (1 - actions_masks[:, i]) * 0.0
            for b_i in range(batch_size):
                # Choose a mbs
                if argmax:
                    mb_sample = mb_probs[b_i].argmax()
                    mb_choice = int(mb_sample)#valid_mbs_batch[b_i][mb_sample]
                else:
                    # Sample a mb
                    
                    #sample_prob = (mb_probs[b_i] + bonus[b_i]) / (mb_probs[b_i] + bonus[b_i]).sum()
                    #print("sample_prob", sample_prob)
                     

                    #print("sample_prob", sample_prob)
                    mb_distr = Categorical(mb_probs[b_i])
                    mb_sample = mb_distr.sample()
                    mb_choice = int(mb_sample)#int(valid_mbs_batch[b_i][mb_sample])
                    #assert mb_probs[b_i, -1, mb_sample] != 0
                
                # Each action needs to be associated with its index in the mb_probs
                actions[b_i].append(mb_choice)
                action_probs[b_i,i] = mb_probs[b_i]
                mb_choices.append(mb_choice)
                mb_count[mb_choice] += 1
                # Set critic_outputs
                critic_outputs[b_i,i] = val_preds[b_i]
                
                # Remove the mb choice
                if self.args.rl or mb_choice != PASS_FB:
                    mb_emb_indices[b_i].remove(mb_choice)

                    cur_mb = self.get_mb_fn(mb_choice)

                    # Verify the builder did not fail
                    if self.args.rl:
                        builders[b_i].attempt_add_mb(mb_choice)
                    else:
                        assert builders[b_i].attempt_add_mb(cur_mb, i)
            



        # imp_ratios = torch.clamp(imp_ratios, 0.01, 2.0)
        actions = torch.tensor(actions).cuda()
        return RouterResult(
            action_probs, 
            actions, 
            critic_outputs, 
            att_weights_list, 
            valid_mbs_ep,
            mb_count,
            actions_masks) 

    # TODO: Move this into model_builder.py
    def _get_valid_mbs(self, builder, mb_emb_indices, cur_depth, prev_actions):
        if isinstance(builder, RLModelBuilder):
            return builder.get_valid_mbs(cur_depth)
        else:
            # Update the valid mbs
            valid_mbs = []
            # if (cur_depth, prev_actions) not in self._valid_mbs_dict:
            for l_i in mb_emb_indices:
                if l_i == PASS_FB:
                    mb_l_i = None
                else:    
                    mb_l_i = self.get_mb_fn(l_i)
                
                if builder.is_valid_mb(mb_l_i, cur_depth):
                    valid_mbs.append(l_i)
            
        #self._valid_mbs_dict[(cur_depth, prev_actions)] = valid_mbs
        # else:    
        #     valid_mbs = self._valid_mbs_dict[(cur_depth, prev_actions)]
        return valid_mbs
    
    def mb_dropout(self, valid_mbs):
        if len(valid_mbs) == 1:
            # Require at least one mb
            return valid_mbs

        nondropped_mbs = []
        for i in range(len(valid_mbs)):

            if random.random() >= self.args.mb_drop_rate:
                nondropped_mbs.append(valid_mbs[i])
        
        if len(nondropped_mbs) == 0:
            # Add random mb if all the mbs were dropped
            nondropped_mbs.append(random.choice(valid_mbs))

        return nondropped_mbs
    
    def predict_step(self, imgs, task_id, actions, mb_masks, action_masks, hidden_state=None):
        """Get action and value for timestep."""
        batch_size = imgs.shape[0]

        # Encode the state
        enc_state = self.state_encoder(imgs)

        # Get task embedding
        task_emb = self.task_embs(task_id)

        if len(actions) > 0:
            # Embeddings for the current FBs in model
            mb_embs = self.mb_embs(actions)
            if hidden_state is not None:
                state = torch.cat((task_emb, enc_state, hidden_state, mb_embs), 1)
            else:
                state = torch.cat((task_emb, enc_state, mb_embs), 1)
        else:
            raise Exception("Require actions.")
        
        

        mb_probs, val_preds, _ = self.trans_router(
            state, mb_masks, action_masks)
        return mb_probs, val_preds
            
                
        
