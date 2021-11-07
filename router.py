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

        # (batch_size, fbs+hidden+enc_state, num_heads, depth)
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
        num_patches = (self.args.img_size**2) // self.args.patch_size
        max_inp = 2 + num_patches + self.args.max_depth
        self.pos_enc = nn.Parameter(torch.randn(1, max_inp, self.args.hidden_size))


    def forward(self, x, mask=None):
        """
            x: task emb, example emb, used fbs embs
        """
        x = x * torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32)).cuda()
        x += self.pos_enc[:, :x.shape[1], :]
        x = self.dropout(x)

        # Run through all encoder fb
        for enc_layer in self.enc_layers:
            x, att_weights = enc_layer(x, mask)
        
        # Return output of last encoder fb
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

    def forward(self, state, fb_mask=None, action_mask=None):
       
        # Add prediction token to state
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = state.shape[0])
        state = torch.cat((cls_tokens, state), 1)
        
        enc_output, att_weights = self.encoder(state, fb_mask)
        

        
        # dec_output, att_weights = self.decoder(
        #     fb_embs, enc_output, fb_mask)
        
        # batch size, len(fb_embs)

        # Get the class prediction
        actor_out = F.gelu(self.actor_out1(enc_output[:, 0]))
        actor_out = self.actor_out2(actor_out)


        val_preds = F.gelu(self.critic_out1(enc_output[:, 0]))
        val_preds = self.critic_out2(val_preds)

        # Apply masks over invalid actions
        if action_mask is not None:
            actor_out += action_mask * -1e9

        # Get probabilities over modules
        fb_probs = F.softmax(actor_out, dim=1)

        return fb_probs, val_preds, att_weights



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
        patch_dim = self.args.patch_size ** 2 * self.args.in_channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.args.patch_size, p2 = self.args.patch_size),
            nn.Linear(patch_dim, self.args.hidden_size),
        )

    def forward(self, imgs):
        # Extract patches and embed them
        patches = self.to_patch_embedding(imgs)
        return patches

        

class Router(nn.Module):
    def __init__(self, args, num_tasks, get_fb_fn, num_fbs):
        super().__init__()
        self.args = args
        self.get_fb_fn = get_fb_fn
        self.num_fbs = num_fbs
        self.num_tasks = num_tasks
        self._valid_fbs_dict = {}

        self.trans_router = TransformerRouter(self.args, self.num_fbs)

        self.fb_embs = nn.Embedding(self.num_fbs+1, self.args.hidden_size)

        self.task_embs = nn.Embedding(self.num_tasks, self.args.hidden_size)
        
        if self.args.metaworld:
            self.state_encoder = MuJoCoEncoder(self.args)
        else:
            self.state_encoder = ImgEncoder(self.args)
                
        # self.critic = Critic(self.args)


    def forward(self, imgs, task_id, argmax=False, should_drop_fbs=True):
        batch_size = imgs.shape[0]

        builders = [ModelBuilder(self.args) for _ in range(batch_size)]

        fb_emb_indices = [list(range(self.num_fbs+1)) for _ in range(batch_size) ]

        critic_outputs = torch.zeros(batch_size, self.args.max_depth, 1).cuda()
        # Store the state of the actions every timestep
        actions_ep = torch.zeros(
            self.args.max_depth, batch_size, self.args.max_depth, dtype=torch.int64).cuda()
        
        actions = [[] for _ in range(batch_size)]

        # fb_mask = torch.zeros(self.num_fbs+1).cuda()
        failed = False
        cur_loss = None
        action_probs = torch.zeros(batch_size, self.args.max_depth, self.num_fbs+1).cuda()
        
        att_weights_list = [[] for _ in range(batch_size)]
        valid_fbs_ep = [[] for _ in range(batch_size)]
        
        fb_count = torch.zeros(self.num_fbs  + 1)

        # Mask used for invalid actions
        actions_masks = torch.ones(
            batch_size,
            self.args.max_depth,
            self.num_fbs+1).cuda()

        # Importance ratios
        imp_ratios = torch.ones(batch_size, self.args.max_depth).cuda()

        # Build the model architecture 
        for i in range(self.args.max_depth):
            # Update the valid fbs over the batch
            valid_fbs_batch = []
            max_emb_len = 0

            for b_i in range(batch_size): 
                if i  == 0:
                    valid_fbs = self._get_valid_fbs(builders[b_i], fb_emb_indices[b_i], i, None)
                else:
                    valid_fbs = self._get_valid_fbs(builders[b_i], fb_emb_indices[b_i], i, tuple(actions[b_i]))
                # Perform fb dropout
                if should_drop_fbs:
                    valid_fbs = self.fb_dropout(valid_fbs)
                valid_fbs_batch.append(torch.tensor(valid_fbs))
                valid_fbs_ep[b_i].append(valid_fbs)
                
                # Update module mask to only mask out invalid actions
                actions_masks[b_i, i, valid_fbs] = 0
            
            
            # Encode the state
            enc_state = self.state_encoder(imgs)

            # Get task embedding
            task_emb = self.task_embs(task_id)
            
            # Get the embeddings for the model currently getting constructed
            if i == 0:
                state = torch.cat((task_emb, enc_state), 1)                
            else:
                fb_embs = self.fb_embs(torch.tensor(actions).cuda())
                state = torch.cat((task_emb, enc_state, fb_embs), 1)

            # Get router prediction
            fb_probs, val_preds, att_weights = self.trans_router(
                state, action_mask=actions_masks[:, i])
            
            
            # Iterate over choices in a batch
            fb_choices = []
            #bonus = torch.zeros_like(fb_probs)
            bonus = (1 - actions_masks[:, i]) * 0.0
            for b_i in range(batch_size):
                # Choose a fbs
                if argmax:
                    fb_sample = fb_probs[b_i].argmax()
                    fb_choice = int(fb_sample)#valid_fbs_batch[b_i][fb_sample]
                else:
                    # Sample a fb
                    #print("PREV PROB", fb_probs[b_i])
                    #print("fb_probs[b_i]", fb_probs[b_i])
                    sample_prob = (fb_probs[b_i] + bonus[b_i]) / (fb_probs[b_i] + bonus[b_i]).sum()
                    #print("sample_prob", sample_prob)
                     

                    #print("sample_prob", sample_prob)
                    fb_distr = Categorical(sample_prob)
                    fb_sample = fb_distr.sample()
                    fb_choice = int(fb_sample)#int(valid_fbs_batch[b_i][fb_sample])
                    #imp_ratios[b_i, :i+1] *= fb_probs[b_i, fb_choice] / sample_prob[fb_choice]

                    #assert fb_probs[b_i, -1, fb_sample] != 0
                
                # Each action needs to be associated with its index in the fb_probs
                actions[b_i].append(fb_choice)
                action_probs[b_i,i] = fb_probs[b_i]
                fb_choices.append(fb_choice)
                fb_count[fb_choice] += 1
                # Set critic_outputs
                critic_outputs[b_i,i] = val_preds[b_i]
                # Remove the fb choice
                if fb_choice != PASS_FB:
                    fb_emb_indices[b_i].remove(fb_choice)

                    cur_fb = self.get_fb_fn(fb_choice)

                    # Verify the builder did not fail
                    assert builders[b_i].attempt_add_fb(cur_fb, i)
            


            # Add actions for this timestep
            actions_ep[i, :, :i+1] = torch.tensor(actions)
        imp_ratios = torch.clamp(imp_ratios, 0.01, 2.0)
        return RouterResult(
            failed, 
            action_probs, 
            actions_ep, 
            critic_outputs, 
            att_weights_list, 
            valid_fbs_ep,
            fb_count,
            actions_masks,
            imp_ratios) 


    def _get_valid_fbs(self, builder, fb_emb_indices, cur_depth, prev_actions):
        # Update the valid fbs
        valid_fbs = []
        if (cur_depth, prev_actions) not in self._valid_fbs_dict:
            for l_i in fb_emb_indices:
                if l_i == PASS_FB:
                    fb_l_i = None
                else:    
                    fb_l_i = self.get_fb_fn(l_i)
                
                if builder.is_valid_fb(fb_l_i, cur_depth):
                    valid_fbs.append(l_i)
            self._valid_fbs_dict[(cur_depth, prev_actions)] = valid_fbs
        else:    
            valid_fbs = self._valid_fbs_dict[(cur_depth, prev_actions)]
        return valid_fbs
    
    def fb_dropout(self, valid_fbs):
        if len(valid_fbs) == 1:
            # Require at least one fb
            return valid_fbs

        nondropped_fbs = []
        for i in range(len(valid_fbs)):

            if random.random() >= self.args.fb_drop_rate:
                nondropped_fbs.append(valid_fbs[i])
        
        if len(nondropped_fbs) == 0:
            # Add random fb if all the fbs were dropped
            nondropped_fbs.append(random.choice(valid_fbs))

        return nondropped_fbs
    
    def predict_step(self, imgs, task_id, actions, fb_masks, action_masks):
        """Get action and value for timestep."""
        batch_size = imgs.shape[0]

        # Encode the state
        enc_state = self.state_encoder(imgs)

        # Get task embedding
        task_emb = self.task_embs(task_id)

        if len(actions) > 0:
            # Embeddings for the current FBs in model
            fb_embs = self.fb_embs(actions)
            state = torch.cat((task_emb, enc_state, fb_embs), 1)
        else:
            raise Exception("Require actions.")
        
        

        fb_probs, val_preds, _ = self.trans_router(
            state, fb_masks, action_masks)
        return fb_probs, val_preds
            
                
        
