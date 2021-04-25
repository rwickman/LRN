import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from typing import List
from torch.nn.utils.rnn import pad_sequence

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
        out = self.mha_W(scaled_att).squeeze(0)

        return out, att_weights

class PointWiseFFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.hidden_size, self.args.dff)
        self.fc2 = nn.Linear(self.args.dff, self.args.hidden_size)
        #self.dropout1 = nn.Dropout(self.args.drop_rate)
        #self.norm1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)

    def forward(self, x):
        pw_out = F.relu(self.fc1(x))
        pw_out = self.fc2(pw_out)
        #pw_out = self.dropout1(pw_out)
        #out = self.norm1(x + pw_out)

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
        att_out, _ = self.mha(x, x, x, mask)
        att_out = self.dropout1(att_out)
        mha_out = self.norm1(x + att_out)

        # Plug through Point-wise FFN
        pw_ffn_out = self.pw_ffn(mha_out)
        pw_ffn_out = self.dropout2(pw_ffn_out)
        out = self.norm2(mha_out + pw_ffn_out)
        
        return out

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mha1 = MultiHeadAttention(self.args)
        self.mha2 = MultiHeadAttention(self.args)

        self.pw_ffn = PointWiseFFN(self.args)

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)
        self.dropout3 = nn.Dropout(self.args.drop_rate)
        
        self.norm1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
    
    def forward(self, x, enc_output, fb_mask=None):
        # Preform self attention over available fbs
        att_out, att_weights1 = self.mha1(x, x, x, fb_mask)
        att_out = self.dropout1(att_out)
        mha_out1 = self.norm1(x + att_out)

        # Perform attention over encoder output
        enc_att_out, att_weights2 = self.mha2(mha_out1, enc_output, enc_output)
        enc_att_out = self.dropout2(enc_att_out)
        mha_out2 = self.norm2(mha_out1 + enc_att_out)

        pw_ffn_out = self.pw_ffn(mha_out2)
        pw_ffn_out = self.dropout3(pw_ffn_out)
        out = self.norm3(mha_out2 + pw_ffn_out)

        return out, att_weights1, att_weights2

class Encoder(nn.Module):
    def __init__(self, args, num_enc_layers=1):
        super().__init__()
        self.args = args
        
        self.enc_layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)
    
        self.pos_enc = positional_encoding(self.args.max_pos_enc, self.args.hidden_size)

    def forward(self, x, mask=None):
        """
            x: task emb, example emb, used fbs embs
        """
        x *= torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32)).cuda()
        x += self.pos_enc[:, :x.shape[1], :]
        x = self.dropout(x)

        # Run through all encoder fb
        for enc_layer in self.enc_layers:
            x = enc_layer(x, mask)
        
        # Return output of last encoder fb
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.pos_enc = positional_encoding(self.args.max_pos_enc, self.args.hidden_size)

        self.dec_layers = nn.ModuleList([DecoderLayer(self.args) for _ in range(self.args.num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)
    
    def forward(self, x, enc_output, fb_mask=None):
        """Args:
            x: Tensor of available fb embeddings.
            enc_output: Tensor of output of last encoder fb.
        """
        num_fbs = x.shape[1]
        x *= torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32))
        x += self.pos_enc[:, :num_fbs, :]
        x = self.dropout(x)
        
        att_weights = {}

        # Run through all decoder fbs
        for i, dec_layer in enumerate(self.dec_layers):
            x, att_weights1, att_weights2 = dec_layer(x, enc_output, fb_mask)

            att_weights['decoder_fb{}_weights1'.format(i+1)] = att_weights1
            att_weights['decoder_fb{}_weights2'.format(i+1)] = att_weights2

        return x, att_weights


class TransformerRouter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = Encoder(self.args, self.args.num_enc_layers)
        self.decoder = Decoder(self.args)

        # Maps each decorder embedding to 1 value
        self.actor_out1 = nn.Linear(self.args.hidden_size, 1)
        #self.actor_out2 = nn.Linear(self.args.hidden_size, 1)
        

    def forward(self, state, fb_embs, fb_mask=None, action_mask=None):
        enc_output = self.encoder(state, action_mask)
        
        dec_output, att_weights = self.decoder(fb_embs, enc_output, fb_mask)
        # batch size, len(fb_embs)
        actor_out = self.actor_out1(dec_output).squeeze(-1)
        #actor_out = self.actor_out2(actor_out).squeeze(-1)
        if fb_mask is not None:
            actor_out += torch.reshape(fb_mask, (fb_mask.shape[0], fb_mask.shape[-1])) * -1e9
        fb_probs = F.softmax(actor_out, dim=-1)
        return fb_probs, att_weights
        


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # W for hidden state
        self.hidden_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        
        # W for encoded state
        self.state_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        # W for fb embeddings
        self.fb_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

         # W for fb embeddings
        self.task_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self.V = nn.Linear(self.args.hidden_size, 1)

        self.mha = MultiHeadAttention(args)

        self.norm1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        self.pw_ffn = PointWiseFFN(self.args)

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)


        
    def forward(self, fb_embs, hidden_state, enc_state, task_emb):
        inp = torch.cat((fb_embs.squeeze(0), hidden_state, enc_state, task_emb))
        mha_out, _ = self.mha(inp, inp, inp)
        mha_out = self.dropout1(mha_out)
        mha_out = self.norm1(mha_out + inp)
        
        pw_out = self.pw_ffn(mha_out)
        pw_out = self.dropout2(pw_out)
        out = self.norm2(pw_out + mha_out)


        # (batch_size, num_fbs, hidden_size)
        fb_trans = self.fb_W(out[:-3])
        
        # (batch_size, hidden_size)
        hidden_trans = self.hidden_W(out[-3].unsqueeze(0)).unsqueeze(1)
        
        state_trans = self.state_W(out[-2].unsqueeze(0)).unsqueeze(1)

        task_trans = self.task_W(out[-1].unsqueeze(0)).unsqueeze(1)

        # (batch_size, num_fbs)
        u_i = self.V(torch.tanh(fb_trans + hidden_trans + state_trans + task_trans)).squeeze(-1)
    
        # Scale the output
        u_i = u_i / torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32))

        # Predictions over fbs
        fb_probs = F.softmax(u_i, dim=-1)
        return fb_probs


class MuJoCoEncoder(nn.Module):
    """Encoder for MuJoCo Observation."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.mw_obs_size, self.args.hidden_size//2)
        self.fc2 = nn.Linear(self.args.hidden_size//2, self.args.hidden_size)
        self.fc3 =  nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def forward(self, obs):
        obs = nn.relu(self.fc1(obs))
        obs = nn.relu(self.fc2(obs))
        return self.fc3(obs)

class ImgEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.conv1 = nn.Conv2d(self.args.in_channels, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.norm1 = nn.InstanceNorm2d(32, affine=True, eps=1e-6)
        self.norm2 = nn.InstanceNorm2d(32, affine=True, eps=1e-6)
                
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(self.args.drop_rate)
        
        # Compute output size of last conv fb
        conv2d_out_size = conv2d_out(
            conv2d_out(self.args.img_size, 3, 1, 1),
         3, 1, 1)
        num_units = int(conv2d_out_size**2 * 32)
        self.fc1 = nn.Linear(num_units, self.args.hidden_size)
        #self.fc2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def forward(self, state):
        state = F.relu(self.norm1(self.conv1(state)))
        #state = F.relu(self.conv1(state))
        state = F.relu(self.norm2(self.conv2(state)))
        #state = F.relu(self.norm3(self.conv3(state)))
        state = self.flatten(state)
        state = self.drop1(state)
        # state =self.fc1(state)
        return self.fc1(state)


# class Critic(nn.Module):
#     def __init__(args):
#         self.critic_cell = nn.LSTMCell(self.args.hidden_size + self.args.task_emb_size, 1)
#         self.dense_out = nn.Dense(1)
    
#     def forward(self, x, h_t, c_t):
#         self.critic_cell

class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.critic_encoder = Encoder(self.args, 1)

        #self.critic_out1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.critic_out1 = nn.Linear(self.args.hidden_size, 1)
        
        self.critic_out2 = nn.Linear(self.args.hidden_size, 1)


    def forward(self, state, actions_mask=None):
        if actions_mask is not None:
            state_mask = torch.reshape(actions_mask, (actions_mask.shape[0], 1, 1, actions_mask.shape[1]))
        else:
            state_mask = None
        enc_output = self.critic_encoder(state, state_mask)
        
        att_logits = self.critic_out1(enc_output)

        if actions_mask is not None:
            att_logits += (actions_mask.unsqueeze(-1) * -1e9)

        dk = torch.tensor(att_logits.shape[-1], dtype=torch.float32)
        att_logits = att_logits / torch.sqrt(dk)
        
        state_att = F.softmax(att_logits, dim=1)

        # Perform weight combination
        comb_state = torch.matmul(state_att.transpose(2,1), enc_output)

        out = self.critic_out2(F.relu(comb_state))
        out = out.squeeze(-1)
        return out

class Router(nn.Module):
    def __init__(self, args, num_tasks, get_fb_fn, num_fbs):
        super().__init__()
        self.args = args
        self.get_fb_fn = get_fb_fn
        self.num_fbs = num_fbs
        self.num_tasks = num_tasks
        self._valid_fbs_dict = {}

        self.trans_router = TransformerRouter(self.args)

        self.fb_embs = nn.Embedding(self.num_fbs+1, self.args.hidden_size)

        self.task_embs = nn.Embedding(self.num_tasks, self.args.hidden_size)
        
        if self.args.metaworld:
            self.state_encoder = MuJoCoEncoder(self.args)
        else:
            self.state_encoder = ImgEncoder(self.args)
        
        #self.router_cell = nn.LSTMCell(self.args.hidden_size + self.args.task_emb_size + self.args.hidden_size, self.args.hidden_size)

        # self.critic_cell = nn.LSTMCell(self.args.hidden_size + self.args.hidden_size + self.args.hidden_size, self.args.hidden_size)
        # self.critic_out = nn.Linear(self.args.hidden_size, 1)
        self.critic = Critic(self.args)
        #self.att = Attention(self.args)


    def forward(self, imgs, task_id, argmax=False, should_drop_fbs=True):
        batch_size = imgs.shape[0]
        
        # Encode the state
        enc_state = self.state_encoder(imgs)

        # Get task embedding
        task_emb = self.task_embs(task_id)

        builders = [ModelBuilder(self.args) for _ in range(batch_size)]

        # Get fb embeddings
        # if self.args.constraints:
        #     fb_emb_indices = self.args.constraints[int(task_id)].copy()
        # else:
        fb_emb_indices = [list(range(self.num_fbs+1)) for _ in range(batch_size) ]
        # valid_fbs = self._get_valid_fbs(builder, fb_emb_indices)
        

        # Get valid fbs

        
        #fb_embs = self.fb_embs(torch.arange(self.num_fbs+1).cuda()).unsqueeze(0)

        
        # h_t_router = torch.zeros(1, self.args.hidden_size).cuda()
        # c_t_router = torch.zeros(1, self.args.hidden_size).cuda()
        fb_emb_choice = self.fb_embs(torch.tensor([PASS_FB]*batch_size).cuda())
        h_t_critic = torch.zeros(batch_size, self.args.hidden_size).cuda()
        c_t_critic = torch.zeros(batch_size, self.args.hidden_size).cuda()

        critic_outputs = torch.zeros(batch_size, self.args.max_depth, 1).cuda()#[[] for _ in range(batch_size) ]
        actions = [[] for _ in range(batch_size)]
        action_indices = [[] for _ in range(batch_size)]
        # fb_mask = torch.zeros(self.num_fbs+1).cuda()
        failed = False
        cur_loss = None
        action_probs = [[] for _ in range(batch_size)]
        
        att_weights_list = [[] for _ in range(batch_size)]
        state = torch.stack((enc_state, task_emb), 1)
        valid_fbs_ep = [[] for _ in range(batch_size)]
        fb_masks_ep = [[] for _ in range(batch_size)]
        fb_count = torch.zeros(self.num_fbs  + 1)
        max_fbs = 0
        # Build the model architecture 
        for i in range(self.args.max_depth):
            # Update the valid fbs over the batch
            valid_fbs_batch = []
            fb_masks_batch = []
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
                max_fbs = max(max_fbs, len(valid_fbs))

            # Add padding
            valid_fbs_batch = pad_sequence(valid_fbs_batch, batch_first=True)

            fb_embs = self.fb_embs(valid_fbs_batch.cuda())
            fb_masks_batch = create_padding_mask(valid_fbs_batch)        

            fb_probs, att_weights = self.trans_router(state, fb_embs, fb_masks_batch)
            #att_weights_list.append(att_weights)

            # Compute critic output
            # inp = torch.stack((enc_state, task_emb, fb_emb_choice), 1)
            v_t = self.critic(state)
            critic_outputs[:, i] = v_t
            # h_t_critic, c_t_critic = self.critic_cell(inp, (h_t_critic, c_t_critic))
            # v_t = self.critic_out(F.relu(h_t_critic))
            # critic_outputs[:, i] = v_t

            
            # Iterate over choices in a batch
            fb_choices = []
            for b_i in range(batch_size):
                pass
                # Choose a fbs
                if argmax:
                    fb_sample = fb_probs[b_i].argmax()
                    fb_choice = valid_fbs_batch[b_i][fb_sample]
                else:
                    # if np.random.rand() > 0.95:
                    #     fb_choice = torch.tensor([np.random.choice(torch.where(fb_mask == 0)[0].cpu().numpy())]).cuda()
                    # else:
                    # Sample a fb
                    fb_distr = Categorical(fb_probs[b_i])
                    fb_sample = fb_distr.sample()
                    fb_choice = int(valid_fbs_batch[b_i][fb_sample])
                    assert fb_probs[b_i][fb_sample] != 0
                
                # Each action needs to be associated with its index in the fb_probs
                actions[b_i].append(fb_choice)
                action_indices[b_i].append(fb_sample)
                action_probs[b_i].append(fb_probs[b_i])
                fb_choices.append(fb_choice)
                fb_count[fb_choice] += 1
                
                # Remove the fb choice
                if fb_choice != PASS_FB:
                    fb_emb_indices[b_i].remove(fb_choice)

                    cur_fb = self.get_fb_fn(fb_choice)

                    # Verify the builder did not fail
                    assert builders[b_i].attempt_add_fb(cur_fb, i)

            # Update fb input embedding choice for next input
            fb_emb_choice = self.fb_embs(torch.tensor(fb_choices).cuda())

            # Setup next state
            if i +1 < self.args.max_depth:
                chosen_fbs = self.fb_embs(torch.tensor(actions).cuda())
                state = torch.stack((enc_state, task_emb), 1)
                # [batch_size, 3, hidden_size]
                state = torch.cat((state, chosen_fbs), 1)

        return RouterResult(
            failed, 
            action_probs, 
            actions, 
            action_indices, 
            critic_outputs, 
            att_weights_list, 
            valid_fbs_ep,
            fb_count,
            max_fbs) 


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
    
    def predict_step(self, imgs, task_id, valid_fbs, actions, actions_mask):
        """Get action and value for timestep."""
        batch_size = imgs.shape[0]


        # Encode the state
        enc_state = self.state_encoder(imgs)

        # Get task embedding
        task_emb = self.task_embs(task_id)
        
        state = torch.stack((enc_state, task_emb), 1)
        if len(actions) > 0:
            # Embeddings for the current FBs in model
            fb_actions_embs = self.fb_embs(actions)

            state = torch.cat((state, fb_actions_embs), 1)

        val_pred = self.critic(state, actions_mask)
        
        # At the end v_t will contain the final critic output
        # Get FB embeddings
        valid_fbs = valid_fbs
        valid_fb_embs = self.fb_embs(valid_fbs)

        state_mask = torch.reshape(actions_mask, (actions_mask.shape[0], 1, 1, actions_mask.shape[1]))
        fb_masks_batch = create_padding_mask(valid_fbs)

        fb_probs, _ = self.trans_router(state, valid_fb_embs, fb_masks_batch, state_mask)

        return fb_probs, val_pred
            
                
        
