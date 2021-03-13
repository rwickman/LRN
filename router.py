import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass
from typing import List

from model_util import *
from layer_manager import LayerManager
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
        
        #print("BEFORE split_heads", x.shape)
        # (batch size, num tokens,  num heads, depth)
        x = torch.reshape(x, (1, -1, self.args.num_heads, self.depth))
        
        # (batch size, num heads, num tokens, depth)
        return x.transpose(2,1)

    def forward(self, q, k, v):
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))

        qk = torch.matmul(q, k.transpose(3,2))

        dk = torch.tensor(q.shape[-1], dtype=torch.float32)
        #print("dk", dk)
        #print(qk.shape)
        att_weights = F.softmax(qk / torch.sqrt(dk), dim=-1)
        # print(att_weights)
        # print("att_weights.shape", att_weights.shape)
        # print("v.shape", v.shape)
        scaled_att = torch.matmul(att_weights, v)
        #print("scaled_att.shape", scaled_att.shape)
        # (batch_size, layers+hidden+enc_state, num_heads, depth)
        scaled_att = scaled_att.transpose(2,1)
        #print("scaled_att..transpose(2,1).shape", scaled_att.shape)
        # Squeeze MHA together for each embedding
        scaled_att = torch.reshape(scaled_att, (1, -1, self.args.hidden_size))
        #print("torch.reshape(scaled_att.shape", scaled_att.shape)
        # Combine the MHA for each embedding
        out = self.mha_W(scaled_att).squeeze(0)
        #print(out.shape)

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
        
    def forward(self, x):
        # Perform MHA attention
        att_out, _ = self.mha(x, x, x)
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
    
    def forward(self, x, enc_output):
        # Preform self attention over available layers
        att_out, att_weights1 = self.mha1(x, x, x)
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
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.enc_layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(self.args.num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)
    
        self.pos_enc = positional_encoding(self.args.max_pos_enc, self.args.hidden_size)

    def forward(self, x):
        """
            x: task emb, example emb, used layers embs
        """
        x *= torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32)).cuda()
        x += self.pos_enc[:, :x.shape[1], :]
        x = self.dropout(x)

        # Run through all encoder layer
        for enc_layer in self.enc_layers:
            x = enc_layer(x)
        
        # Return output of last encoder layer
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.pos_enc = positional_encoding(self.args.max_pos_enc, self.args.hidden_size)

        self.dec_layers = nn.ModuleList([DecoderLayer(self.args) for _ in range(self.args.num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)
    
    def forward(self, x, enc_output):
        """Args:
            x: Tensor of available layer embeddings.
            enc_output: Tensor of output of last encoder layer.
        """
        num_layers = x.shape[1]
        x *= torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32))
        x += self.pos_enc[:, :num_layers, :]
        x = self.dropout(x)
        
        att_weights = {}

        # Run through all decoder layers
        for i, dec_layer in enumerate(self.dec_layers):
            x, att_weights1, att_weights2 = dec_layer(x, enc_output)

            att_weights['decoder_layer{}_weights1'.format(i+1)] = att_weights1
            att_weights['decoder_layer{}_weights2'.format(i+1)] = att_weights2

        return x, att_weights


class TransformerRouter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.args)

        # Maps each decorder embedding to 1 value
        self.actor_out1 = nn.Linear(self.args.hidden_size, 16)
        self.actor_out2 = nn.Linear(16, 1)
        

    def forward(self, state, layer_embs):
        enc_output = self.encoder(state)
        
        dec_output, att_weights = self.decoder(layer_embs, enc_output)
        
        # batch size, len(layer_embs)
        #print("DEC OUT shape", dec_output.shape)
        actor_out = self.actor_out1(dec_output)
        actor_out = self.actor_out2(actor_out).squeeze(-1)


        layer_probs = F.softmax(actor_out, dim=-1)
        return layer_probs, att_weights
        


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # W for hidden state
        self.hidden_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        
        # W for encoded state
        self.state_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        # W for layer embeddings
        self.layer_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

         # W for layer embeddings
        self.task_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self.V = nn.Linear(self.args.hidden_size, 1)

        self.mha = MultiHeadAttention(args)

        self.norm1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        self.pw_ffn = PointWiseFFN(self.args)

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)


        
    def forward(self, layer_embs, hidden_state, enc_state, task_emb):
        inp = torch.cat((layer_embs.squeeze(0), hidden_state, enc_state, task_emb))
        mha_out, _ = self.mha(inp, inp, inp)
        mha_out = self.dropout1(mha_out)
        mha_out = self.norm1(mha_out + inp)
        
        pw_out = self.pw_ffn(mha_out)
        pw_out = self.dropout2(pw_out)
        out = self.norm2(pw_out + mha_out)


        # (batch_size, num_layers, hidden_size)
        layer_trans = self.layer_W(out[:-3])
        #print("layer_trans.shape", layer_trans.shape)
        
        # (batch_size, hidden_size)
        hidden_trans = self.hidden_W(out[-3].unsqueeze(0)).unsqueeze(1)
        #print("hidden_trans.shape", hidden_trans.shape)
        
        state_trans = self.state_W(out[-2].unsqueeze(0)).unsqueeze(1)

        task_trans = self.task_W(out[-1].unsqueeze(0)).unsqueeze(1)

        # (batch_size, num_layers)
        u_i = self.V(torch.tanh(layer_trans + hidden_trans + state_trans + task_trans)).squeeze(-1)
    
        # Scale the output
        u_i = u_i / torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32))

        # Predictions over layers
        layer_probs = F.softmax(u_i, dim=-1)
        return layer_probs

class StateEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.conv1 = nn.Conv2d(self.args.in_channels, 32, 3, stride=1, padding=1)
        #self.norm1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        #self.norm2 = nn.GroupNorm(1, 32)
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.norm3 = nn.LayerNorm(1, 32)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(self.args.drop_rate)
        
        # Compute output size of last conv layer
        conv2d_out_size = conv2d_out(
            conv2d_out(self.args.img_size, 3, 1, 1),
         3, 1, 1)
        num_units = int(conv2d_out_size**2 * 32)
        self.fc1 = nn.Linear(num_units, self.args.hidden_size)
        #self.fc2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def forward(self, state):
        #state = F.relu(self.norm1(self.conv1(state)))
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
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



class Router(nn.Module):
    def __init__(self, args, num_tasks):
        super().__init__()
        self.args = args
        self.layer_manager = LayerManager(self.args)
        self.num_layers = self.layer_manager.get_num_layers()
        self.num_tasks = num_tasks

        self.trans_router = TransformerRouter(self.args)

        self.layer_embs = nn.Embedding(self.num_layers+1, self.args.hidden_size)

        self.task_embs = nn.Embedding(self.num_tasks, self.args.hidden_size)
        
        self.state_encoder = StateEncoder(self.args)
        
        #self.router_cell = nn.LSTMCell(self.args.hidden_size + self.args.task_emb_size + self.args.hidden_size, self.args.hidden_size)

        self.critic_cell = nn.LSTMCell(self.args.hidden_size + self.args.hidden_size + self.args.hidden_size, self.args.hidden_size)
        self.critic_out = nn.Linear(self.args.hidden_size, 1)
        
        #self.att = Attention(self.args)


    def forward(self, state, task_id, argmax=False):
        # Encode the state
        enc_state = self.state_encoder(state)

        # Get task embedding
        task_emb = self.task_embs(task_id)

        # Get layer embeddings
        # if self.args.constraints:
        #     layer_emb_indices = self.args.constraints[int(task_id)].copy()
        # else:
        layer_emb_indices = list(range(self.num_layers+1))
        
        #layer_embs = self.layer_embs(torch.arange(self.num_layers+1).cuda()).unsqueeze(0)

        
        # h_t_router = torch.zeros(1, self.args.hidden_size).cuda()
        # c_t_router = torch.zeros(1, self.args.hidden_size).cuda()
        layer_emb_choice = self.layer_embs(torch.tensor([PASS_LAYER]).cuda())
        h_t_critic = torch.zeros(1, self.args.hidden_size).cuda()
        c_t_critic = torch.zeros(1, self.args.hidden_size).cuda()

        critic_outputs = []

        builder = ModelBuilder(self.args)

        actions = []
        action_indices = []
        layer_mask = torch.zeros(self.num_layers+1).cuda()
        failed = False
        cur_loss = None
        action_probs = []
        
        att_weights_list = []
        state = torch.cat((enc_state, task_emb)).unsqueeze(0)

        # Build the model architecture 
        for i in range(self.args.max_depth):
            inp = torch.cat((enc_state, task_emb, layer_emb_choice), 1)
            # Update inputs
            layer_embs = self.layer_embs(torch.tensor(layer_emb_indices).cuda()).unsqueeze(0)
            
            #h_t_router, c_t_router = self.critic_cell(inp, (h_t_router, c_t_router))
            #layer_probs = self.att(layer_embs, h_t_router, enc_state, task_emb)
            layer_probs, att_weights = self.trans_router(state, layer_embs)
            att_weights_list.append(att_weights)

            # Compute critic output 
            h_t_critic, c_t_critic = self.critic_cell(inp, (h_t_critic, c_t_critic))
            v_t = self.critic_out(F.relu(h_t_critic))
            critic_outputs.append(v_t.squeeze(0))

            # Choose a layers
            if argmax:
                pass
            else:
                # if np.random.rand() > 0.95:
                #     layer_choice = torch.tensor([np.random.choice(torch.where(layer_mask == 0)[0].cpu().numpy())]).cuda()
                # else:
                # Sample a layer
                layer_distr = Categorical(layer_probs)
                layer_sample = layer_distr.sample()
                layer_choice = layer_emb_indices[layer_sample]
            
            # Each action needs to be associated with its index in the layer_probs
            actions.append(layer_choice)
            action_indices.append(layer_sample)
            action_probs.append(layer_probs)

            # Update layer input embedding choice for next input
            layer_emb_choice = self.layer_embs(torch.tensor([layer_choice]).cuda())

            # Setup next state
            if i +1 < self.args.max_depth:
                chosen_layers = self.layer_embs(torch.tensor(actions).cuda())
                # print("actions", actions)
                # print("chosen_layers", chosen_layers)
                # print("chosen_layers.shape", chosen_layers.shape) 
                state = torch.cat((enc_state, task_emb, chosen_layers)).unsqueeze(0)
                #print("state.shape", state.shape)

            # Check if constraints are violated
            if self.args.constraints and int(layer_choice) not in self.args.constraints[int(task_id)]:
                failed = True
                break 

            # If choice is to PASS, then skip current step
            if layer_choice == PASS_LAYER:
                continue
            
            # Verify did not choose a layer that has already been selected
            assert layer_choice in layer_emb_indices

            # Add layer choice
            #layer_mask[layer_choice] = 1

            cur_layer = self.layer_manager.get_layer(layer_choice)
            if not builder.attempt_add_layer(cur_layer):
                # layer was not successfully added
                failed = True
                break 
        
            # Remove the layer choice
            layer_emb_indices.remove(layer_choice)

        if builder.cur_out_size != self.args.num_classes or builder.get_prev_layer() is None:
            failed = True
        
        

        return RouterResult(failed, action_probs, actions, action_indices, builder.build(), critic_outputs, att_weights_list) 
    

    def build_fixed(self, layer_ids):
        """Build a model given the layer IDs."""
        builder = ModelBuilder(self.args)
        failed = False
        for layer_id in layer_ids:
            cur_layer = self.layer_manager.get_layer(layer_id)
            if not builder.attempt_add_layer(cur_layer):
                failed = True
                break
        return failed, builder.build()
            

    def save_layers(self):
        self.layer_manager.save()
    
    def load_layers(self):
        self.layer_manager.load()
