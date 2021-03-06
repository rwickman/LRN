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


@dataclass
class RouterResult:
    failed: bool
    action_probs: List[torch.Tensor]
    actions: List[int]
    # The index in the action probability that corresponds to the action (i.e., layer)
    action_indices: List[int]
    model: nn.Sequential
    critic_outputs: List[torch.Tensor]

PASS_LAYER = 0

class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.depth = self.args.hidden_size // self.args.num_heads

        self.W_q = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.W_k = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.W_v = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        # W for multi-head attention
        self.mha_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def split_heads(self, x):
        x = torch.reshape(x, (1, -1, self.args.num_heads, self.depth))
        return x.transpose(2,1)

    def forward(self, x):
        q = self.split_heads(self.W_q(x))
        k = self.split_heads(self.W_k(x))
        v = self.split_heads(self.W_v(x))

        #print(q.shape)
        qk = torch.matmul(q, k.transpose(3,2))

        dk = torch.tensor(q.shape[-1], dtype=torch.float32)
        att_weights = F.softmax(qk / torch.sqrt(dk), dim=1)
        scaled_att = torch.matmul(att_weights, v)

        # (batch_size, layers+hidden+enc_state, num_heads, depth)
        scaled_att = scaled_att.transpose(2,1)
        
        # Squeeze MHA together for each embedding
        scaled_att = torch.reshape(scaled_att, (1, -1, self.args.hidden_size))

        # Combine the MHA for each embedding
        out = self.mha_W(scaled_att).squeeze(0)

        return out

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

        self.V = nn.Linear(self.args.hidden_size, 1)

        self.mha = MultiheadAttention(args)

        self.point_wise_linear1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.point_wise_linear2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        
        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)

        self.norm1 = nn.GroupNorm(1, self.args.hidden_size)
        self.norm2 = nn.GroupNorm(1, self.args.hidden_size)

        
    def forward(self, layer_embs, hidden_state, layer_mask, enc_state):
        inp = torch.cat((layer_embs, hidden_state, enc_state))

        
        mha_out = self.mha(inp)
        mha_out = self.dropout1(mha_out)
        mha_out  = self.norm1(mha_out + inp)
        
        pw_out = self.point_wise_linear1(mha_out)
        pw_out = self.point_wise_linear2(pw_out)
        pw_out = self.dropout2(pw_out)
        out = self.norm2(pw_out + mha_out)


        # (batch_size, num_layers, hidden_size)
        layer_trans = self.layer_W(out[:-2])
        #print("layer_trans.shape", layer_trans.shape)
        # (batch_size, 1, hidden_size)
        hidden_trans = self.hidden_W(out[-2].unsqueeze(0)).unsqueeze(1)
        #print("hidden_trans.shape", hidden_trans.shape)
        
        state_trans = self.state_W(out[-1].unsqueeze(0)).unsqueeze(1)
        
        # (batch_size, num_layers)
        u_i = self.V(torch.tanh(layer_trans + hidden_trans + state_trans)).squeeze(-1)
    
        # Scale the output
        u_i = u_i / torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32))

        # Predictions over layers
        layer_probs = F.softmax(u_i, dim=1)
        return layer_probs 

class StateEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.conv1 = nn.Conv2d(self.args.in_channels, 32, 3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(1, 32)
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.norm3 = nn.GroupNorm(1, 32)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(self.args.drop_rate)
        
        # Compute output size of last conv layer
        conv2d_out_size = conv2d_out(
            conv2d_out(self.args.img_size, 3, 1, 1),
         3, 1, 1)
        num_units = int(conv2d_out_size**2 * 32)
        self.fc1 = nn.Linear(num_units, self.args.state_size)
        #self.fc2 = nn.Linear(self.args.state_size, self.args.state_size)

    def forward(self, state):
        state = F.relu(self.norm1(self.conv1(state)))
        state = F.relu(self.norm2(self.conv2(state)))
        #state = F.relu(self.norm3(self.conv3(state)))
        state = self.flatten(state)
        state = self.drop1(state)
        # state =self.fc1(state)
        return self.fc1(state)


# class Critic(nn.Module):
#     def __init__(args):
#         self.critic_cell = nn.LSTMCell(self.args.state_size + self.args.task_emb_size, 1)
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

        self.layer_embs = nn.Embedding(self.num_layers+1, self.args.hidden_size)

        self.task_embs = nn.Embedding(self.num_tasks, self.args.task_emb_size)
        
        self.state_encoder = StateEncoder(self.args)
        
        self.router_cell = nn.LSTMCell(self.args.state_size + self.args.task_emb_size + self.args.hidden_size, self.args.hidden_size)

        self.critic_cell = nn.LSTMCell(self.args.state_size + self.args.task_emb_size + self.args.hidden_size, self.args.hidden_size)
        self.critic_out = nn.Linear(self.args.hidden_size, 1)
        
        self.att = Attention(self.args)


    def forward(self, state, task_id, argmax=False):
        # Encode the state
        enc_state = self.state_encoder(state)

        # Get task embedding
        task_emb = self.task_embs(task_id)

        # Get layer embeddings
        layer_emb_indices = list(range(self.num_layers+1))
        #layer_embs = self.layer_embs(torch.arange(self.num_layers+1).cuda()).unsqueeze(0)

        h_t_router = torch.zeros(1, self.args.hidden_size).cuda()
        c_t_router = torch.zeros(1, self.args.hidden_size).cuda()
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

        # Build the model architecture 
        for i in range(self.args.max_depth):
            # Update inputs
            inp = torch.cat((enc_state, task_emb, layer_emb_choice), 1)
            layer_embs = self.layer_embs(torch.tensor(layer_emb_indices).cuda())
            
            # Router  hidden state, cell memory
            h_t_router, c_t_router = self.router_cell(inp, (h_t_router, c_t_router))

            # Compute critic output 
            h_t_critic, c_t_critic = self.critic_cell(inp, (h_t_critic, c_t_critic))
            v_t = self.critic_out(F.relu(h_t_critic))
            critic_outputs.append(v_t)
            
            # Get attention distribution over layers
            layer_probs = self.att(layer_embs, h_t_router, layer_mask, enc_state)

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

            # If choice is to PASS, then skip current step
            if layer_choice == PASS_LAYER:
                continue
            
            # Verify did not choose a layer that has already been selected
            assert layer_choice in layer_emb_indices

            # Add layer choice
            #layer_mask[layer_choice] = 1

            # Remove the layer choice
            layer_emb_indices.remove(layer_choice)

            cur_layer = self.layer_manager.get_layer(layer_choice-1)
            if not builder.attempt_add_layer(cur_layer):
                # layer was not successfully added
                failed = True
                break 
        
        if builder.cur_out_size != self.args.num_classes or builder.get_prev_layer() is None:
            failed = True
        
        return RouterResult(failed, action_probs, actions, action_indices, builder.build(), critic_outputs) 
    
    def save_layers(self):
        self.layer_manager.save()
    
    def load_layers(self):
        self.layer_manager.load()
