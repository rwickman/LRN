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
    model: nn.Sequential
    critic_outputs: List[torch.Tensor]

PASS_LAYER = 0

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # W for hidden state
        self.hidden_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        
        # W for layer embeddings
        self.layer_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self.V = nn.Linear(self.args.hidden_size, 1)
        
    def forward(self, layer_embs, hidden_state, layer_mask):
        # (batch_size, num_layers, hidden_size)
        layer_trans = self.layer_W(layer_embs)

        # (batch_size, 1, hidden_size)
        hidden_trans = self.hidden_W(hidden_state).unsqueeze(1)

        # (batch_size, num_layers)
        u_i = self.V(F.tanh(layer_trans + hidden_trans)).squeeze(-1)
    
        # Scale the output
        u_i = u_i / torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32))

        # Predictions over layers
        layer_probs = F.softmax(u_i.masked_fill(layer_mask==1, float('-inf')), dim=1)
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
        layer_embs = self.layer_embs(torch.arange(self.num_layers+1).cuda()).unsqueeze(0)

        h_t_router = torch.zeros(1, self.args.hidden_size).cuda()
        c_t_router = torch.zeros(1, self.args.hidden_size).cuda()
        layer_emb_choice = self.layer_embs(torch.tensor([PASS_LAYER]).cuda())
        h_t_critic = torch.zeros(1, self.args.hidden_size).cuda()
        c_t_critic = torch.zeros(1, self.args.hidden_size).cuda()

        critic_outputs = []

        builder = ModelBuilder(self.args)

        actions = []
        layer_mask = torch.zeros(self.num_layers+1).cuda()
        failed = False
        cur_loss = None
        action_probs = []
        added_layers = []
        # print("enc_state", enc_state)
        # print("task_emb", task_emb)
        # Build the model architecture 
        for i in range(self.args.max_depth):
            # Update input
            
            # print("task_emb", task_emb)
            #enc_state = self.state_encoder(state)
            inp = torch.cat((enc_state, task_emb, layer_emb_choice), 1)

            # Router  hidden state, cell memory
            h_t_router, c_t_router = self.router_cell(inp, (h_t_router, c_t_router))

            # Compute critic output 
            h_t_critic, c_t_critic = self.critic_cell(inp, (h_t_critic, c_t_critic))
            v_t = self.critic_out(F.relu(h_t_critic))
            critic_outputs.append(v_t)
            
            # Get attention distribution over layers
            layer_probs = self.att(layer_embs, h_t_router, layer_mask)

            # Choose a layers
            if argmax:
                pass
            else:
                # if np.random.rand() > 0.95:
                #     layer_choice = torch.tensor([np.random.choice(torch.where(layer_mask == 0)[0].cpu().numpy())]).cuda()
                # else:
                # Sample a layer
                layer_distr = Categorical(layer_probs)
                layer_choice = layer_distr.sample()
            
            actions.append(layer_choice.cpu().numpy()[0])
            action_probs.append(layer_probs)

            # Update layer input embedding choice for next input
            layer_emb_choice = self.layer_embs(layer_choice)
            
            # If choice is to PASS, then skip current step
            if layer_choice == PASS_LAYER:
                continue
            
            # Verify did not choose a layer that has already been selected
            assert layer_mask[layer_choice] == 0

            # Add layer choice
            layer_mask[layer_choice] = 1

            cur_layer = self.layer_manager.get_layer((layer_choice-1).cpu().numpy()[0])
            if not builder.attempt_add_layer(cur_layer):
                # layer was not successfully added
                failed = True
                break
            added_layers.append(cur_layer)    
        
        if builder.cur_out_size != self.args.num_classes or builder.get_prev_layer() is None:
            failed = True
        
        return RouterResult(failed, action_probs, actions, builder.build(), critic_outputs), added_layers 
    
    def save_layers(self):
        self.layer_manager.save()
    
    def load_layers(self):
        self.layer_manager.load()