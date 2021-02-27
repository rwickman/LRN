import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model_util import *

PASS_LAYER = 0



class Attention(nn.Module):
    def __init__(self, layer_emb_size, hidden_size):
        super().__init__()
        # W for hidden state
        self.hidden_W = nn.Linear(layer_emb_size, hidden_size)
        
        # W for layer embeddings
        self.layer_W = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        
    def forward(self, layer_embs, hidden_state):

        # (batch_size, num_layers, hidden_size)
        layer_trans = self.layer_W(hidden_state)

        # (batch_size, 1, hidden_size)
        hidden_trans = self.hidden_W(hidden_state).unsqueeze(1)

        # (batch_size, num_layers)
        u_i = self.V(F.tanh(layer_trans + hidden_trans)).squeeze(-1)
        
        # TODO: Add mask for layer already removed to set els in u_i to 0

        # Predictions over layers
        layer_probs = F.softmax(u_i)
        return layer_probs 

class StateEncoder(nn.Module):
    def __init__(self, img_size, state_size, drop_rate=0.2):
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(8, 32, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(drop_rate)
        
        # Compute output size of last conv layer
        conv2d_out_size = conv2d_out(
            conv2d_out(img_size, 3, 2),
         3, 2)
        
        self.fc1 = nn.Linear(conv2d_out_size, state_size)
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = drop1(state)
        return F.relu(self.fc1(state))


# Should you keep inputing state and task emb or not...?

class Router(nn.Module):
    def __init__(self, img_size, state_size, num_layers, layer_emb_size, task_emb_size, hidden_size, max_depth, batch_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.layer_embs = nn.Embedding(num_layers+1, layer_emb_size)

        self.task_embs = nn.Embedding(num_tasks, task_emb_size)
        
        self.state_encoder = StateEncoder(img_size, state_size)
        
        self.router_cell = nn.LSTMCell(state_dim + task_dim, hidden_size)
        self.att = Attention(layer_emb_size, hidden_size)

    def forward(self, state, task_id, argmax=False):
        # Encode the state
        enc_state = self.state_encoder(self.state)

        # Get task embedding
        task_emb = self.task_embs(task_id)

        # Get layer embeddings
        layer_embs = self.layer_embs(list(range(num_layers+1)))

        inp = torch.cat((enc_state, task_emb), 1) 
        h_t = torch.zeros(self.batch_size, self.hidden_size)
        c_t = torch.zeros(self.batch_size, self.hidden_size)
        # hidden state, cell memory
        for i in range(max_depth):
            h_t, c_t = self.router_cell(inp, (h_t, c_t))

            # Get attention distribution over layers
            layer_probs = self.att(h_t, layer_embs)

            # Choose a layers
            if argmax:
                pass
            else:
                # Sample a layer
                layer_distr = Categorical(layer_probs)
                layer_choice = layer_distr.sample()
            
            # If choose is to PASS then end building model
            if layer_choice == PASS_LAYER:
                break
            
            # TODO: Check if layers dimensions match

            # TODO: 
            


            





        


        
    
