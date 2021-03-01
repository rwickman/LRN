import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from model_util import *
from layer_manager import LayerManager
from model_builder import ModelBuilder

PASS_LAYER = 0

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # W for hidden state
        self.hidden_W = nn.Linear(self.args.layer_emb_size, self.args.hidden_size)
        
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

        # TODO: Add mask for layer already removed to set els in u_i to 0

        # Predictions over layers
        layer_probs = F.softmax(u_i.masked_fill(layer_mask==1, float('-inf')), dim=1)
        return layer_probs 

class StateEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.conv1 = nn.Conv2d(self.args.in_channels, 32, 3, stride=1, padding=1)
        self.bn1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(1, 32)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(self.args.drop_rate)
        
        # Compute output size of last conv layer
        conv2d_out_size = conv2d_out(
            conv2d_out(self.args.img_size, 3, 1, 1),
         3, 1, 1)
        num_units = int(conv2d_out_size**2 * 32)
        self.fc1 = nn.Linear(num_units, self.args.state_size)

    def forward(self, state):
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = self.flatten(state)
        state = self.drop1(state)
        return self.fc1(state)


# Should you keep inputing state and task emb or not...?

class Router(nn.Module):
    def __init__(self, args, num_tasks):
        super().__init__()
        self.args = args
        self.layer_manager = LayerManager(self.args)
        self.num_layers = self.layer_manager.get_num_layers()
        self.num_tasks = num_tasks

        self.layer_embs = nn.Embedding(self.num_layers+1, self.args.layer_emb_size)

        self.task_embs = nn.Embedding(self.num_tasks, self.args.task_emb_size)
        
        self.state_encoder = StateEncoder(self.args)
        
        self.router_cell = nn.LSTMCell(self.args.state_size + self.args.task_emb_size, self.args.hidden_size)
        self.att = Attention(self.args)
        

    def forward(self, state, task_id, argmax=False):
        # Encode the state
        enc_state = self.state_encoder(state)

        # Get task embedding
        task_emb = self.task_embs(task_id)

        # Get layer embeddings
        layer_embs = self.layer_embs(torch.arange(self.num_layers+1)).unsqueeze(0)

        inp = torch.cat((enc_state, task_emb), 1)

        h_t = torch.zeros(1, self.args.hidden_size)
        c_t = torch.zeros(1, self.args.hidden_size)
        out_size = (self.args.img_size, self.args.img_size)
        

        builder = ModelBuilder(self.args)

        actions = []
        layer_mask = torch.zeros(self.num_layers+1)
        failed = False
        cur_loss = None
        action_probs = []

        # Build the model architecture 
        for i in range(self.args.max_depth):
            # hidden state, cell memory
            h_t, c_t = self.router_cell(inp, (h_t, c_t))

            # Get attention distribution over layers
            layer_probs = self.att(layer_embs, h_t, layer_mask)

            # Choose a layers
            if argmax:
                pass
            else:
                # Sample a layer
                layer_distr = Categorical(layer_probs)
                layer_choice = layer_distr.sample()
            
            actions.append(layer_choice.numpy()[0])
            action_probs.append(layer_distr.log_prob(layer_choice))

            # If choice is to PASS, then skip current step
            if layer_choice == PASS_LAYER:
                continue
            
            # Verify did not choose a layer that has already been selected
            assert layer_mask[layer_choice] == 0

            # Add layer choice
            layer_mask[layer_choice] = 1

            cur_layer = self.layer_manager.get_layer((layer_choice-1).numpy()[0])
            if not builder.attempt_add_layer(cur_layer):
                # layer was not successfully added
                failed = True
                break
        
        if builder.cur_out_size != self.args.num_classes:
            failed = True
        
        return failed, action_probs, actions, builder.build()