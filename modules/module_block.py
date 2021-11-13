import torch
import torch.nn as nn
from configs import ale_config

class ModuleBlock(nn.Module):
    def __init__(self, layer, norm=None, act=None):
        super().__init__()
        self.layer = layer  
        self.norm = norm
        self.act = act
    
    def forward(self, x):
        x = self.layer(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.act is not None:
            x = self.act(x)

        return x



class LSTMBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        # Save previous states
        self.save_states = True

        self.reset()
    
    def reset(self):        
        self.hidden_states = []
        self.cell_states = []
        # Dictionary for remembering what timestep each module was called
        self.step_to_state_idx = {}
    
    def add_zero(self):
        self.hidden_states.append(torch.zeros(1, self.layer.hidden_size, device=ale_config.device))
        self.cell_states.append(torch.zeros(1, self.layer.hidden_size, device=ale_config.device))


    def forward(self, x, idx=-1):
        if self.save_states:
            # Record the timestep that this was called
            self.step_to_state_idx[idx] = len(self.hidden_states)
            state_idx = -1
        else:
            # Must be training so get the correct state index
            state_idx = self.step_to_state_idx[idx]

        cur_hidden_state = self.hidden_states[state_idx]
        cur_cell_state = self.cell_states[state_idx]

        next_hidden_state, next_cell_state = self.layer(x, (cur_hidden_state.detach(), cur_cell_state.detach()))
        
        if self.save_states:
            self.hidden_states.append(next_hidden_state)
            self.cell_states.append(next_cell_state)
        
        return next_hidden_state
