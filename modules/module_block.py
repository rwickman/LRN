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
        self.first_call = True
        # self.reset()
    
    def reset(self):        
        """Called at the beginning of an a batch."""
        if not self.first_call:
            self.hidden_state = self.hidden_state_buffer[-1].to(ale_config.device)
            self.cell_state = self.cell_state_buffer[-1].to(ale_config.device)
        print("\nCLEARING BUFFERS\n")
        self.hidden_state_buffer = []
        self.cell_state_buffer = []
        # Dictionary for remembering what timestep each module was called
        self.step_to_state_idx = {}
        self.first_call = False
 
    def add_zero(self):
        """Called at the beginning of an episode to set initial states."""
        # self.hidden_state_buffer.append(torch.zeros(1, self.layer.hidden_size).detach)
        # self.cell_state_buffer.append(torch.zeros(1, self.layer.hidden_size).detach())
        
        self.hidden_state = torch.zeros(1, self.layer.hidden_size, device=ale_config.device)
        self.cell_state = torch.zeros(1, self.layer.hidden_size, device=ale_config.device)

    def init_states(self, idx):
        """Initialize the hidden and cell states to a saved value."""
        #print("self.step_to_state_idx", self.step_to_state_idx)
        
        state_idx = self.step_to_state_idx[idx]
        self.hidden_state = self.hidden_state_buffer[state_idx].clone().to(ale_config.device)
        self.cell_state = self.cell_state_buffer[state_idx].clone().to(ale_config.device)


    def forward(self, x, idx=None):

        if self.save_states:
            # Record the timestep that this was called
            self.step_to_state_idx[idx] = len(self.hidden_state_buffer)
            # Save states
            self.hidden_state_buffer.append(self.hidden_state.detach().clone().cpu())
            self.cell_state_buffer.append(self.cell_state.detach().clone().cpu())
        else:
            # Update current hidden_state
            state_idx = self.step_to_state_idx[idx]
            self.hidden_state_buffer[state_idx] = self.hidden_state.detach().clone().cpu()
            self.cell_state_buffer[state_idx] = self.cell_state.detach().clone().cpu()

        
        self.hidden_state, self.cell_state = self.layer(x, (self.hidden_state, self.cell_state))


        return self.hidden_state
