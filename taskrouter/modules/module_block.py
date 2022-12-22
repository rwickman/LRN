import torch
import torch.nn as nn

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