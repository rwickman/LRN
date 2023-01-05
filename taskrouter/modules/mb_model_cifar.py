import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from dataclasses import dataclass
from .module_block import ModuleBlock




class SimpleConvNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.relu(y)
        y = self.maxpool(y)
        return y

# def default_group_3x3(in_channels, out_channels, stride=1):
#     mbs = []

#     # 3x3 -> 3x3
#     for i in range(4):
#         mbs.append(
#                 ModuleBlock(
#                     nn.Sequential(
#                         nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
#                         nn.GroupNorm(4, out_channels), 
#                         nn.ReLU(),
#                         nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride),
#                         nn.GroupNorm(4, out_channels), 
#                         nn.ReLU()
#                     )
#                 )
#         )
#     return mbs


def default_group(num_dup, in_channels, out_channels, stride=1, only_conv=False):
    mbs = []
    for _ in range(num_dup):
        mbs.append(
            ModuleBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2))))
            #     nn.BatchNorm2d(out_channels),
            #     nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(2))
            # ))    


    return mbs

def default_group_linear(num_dup, inp_size, out_size, use_act=True):
    mbs = []
    for _ in range(num_dup):
        if use_act:
            mbs.append(ModuleBlock(
                nn.Linear(inp_size, out_size),
                nn.ReLU(inplace=True)
            ))
        else:
            mbs.append(ModuleBlock(
                nn.Linear(inp_size, out_size)
            ))
    
    return mbs

# 4 3x3 conv layer with 32 filters followed by batchnorm and ReLU
# 3 FC layers of 127 units each

class ModuleBlockModelCIFAR(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config

        self._init_layers()

    def _init_layers(self):
        mbs = []

        # Add the routed conv layers
        # for i, stride in enumerate(self.config.strides):
        #     if i == 0:
        #         in_channels = 3
        #     else:
        #         in_channels = 32

        #     mbs.extend(
        #         default_group(self.config.unique_hidden_modules, in_channels, 32, stride=stride))
        self.shared_inp = nn.Sequential(
            SimpleConvNetBlock(self.config.in_channels, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            SimpleConvNetBlock(32, 32, 3),
            nn.Flatten())

        # Add the routed FC layers
        mbs.extend(
            default_group_linear(self.config.unique_hidden_modules, self.config.conv_out_size, 128))
        mbs.extend(
            default_group_linear(self.config.unique_hidden_modules, 128, 128))
        mbs.extend(
            default_group_linear(self.config.unique_hidden_modules, 128, self.config.out_size, use_act=False))
        
        self._mbs = nn.ModuleList(mbs)

    def get_mb(self, idx):
        return self._mbs[idx]
    
    def forward(self, x, router_outs, task_ids):
        """Forward the batch of inputs through the Function Blocks.

        Args:
            x: batch of inputs
            router_outs: batch of router_outputs.
        
        Returns:
            a Tensor of outputs
        """
        batch_size = x.shape[0]
        #y = torch.zeros(batch_size, self._max_out_size, device=self.config.device)
        y = torch.tensor([-1e9], device=self.config.device).repeat(batch_size, self.config.out_size)
        x = self.shared_inp(x)
        for i in range(batch_size):
            
            cur_x = x[i:i+1]
            actions = router_outs[i].nonzero()[:, 1].flatten()
            #print("actions", task_ids[i], actions)
            for t, a_t in enumerate(actions):    
                #cur_x = cur_x.view(cur_x.shape[0], self.config.conv_out_size)
                cur_x = self._mbs[a_t](cur_x) * router_outs[i][t][a_t]
                 

            #cur_x = cur_x.view(1, -1)
            y[i] = cur_x

        return y

    def get_num_mbs(self):
        return len(self._mbs)





