import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from .module_block import ModuleBlock




def default_group(in_channels, out_channels, stride=1):
    mbs = []

    mbs.append(
            ModuleBlock(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
                nn.GroupNorm(4, out_channels), 
                nn.LeakyReLU()
            )
    )

    mbs.append(
        ModuleBlock(
            nn.Conv2d(in_channels, out_channels, 5, padding=2, stride=stride),
            nn.GroupNorm(4, out_channels), 
            nn.LeakyReLU()
            )
    )

    mbs.append(
        ModuleBlock(
            nn.Conv2d(in_channels, out_channels, 7, padding=3, stride=stride),
            nn.GroupNorm(4, out_channels), 
            nn.LeakyReLU()
            )
    )

    return mbs


class ModuleBlockModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._max_out_size = max(config.out_sizes)
        self._init_layers() 

        #self.create_mb_probs()

    def _init_layers(self):
        out_layers = []
        mbs = []

        # Input layer
        inp_mbs = default_group(self.config.in_channels, self.config.num_filters)
        mbs.extend(inp_mbs)


        # Hidden layers
        for i in range(self.config.num_hidden):        
            hidden_mbs = default_group(self.config.num_filters, self.config.num_filters)
            mbs.extend(hidden_mbs)
            

            mbs.append(
                ModuleBlock(
                    nn.MaxPool2d(3, stride=1, padding=1)
                )
            )

            mbs.append(
                ModuleBlock(
                    nn.AvgPool2d(3, stride=1, padding=1)
                )
            )

            mbs.append(
                ModuleBlock(
                    nn.Identity()
                )
            )
        
        # Hidden layers that add more filters
        for i in range(self.config.num_downsample-1):
            down_mbs = default_group(self.config.num_filters * (i+1), self.config.num_filters * (i+2), 2)
            mbs.extend(down_mbs)
        
        mbs.append(
            ModuleBlock(
                nn.MaxPool2d(3)
            )
        )

        mbs.append(
            ModuleBlock(
                nn.AvgPool2d(3)
            )
        )



        # Add output layers
        out_in_size = 128 * 4 * 4#self.config.num_filters * (self.config.num_downsample + 1)  * round(self.config.img_size / 2 ** self.config.num_downsample) ** 2
        
        for i in range(len(self.config.out_sizes)):
            out_layers.append(
                nn.Linear(out_in_size, self.config.out_sizes[i])
            )
        
        self._mbs = nn.ModuleList(mbs)
        self._out_layers = nn.ModuleList(out_layers)

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
        y = torch.zeros(batch_size, self._max_out_size, device=self.config.device)

        for i in range(batch_size):
            cur_x = x[i:i+1]
            actions = router_outs[i].nonzero()[:, 1].flatten()
            for t, a_t in enumerate(actions):    
                cur_x = self._mbs[a_t](cur_x) * router_outs[i][t][a_t]
            
            cur_x = cur_x.view(1, -1)
            
            y[i, :self.config.out_sizes[task_ids[i, 0]]] = self._out_layers[task_ids[i]](cur_x)

        return y

    def get_num_mbs(self):
        return len(self._mbs)





