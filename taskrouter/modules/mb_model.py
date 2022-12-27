import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from dataclasses import dataclass
from .module_block import ModuleBlock




def default_group_3x3(in_channels, out_channels, stride=1):
    mbs = []

    # 3x3 -> 3x3
    for i in range(4):
        mbs.append(
                ModuleBlock(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
                        nn.GroupNorm(4, out_channels), 
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride),
                        nn.GroupNorm(4, out_channels), 
                        nn.ReLU()
                    )
                )
        )
    return mbs


def default_group(in_channels, out_channels, stride=1, only_conv=False):
    mbs = []

    # 3x3 -> 3x3
    mbs.append(
            ModuleBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU()
                )
            )
    )


    # 5x5 -> 5x5
    mbs.append(
            ModuleBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 5, padding=2, stride=1),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 5, padding=2, stride=stride),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU()
                )
            )
    )


    # 7x7 -> 7x7
    mbs.append(
            ModuleBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 7, padding=3, stride=1),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 7, padding=3, stride=stride),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU()
                )
            )
    )

    # 1x7 -> 7x1
    mbs.append(
            ModuleBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1, 7), padding=(0,3), stride=1),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3,0), stride=stride),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU()
                )
            )
    )
    if not only_conv:
        mbs.append(
            ModuleBlock(
                nn.MaxPool2d(3, stride=stride, padding=1)
            )
        )

        mbs.append(
            ModuleBlock(
                nn.AvgPool2d(3, stride=stride, padding=1)
            )
        )

        # "Identity" layer used
        if stride == 1:
            mbs.append(
                ModuleBlock(
                    nn.Identity()
                )
            )
        else:
            mbs.append(
                ModuleBlock(
                    nn.Conv2d(out_channels, out_channels, 1, stride=stride)
                )
            )

    return mbs


class ModuleBlockModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config

        self._max_out_size = max(config.out_sizes)
        self._init_layers()

    def _init_layers(self):
        out_layers = []
        mbs = []

        # Input layer
        self.shared_inp = ModuleBlock(
            nn.Conv2d(1, 48, 1),
            nn.GroupNorm(4, 48),
            nn.ReLU()
        )

        for stride in (self.config.strides):
            if self.args.only_3x3:
                mbs.extend(default_group_3x3(48, 48, stride=stride))
            else:
                mbs.extend(default_group(48, 48, stride=stride, only_conv=self.args.only_conv))


        # Add output layers
        out_in_size = self.config.num_filters * 4**2
        for i in range(len(self.config.out_sizes)):
            out_layers.append(
                nn.Linear(out_in_size, self.config.out_sizes[i])
            )
        
        self._dropout = nn.Dropout(self.args.drop_rate)

        self._mbs = nn.ModuleList(mbs)
        self._group_norm_out = nn.GroupNorm(4, num_channels=48)
        self._act_out = nn.ReLU()

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
        #y = torch.zeros(batch_size, self._max_out_size, device=self.config.device)
        y = torch.tensor([-1e9], device=self.config.device).repeat(batch_size, self._max_out_size)
        
        for i in range(batch_size):
            cur_x = x[i:i+1]
            actions = router_outs[i].nonzero()[:, 1].flatten()
            #print("actions", task_ids[i], actions)
            cur_x = self.shared_inp(cur_x)
            for t, a_t in enumerate(actions):
                cur_x = self._mbs[a_t](cur_x) * router_outs[i][t][a_t]

            if not self.args.only_conv:
                cur_x = self._group_norm_out(cur_x)
                cur_x = self._act_out(cur_x)

            cur_x = self._dropout(cur_x)
            cur_x = cur_x.view(1, -1)
            y[i, :self.config.out_sizes[task_ids[i, 0]]] = self._out_layers[task_ids[i]](cur_x)

        return y

    def get_num_mbs(self):
        return len(self._mbs)





