import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from model_util import PASS_FB

class FunctionBlock(nn.Module):
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

class FBModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._num_filters = 32
        self._filter_size = 3
        self._hidden_size = 128
        self._conv_to_fc_in = (self.args.img_size//2)**2*self._num_filters
        self.max_pool = nn.MaxPool2d(2)
        self._init_layers()
        self.create_fb_probs()

    def _init_layers(self):
        fbs = []
        for i in range(self.args.num_dup_layers):
            # Input layer
            fbs.append(
                    FunctionBlock(
                        nn.Conv2d(self.args.in_channels, self._num_filters, self._filter_size, stride=1, padding=1),
                        nn.InstanceNorm2d(self._num_filters, affine=True, eps=1e-6), 
                        nn.ReLU()
                    )
                )

            # Hidden layers
            fbs.append(
                FunctionBlock(
                    nn.Conv2d(self._num_filters, self._num_filters, self._filter_size, stride=1, padding=1),
                    nn.InstanceNorm2d(self._num_filters, affine=True, eps=1e-6),
                    nn.ReLU()
                )
            )

            fbs.append(
                FunctionBlock(
                    nn.Linear(self._conv_to_fc_in, self._hidden_size),
                    nn.LayerNorm(self._hidden_size, eps=1e-6),
                    nn.ReLU()
                )
            )
            
            fbs.append(
                FunctionBlock(
                    nn.Linear(self._hidden_size, self._hidden_size),
                    nn.LayerNorm(self._hidden_size, eps=1e-6),
                    nn.ReLU()
                )
            )

            # Add last layers
            fbs.append(
                FunctionBlock(
                    nn.Linear(self._hidden_size, self.args.num_classes)
                )
            )

            fbs.append(
                FunctionBlock(
                    nn.Linear(self._conv_to_fc_in, self.args.num_classes)
                )
            )
        
        self._fbs = nn.ModuleList(fbs)
        
        
    def create_fb_probs(self):
        self.fb_prob_dict = {}
        d = self.args.num_dup_layers
        
        if self.args.max_depth == 4:
            self.fb_prob_dict["E0"] = 1/d

            self.fb_prob_dict["E1"] = 1/(2*d) + (d-1)/(2*d) * (1/(2*d-1))
            self.fb_prob_dict["E2"] = 1/(2*d) + d/(2*d) * (1/(2*d-1))
            self.fb_prob_dict["E3"] = 1/2 * (1/d)
            self.fb_prob_dict["E4"] = 1/2 * (d/(2*d-1)) * (1/d) + 1/2 * 1 * (1/d)
            self.fb_prob_dict["E5"] = 1/2 * ((d-1) / (2*d - 1)) * (1/d)
        elif self.args.max_depth == 5:
            self.fb_prob_dict["E0"] = 1/d
            self.fb_prob_dict["E1"] = 1/(2*d) + (d-1)/(2*d) * (1/(2*d-1)) + ((d-1)/(2*d)) * ((d-2)/(2*d-1)) * (1/(2*d-2))
            self.fb_prob_dict["E2"] = 1/(2*d) + d/(2*d) * (1/(2*d-1)) + (d/(2*d)) * ((d-1)/(2*d-1)) * (1/(2*d-2))
            self.fb_prob_dict["E3"] = 1/2 * (1/d) + (1/2) * (d/(2*d-1)) * (1/d)
            self.fb_prob_dict["E4"] = 1/2 * (d/(2*d-1)) * 1 * (1/d) +   (1/2) * 1 * 1 * (1/d) + (1/2) * ((d-1) / (2*d-1)) * (d/(2*d-2)) * (1/d)
            self.fb_prob_dict["E5"] = 1/2 * ((d-1) / (2*d - 1)) * ((d-2) / (2*d - 2)) * (1/d)
        elif self.args.max_depth == 6:
            self.fb_prob_dict["E0"] = 1/d
            self.fb_prob_dict["E1"] = (1/(2*d)) \
                                    + ((d-1)/(2*d)) * (1/(2*d-1)) \
                                    + ((d-1)/(2*d)) *(d-2)/(2*d-1) * (1/(2*d-2)) \
                                    + ((d-1)/(2*d)) * ((d-2)/(2*d-1)) * ((d-3)/(2*d-2)) * (1/(2*d-3))
            self.fb_prob_dict["E2"] = 1/(2*d) \
                                    + d/(2*d) * (1/(2*d-1)) \
                                    + (d/(2*d)) * ((d-1)/(2*d-1)) * (1/(2*d-2)) \
                                    + (d/(2*d)) * ((d-1)/(2*d-1)) * ((d-2)/(2*d-2)) * (1/(2*d-3))
            self.fb_prob_dict["E3"] = (1/2) * (1/d) \
                                    + (1/2) * ((d-1)/d) * (1/(d-1)) \
                                    + (1/2) * ((d-1)/d) * ((d-2)/(d-1)) * (1/(d-2)) \
                                    + (1/2) * (d/(2*d-1)) * (1/d) \
                                    + (1/2) * (d/(2*d-1)) * ((d-1)/d) * (1/(d-1)) \
                                    + (1/2) * ((d-1)/(2*d-1)) * (d/(2*d-2)) * (1/d)

            self.fb_prob_dict["E4"] = (1/2) * 1 * 1 * 1 * (1/d) \
                                    + (1/2) * (d/(2*d-1)) * 1 * 1 * (1/d) \
                                    + (1/2) * ((d-1)/(2*d-1)) * ((d)/(2*d-2)) * 1 * (1/d) \
                                    + (1/2) * ((d-1)/(2*d-1)) * ((d-2)/(2*d-2)) * (d/(2*d-3)) * (1/d)
    
            self.fb_prob_dict["E5"] = 1/2 * ((d-1) / (2*d - 1)) * ((d-2) / (2*d - 2)) * ((d-3) / (2*d - 3)) * (1/d) 
        print(self.fb_prob_dict)
        
        self.target_prob = torch.zeros(len(self._fbs), 1).cuda()

        for a_i in range(len(self._fbs)):
            fb_type = self.action_to_fb_type(a_i+1)
            uni_fb_prob = self.get_fb_prob(fb_type)
            self.target_prob[a_i] = uni_fb_prob

    def action_to_fb_type(self, action):
        if action == PASS_FB:
            return "E"
        else:
            return "E" + str((action-1) % 6)
    
    def get_fb_prob(self, fb_type):
        return self.fb_prob_dict[fb_type]

    def get_fb(self, idx):
        return self._fbs[idx-1]
    
    def forward(self, x, actions):
        """Forward the batch of inputs through the Function Blocks.

        Args:
            x: batch of inputs
            actions: batch of actions giving routing connections.
        
        Returns:
            a Tensor of outputs
        """
        batch_size = x.shape[0]
        y = torch.zeros((batch_size, self.args.num_classes)).cuda()

        for i in range(batch_size):
            cur_x = x[i:i+1]
            prev_layer = None
            for t, a_t in enumerate(actions[i]):
                if a_t != PASS_FB:
                    cur_fb =  self.get_fb(a_t)
                    if isinstance(prev_layer, nn.Conv2d) and isinstance(cur_fb.layer, nn.Linear):
                        cur_x = self.max_pool(cur_x)
                        cur_x = torch.flatten(cur_x, start_dim=1)
                    cur_x = cur_fb(cur_x)
                    prev_layer = cur_fb.layer
            y[i] = cur_x

        return y

    def get_num_fbs(self):
        return len(self._fbs)
