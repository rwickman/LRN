import torch
import torch.nn as nn
import os
from dataclasses import dataclass

@dataclass
class FunctionBlock:
    layer: nn.Module
    norm: nn.Module = None
    act: nn.Module = None



# TODO: Move ReLU and Layer Norm here
class LayerManager:
    def __init__(self, args):
        self.args = args
        self._layer_dict = {}
        self._norm_layers = {}
        self._num_filters = 32
        self._filter_size = 3
        self._hidden_size = 128
        if self.args.load:
            self.load()
        else:
            self._init_layers()


    def _init_layers(self):
        for i in range(self.args.num_dup_layers):
            # Input layer
            self._add_layer(
                    FunctionBlock(
                        nn.Conv2d(self.args.in_channels, self._num_filters, self._filter_size, stride=1, padding=1),
                        nn.InstanceNorm2d(self._num_filters, affine=True, eps=1e-6), 
                        nn.ReLU()
                    )
                )

            # Hidden layers
            self._add_layer(
                FunctionBlock(
                    nn.Conv2d(self._num_filters, self._num_filters, self._filter_size, stride=1, padding=1),
                    nn.InstanceNorm2d(self._num_filters, affine=True, eps=1e-6),
                    nn.ReLU()
                )
            )

            self._add_layer(
                FunctionBlock(
                    nn.Linear(self.args.img_size**2*self._num_filters, self._hidden_size),
                    nn.LayerNorm(self._hidden_size, eps=1e-6),
                    nn.ReLU()
                )
            )
            
            self._add_layer(
                FunctionBlock(
                    nn.Linear(self._hidden_size, self._hidden_size),
                    nn.LayerNorm(self._hidden_size, eps=1e-6),
                    nn.ReLU()
                )
            )

            # Add last layers
            self._add_layer(
                FunctionBlock(
                    nn.Linear(self._hidden_size, self.args.num_classes)
                )
            )

            self._add_layer(
                FunctionBlock(
                    nn.Linear(self.args.img_size**2*self._num_filters, self.args.num_classes)
                )
            )

        print(self._layer_dict)

    def save(self):
        """Save the layers."""
        torch.save(
            self._layer_dict, 
            os.path.join(self.args.save_dir, "layer_modules.pt"))
    
    def load(self):
        self._layer_dict = torch.load(
            os.path.join(self.args.save_dir, "layer_modules.pt"))

    def _add_layer(self, layer):
        self._layer_dict[len(self._layer_dict)+1] = layer
        #\self._norm_layers[len(self._norm_layers)] = 

    def get_fb(self, idx):
        return self._layer_dict[idx]
    
    def get_num_layers(self):
        return len(self._layer_dict)