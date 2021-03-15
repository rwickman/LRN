import torch
import torch.nn as nn
import os

# TODO: Encode layers in config file
class LayerManager:
    def __init__(self, args):
        self.args = args
        self._layer_dict = {}
        if self.args.load:
            self.load()
        else:
            self._init_layers()


    def _init_layers(self):
        for i in range(self.args.num_dup_layers):
            # Input layer
            self._add_layer(nn.Conv2d(self.args.in_channels, 32, 3, stride=1, padding=1))

            # Hidden layers
            self._add_layer(nn.Conv2d(32, 32, 3, stride=1, padding=1))
            self._add_layer(nn.Linear(self.args.img_size**2*32, 128))
            self._add_layer(nn.Linear(128, 128))

            # Add last layers
            self._add_layer(nn.Linear(128, self.args.num_classes))
            self._add_layer(nn.Linear(self.args.img_size**2*32, self.args.num_classes))

        # Add softmax for prediction
        # self._add_layer(nn.Softmax)
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

    def get_layer(self, idx):
        return self._layer_dict[idx]
    
    def get_num_layers(self):
        return len(self._layer_dict)