import torch
import torch.nn as nn
import os

# TODO: Encode layers in config file
class LayerManager:
    def __init__(self, args, num_duplicates=2):
        self.args = args
        self._layer_dict = {}
        self._num_duplicates = num_duplicates
        if self.args.load:
            self.load()
        else:
            self._init_layers()

        # TODO: Load Layers

    def _init_layers(self):
        for i in range(self._num_duplicates):
            # Input layer
            self._add_layer(nn.Conv2d(1, 32, 3, stride=1, padding=1))

            # Intermediate layers
            self._add_layer(nn.Conv2d(32, 32, 3, stride=1, padding=1))
            self._add_layer(nn.Linear(self.args.img_size**2*32, 128))
            self._add_layer(nn.Linear(128, 128))

            # Add last layers
            self._add_layer(nn.Linear(128, 10))
            self._add_layer(nn.Linear(self.args.img_size**2*32, 10))

        # Add softmax for prediction
        self._add_layer(nn.Softmax)

    def save(self):
        """Save the layers."""
        torch.save(
            self._layer_dict, 
            os.path.join(self.args.save_dir, "layer_modules"))
    
    def load(self):
        self._layer_dict = torch.load(
            os.path.join(self.args.save_dir, "layer_modules"))

    def _add_layer(self, layer):
        self._layer_dict[len(self._layer_dict)] = layer

    def get_layer(self, idx):
        return self._layer_dict[idx]