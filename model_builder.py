import torch.nn as nn 

from model_util import *

class ModelBuilder:
    def __init__(self, args):
        self.args = args
        self._layers = []
        self._update_out_size()
        

    def attempt_add_layer(self, cur_layer):
        # TODO: Add flatten layer if prev layer is Conv and cur_layer is linear
        #prev_layer = get_prev_layer() 
        #if get_prev_layer
        prev_layer = self.get_prev_layer()
        if is_valid_mapping(prev_layer, cur_layer, self.cur_out_size):
            if isinstance(prev_layer, nn.Conv2d) and isinstance(cur_layer, nn.Linear):
                self._layers.append(nn.Flatten())
            self._layers.append(cur_layer)
            self._update_out_size()
            return True
        else:
            return False
    
    def get_prev_layer(self):
        if len(self._layers) == 0:
            return None
        else:
            return self._layers[-1]

    def _update_out_size(self):
        prev_layer = self.get_prev_layer()
        if prev_layer is not None:
            if isinstance(prev_layer, nn.Conv2d):
                out_dim = conv2d_out(
                    self.args.img_size,
                    prev_layer.kernel_size[0],
                    prev_layer.stride[0],
                    prev_layer.padding[0])
                
                self.cur_out_size = out_dim ** 2 * 32
            elif isinstance(prev_layer, nn.Linear):
                self.cur_out_size = prev_layer.out_features
            else:
                raise Exception("Invalid layer.")
        else:
            # There is not a previous layer so set to input size

            self.cur_out_size = self.args.in_channels


    def build(self):
        """Build a model out of the layers."""
        model_arch = []
        for i, layer in enumerate(self._layers):
            model_arch.append(layer)
            # Add ReLU after each layer, except for last layer
            if i + 1 < len(self._layers):
                model_arch.append(nn.ReLU())
            
        return nn.Sequential(*model_arch)

        