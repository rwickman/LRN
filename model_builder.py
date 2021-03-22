import torch.nn as nn 

from model_util import *

class ModelBuilder:
    def __init__(self, args):
        self.args = args
        self._fbs = []
        self._update_out_size()

    def attempt_add_fb(self, cur_fb, cur_depth):
        prev_layer = self.get_prev_layer()
        if is_valid_mapping(prev_layer, cur_fb.layer, self.cur_out_size, cur_depth, self.args.max_depth, self.args.num_classes):
            # if isinstance(prev_layer, nn.Conv2d) and isinstance(cur_fb.layer, nn.Linear):
            #     self._fbs.append(nn.Flatten())
            self._fbs.append(cur_fb)
            self._update_out_size()
            return True
        else:
            return False

    def is_valid_fb(self, cur_fb, cur_depth):
        prev_layer = self.get_prev_layer()
        if cur_fb is not None:
            cur_layer = cur_fb.layer
        else:
            cur_layer = None 
        return is_valid_mapping(prev_layer, cur_layer, self.cur_out_size, cur_depth, self.args.max_depth, self.args.num_classes)

    def get_prev_layer(self):
        if len(self._fbs) == 0:
            return None
        else:
            return self._fbs[-1].layer

    def _update_out_size(self):
        prev_layer = self.get_prev_layer()
        if prev_layer is not None:
            if isinstance(prev_layer, nn.Conv2d):
                out_dim = conv2d_out(
                    self.args.img_size,
                    prev_layer.kernel_size[0],
                    prev_layer.stride[0],
                    prev_layer.stride[0])

                               # if i + 1 < len(self._fbs) and not isinstance(layer, nn.Flatten):
            #     model_arch.append(nn.ReLU())
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
        # TODO: Don't add ReLU if using Flatten layer
        model_arch = []
        prev_layer = None
        for i, cur_fb in enumerate(self._fbs):
            
            
            if prev_layer is not None and isinstance(prev_layer, nn.Conv2d) and isinstance(cur_fb.layer, nn.Linear):
                model_arch.append(nn.Flatten())
            
            # Add the layer of the FB
            model_arch.append(cur_fb.layer)

            # Add normalization if available
            if cur_fb.norm is not None:
                model_arch.append(cur_fb.norm)
            
            # Add activation function if available
            if cur_fb.act is not None:
                model_arch.append(cur_fb.act)
            
            # Add ReLU after each layer, except for last layer
            # if i + 1 < len(self._fbs) and not isinstance(layer, nn.Flatten):
            #     model_arch.append(nn.ReLU())
        #model_arch.append(nn.Softmax())
            prev_layer = cur_fb.layer
        return nn.Sequential(*model_arch).cuda()

        