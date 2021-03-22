import torch.nn as nn 

from model_util import *

class ModelBuilder:
    def __init__(self, args):
        self.args = args
        self._prev_layer = None
        self._update_out_size()

    def attempt_add_fb(self, cur_fb, cur_depth):
        if is_valid_mapping(self._prev_layer, cur_fb.layer, self.cur_out_size, cur_depth, self.args.max_depth, self.args.num_classes):
            self._prev_layer = cur_fb.layer
            self._update_out_size()
            return True
        else:
            return False

    def is_valid_fb(self, cur_fb, cur_depth):
        if cur_fb is not None:
            cur_layer = cur_fb.layer
        else:
            cur_layer = None 
        return is_valid_mapping(self._prev_layer, cur_layer, self.cur_out_size, cur_depth, self.args.max_depth, self.args.num_classes)

    def _update_out_size(self):
        if self._prev_layer is not None:
            if isinstance(self._prev_layer, nn.Conv2d):
                out_dim = conv2d_out(
                    self.args.img_size,
                    self._prev_layer.kernel_size[0],
                    self._prev_layer.stride[0],
                    self._prev_layer.stride[0])

                               # if i + 1 < len(self._fbs) and not isinstance(layer, nn.Flatten):
            #     model_arch.append(nn.ReLU())
                self.cur_out_size = out_dim ** 2 * 32
            elif isinstance(self._prev_layer, nn.Linear):
                self.cur_out_size = self._prev_layer.out_features
            else:
                raise Exception("Invalid layer.")
        else:
            # There is not a previous layer so set to input size
            self.cur_out_size = self.args.in_channels
        