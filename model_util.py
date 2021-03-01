
import torch.nn as nn


def conv2d_out(inp_size, kernel_size, stride, padding=0):
    # W - F + 2P / S + 1
    return (inp_size - kernel_size + 2*padding) / stride + 1
 
def is_valid_mapping(prev_layer, cur_layer, out_size):
    """Checks if a mapping between layers is valid."""
    if prev_layer == None:
        if isinstance(cur_layer, nn.Conv2d):
            # Check if the current layer is an "input" layer
            return cur_layer.in_channels == out_size
        else:
            # First layer must be Conv layer
            return False

    elif isinstance(prev_layer, nn.Conv2d):
        if isinstance(cur_layer, nn.Conv2d):
            # Check if channels  match
            return prev_layer.out_channels == cur_layer.in_channels
        elif isinstance(cur_layer, nn.Linear):
            # The output of the Conv layer must match linear
            print("out_size", out_size)
            print("cur_layer.in_features", cur_layer.in_features)
            return out_size == cur_layer.in_features
        else:
            # Conv layers must be followed Conv or a Linear layer
            return False
    
    elif isinstance(prev_layer, nn.Linear):
        if isinstance(cur_layer, nn.Conv2d):
            # A Conv layer cannot follow a Linear layer
            return False
        elif isinstance(cur_layer, nn.Linear):
            # Check if features match
            return prev_layer.out_features == cur_layer.in_features
        elif isinstance(cur_layer, nn.Softmax):
            return True
        else:
            print("prev_layer", prev_layer, "cur_layer", cur_layer)
            raise Exception("Invalid layers.")
    elif isinstance(prev_layer, nn.Softmax):
        return cur_layer is None
    else:
        raise Exception("Invalid layers.")