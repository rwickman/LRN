import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

@dataclass
class RouterResult:
    failed: bool
    action_probs: List[torch.Tensor]
    actions: List[int]
    # The index in the action probability that corresponds to the action (i.e., layer)
    action_indices: List[int]
    model: nn.Sequential
    critic_outputs: List[torch.Tensor]
    att_weights: List[Dict[str, torch.Tensor]]

PASS_LAYER = 0

def conv2d_out(inp_size, kernel_size, stride, padding=0):
    # W - F + 2P / S + 1
    return (inp_size - kernel_size + 2*padding) / stride + 1
 
def is_valid_mapping(prev_layer, cur_layer, out_size, cur_depth, max_depth, num_classes):
    # TODO: Cover PASS_LAYER AND TAKE CUR_DEPTH AND MAX_DEPTH INTO ACCOUNT
    """Checks if a mapping between layers is valid."""
    if prev_layer is None:
        if cur_layer is None:
            # Cannot use PASS_LAYER as first layer
            return False
        elif isinstance(cur_layer, nn.Conv2d):
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
            return out_size == cur_layer.in_features
        elif cur_layer is None:
            # Only use PASS_LAYER if not at last depth
            return cur_depth < max_depth - 1
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
        elif cur_layer is None:
            # Only use PASS_LAYER if not at last depth or already added last layer
            return cur_depth < max_depth - 1 or prev_layer.out_features == num_classes
        else:
            raise Exception("Invalid layers.")
    else:
        raise Exception("Invalid layers.")


def get_angles(pos, i, hidden_size):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(hidden_size))
  return pos * angle_rates

def positional_encoding(position, hidden_size):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(hidden_size)[np.newaxis, :],
                          hidden_size)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return torch.tensor(pos_encoding, dtype=torch.float32).cuda()



def plot_attention_head(ax, state, layer_ids, attention):
  # The plot is of the attention when a token was generated.
  # The model didn't generate `<START>` in the output. Skip it. 
  ax.matshow(attention)
  ax.set_xticks(range(len(state)))
  ax.set_yticks(range(len(layer_ids)))

  ax.set_xticklabels(
      state, rotation=90)
  
  ax.set_yticklabels(layer_ids)


def plot_attention_weights(state_labels, layer_ids, attention_heads):
    #fig = plt.figure(figsize=(16, 8))
    num_att_layers = len(attention_heads)
    num_heads = attention_heads[0]["decoder_layer1_weights2"].shape[1]
    print(num_heads, num_att_layers)
    fig, axes = plt.subplots(num_att_layers, num_heads)
    
    for depth, layer_att in enumerate(attention_heads):
        for h, head in enumerate(layer_att["decoder_layer1_weights2"][0].detach().cpu().numpy()):
            #axes = fig.add_subplot(len(attention_heads), 4, h * depth + depth)
            if num_att_layers == 1 and num_heads == 1:
                cur_ax = axes
            elif num_heads == 1:
                cur_ax = axes[depth]
            elif num_att_layers == 1:
                cur_ax = axes[h]
            else:
                cur_ax = axes[depth, h]

            plot_attention_head(cur_ax, state_labels[depth], layer_ids[depth], head)

            cur_ax.set_xlabel('Head {}'.format(h+1))

    # plt.tight_layout()
    plt.show()

