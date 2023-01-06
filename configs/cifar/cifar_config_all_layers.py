import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


out_size = 5
img_size=32
in_channels=3
num_filters=48
conv_out_size = 128
# num_hidden=4
# num_downsample=4
num_tasks = 20

routing_depth = 7
strides = [1, 1, 1, 1]
num_modules = []

unique_hidden_modules = 20


num_modules = routing_depth * unique_hidden_modules#unique_inp_modules + num_hidden * unique_hidden_modules + num_downsample * unique_downsample_modules

"""Create the valid modules for each layer."""

valid_mbs = {}

for i in range(routing_depth):
    valid_mbs[i] = list(range(i * unique_hidden_modules, (i+1) * unique_hidden_modules)) 
