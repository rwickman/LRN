import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


out_sizes=[]
img_size=105
in_channels=1
num_filters=48

# num_hidden=4
# num_downsample=4
num_tasks = 20

omniglot_dir = "/media/data/datasets/omniglot/multitask"

routing_depth = 8
strides = [2, 2, 2, 2, 2, 1, 1, 1]
unique_inp_modules = 3
unique_hidden_modules = 6
unique_downsample_modules = 3

num_modules = 8 * 7#unique_inp_modules + num_hidden * unique_hidden_modules + num_downsample * unique_downsample_modules

"""Create the valid modules for each layer."""

valid_mbs = {}

for i in range(len(strides)):
    valid_mbs[i] = list(range(i * 7, (i+1) * 7)) 

print(valid_mbs)