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
strides = [2, 2, 1, 2, 2, 1, 2, 1]
unique_inp_modules = 3
unique_hidden_modules = 6
unique_downsample_modules = 3

num_modules = 8 * 4#unique_inp_modules + num_hidden * unique_hidden_modules + num_downsample * unique_downsample_modules

"""Create the valid modules for each layer."""

valid_mbs = {}
for i in range(len(strides)):
    valid_mbs[i] = list(range(i * 4, (i+1) * 4))

# Add extension to use module where strides match
for i in range(len(strides)):
    for j in range(len(strides)):
        if strides[i] == strides[j] and i != j:
            valid_mbs[i].extend(valid_mbs[j])
    


print(valid_mbs)