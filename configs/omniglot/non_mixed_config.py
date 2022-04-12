import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


out_sizes=[]
img_size=105
in_channels=1
num_filters=32
num_hidden=6
num_downsample=4
num_tasks = 20

omniglot_dir = "/media/data/datasets/omniglot/multitask"

routing_depth = 1 + num_hidden + num_downsample

unique_inp_modules = 3
unique_hidden_modules = 6
unique_downsample_modules = 3
num_modules = unique_inp_modules + num_hidden * unique_hidden_modules + num_downsample * unique_downsample_modules - 1

"""Create the valid modules for each layer."""

valid_mbs = {
    0: [0, 1, 2]
}
# Create entries for hidden modules
hidden_start_idx = unique_downsample_modules
for i in range(num_hidden):
    valid_mbs[i+1] = list(range(
        hidden_start_idx + unique_hidden_modules * i, hidden_start_idx + unique_hidden_modules * (i+1)))

# Create entries for downsample modules
down_start_idx = hidden_start_idx + num_hidden * unique_hidden_modules


for i in range(num_downsample-1):
    valid_mbs[i+1+num_hidden] = list(range(
        down_start_idx + unique_downsample_modules * i,
        down_start_idx + unique_downsample_modules * (i+1)))

valid_mbs[i+1+num_hidden+1] = list(range(
    down_start_idx + unique_downsample_modules * 3,
    down_start_idx + unique_downsample_modules * 3 + 2))


# Create entries for output modules
output_start_idx = down_start_idx + unique_downsample_modules * (num_downsample-1) + 2
print(valid_mbs)
# valid_mbs[1+num_hidden+num_downsample] = list(range(
#     output_start_idx, output_start_idx + num_tasks))