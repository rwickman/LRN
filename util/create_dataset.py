import torch
import torchvision
import numpy as np
import os
from tqdm import tqdm
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=True)

# num:
    # example, target (0 or 1)

# For each number 0-9 need 1k positive classes and 1k for each other numbers
# EX: 0 needs 1k positive and 1k 9 negative, 1k 8 negative, ..., 1k 1 negative 

count = {}
for i in range(10):
  count[i] = {}
  for j in range(10):
    count[i][j] = 0

MAX_EXAMPLES = 100
done = True
with open("mnist-mtl.npy", "wb") as f:
    for img, target in tqdm(train_loader):
        target = target.numpy()[0]
        for i in range(10):
          if count[i][target] < MAX_EXAMPLES:
            count[i][target] += 1
            # Task, Image, Target
            ex = i, img.numpy(), np.array(int(target == i))
            np.save(f, ex)

    


with open("mnist-mtl.npy", 'rb') as f:
  fsz = os.fstat(f.fileno()).st_size
  print(fsz)
  out = np.load(f, allow_pickle=True)
  while f.tell() < fsz:
      out = np.vstack((out, np.load(f, allow_pickle=True)))
  print(out.shape)
