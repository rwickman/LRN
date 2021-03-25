import torch
import torchvision
import numpy as np
import argparse, os
from tqdm import tqdm
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=True,
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

def create_datasets(args):

  train_count = {}
  test_count = {}
  for task in args.tasks:
    train_count[task] = {}
    test_count[task] = {}

    for j in range(10):
      train_count[task][j] = 0
      test_count[task][j] = 0

  MAX_EXAMPLES = 200
  done = True
  with open(args.out_train, "wb") as f:
    for img, target in tqdm(train_loader):
        target = target.numpy()[0]
        for task in args.tasks:
          if train_count[task][target] < args.num_train:
            train_count[task][target] += 1
            # Task, Image, Target
            ex = task, img.numpy(), np.array(int(target == task))
            np.save(f, ex)

  with open(args.out_test, "wb") as f:
    for img, target in tqdm(test_loader):
      target = target.numpy()[0]
      for task in args.tasks:
        if test_count[task][target] < args.num_test:
            test_count[task][target] += 1
            ex = task, img.numpy(), np.array(int(target == task))
            np.save(f, ex)

# with open("mnist-mtl.npy", 'rb') as f:
#   fsz = os.fstat(f.fileno()).st_size
#   print(fsz)
#   out = np.load(f, allow_pickle=True)
#   while f.tell() < fsz:
#       out = np.vstack((out, np.load(f, allow_pickle=True)))
#   print(out.shape)

def main(args):
  create_datasets(args)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tasks", nargs="*", type=int, default=list(range(10)),
      help="Tasks to include in dataset.")
  parser.add_argument("--num_train", default=1000,
      help="Number training examples per target and task pair.")
  parser.add_argument("--num_test", default=200,
      help="Number test examples per target and task pair.")
  parser.add_argument("--out_train", default="mnist-mtl.npy",
      help="Output train npy file.")
  parser.add_argument("--out_test", default="mnist-mtl-test.npy",
      help="Output test npy file.")

  main(parser.parse_args())