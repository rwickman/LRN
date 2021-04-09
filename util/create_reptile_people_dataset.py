import torch
import torchvision
import numpy as np
import argparse, os
from tqdm import tqdm





def load_datasets(args):

  if args.dataset.lower() == "mnist":
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                              ]))
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                              ]))
  elif args.dataset.lower() == "cifar":
    train_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                              ]))
    test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                              ]))

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True)
  
  return train_loader, test_loader

# num:
    # example, target (0 or 1)

# For each number 0-9 need 1k positive classes and 1k for each other numbers
# EX: 0 needs 1k positive and 1k 9 negative, 1k 8 negative, ..., 1k 1 negative 

# FOR CIFAR-100 you will have

def normalize(img):
  return img / 127.5 - 1


#def create_cifar_dataset(args):
#  train_loader, test_loader = load_datasets(args)
#  # Create training data
#  with open(args.out_train, "wb") as f:
#    for img, target in tqdm(train_loader):
#      target = target.numpy()
#      # Create course label task
#      task = target // 5
#      # Create fine label relative to task
#      label = target % 5
#      ex = int(task), img.numpy(), np.array(label)
#      np.save(f, ex)
#  
#  # Create testing data
#  with open(args.out_test, "wb") as f:
#    for img, target in tqdm(test_loader):
#      target = target.numpy()
#      # Create course label task
#      task = target // 5
#      # Create fine label relative to task
#      label = target % 5
#
#      ex = int(task), img.numpy(), np.array(label)
#      np.save(f, ex)

def create_cifar_dataset(args):
  train_loader, test_loader = load_datasets(args)
  # Create training data
  with open(args.out_train, "wb") as f:
    for img, target in tqdm(train_loader):
      target = target.numpy()
      # Create course label task
      task = target // 5
      if task != 15 and task != 14:
          continue
      elif task == 14:
        task = 0
      else:
        task = 1
      # Create fine label relative to task
      label = target % 5
      ex = int(task), img.numpy(), np.array(label)
      np.save(f, ex)
  
  # Create testing data
  with open(args.out_test, "wb") as f:
    for img, target in tqdm(test_loader):
      target = target.numpy()
      # Create course label task
      task = target // 5
      if task != 15 and task != 14:
          continue
      elif task == 14:
        task = 0
      else:
        task = 1
      # Create fine label relative to task
      label = target % 5

      ex = int(task), img.numpy(), np.array(label)
      np.save(f, ex)



def create_mnist_dataset(args):
  train_loader, test_loader = load_datasets(args)
  train_count = {}
  test_count = {}
  for task in args.tasks:
    train_count[task] = {}
    test_count[task] = {}

    for j in range(10):
      train_count[task][j] = 0
      test_count[task][j] = 0

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
  if args.dataset == "mnist":
    create_mnist_dataset(args)
  elif args.dataset == "cifar":
    create_cifar_dataset(args)
  else:
    raise Exception("Invalid dataset.")

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

  parser.add_argument("--dataset", default="mnist",
      help="Dataset type (e.g., mnist or cifar).")

  main(parser.parse_args())
