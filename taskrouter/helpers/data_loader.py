import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

import sys


seed = 1


class RandomBlur:
    def __init__(self, p=0.1, kernel_sizes=[1, 3, 5, 7]):
        self.p = p
        # self.blur = transforms.GaussianBlur(7)
        self.blur_sizes = kernel_sizes

    def __call__(self, x):

        if random.random() <= self.p:
            kernel_size = random.choice(self.blur_sizes)
            self.blur = transforms.GaussianBlur(kernel_size)
            x = self.blur(x)
        
        return x

class ThresholdTransform:
  def __init__(self, threshold=240):
    self.thr = threshold / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data typ

class OmniglotDataset(Dataset):
    def __init__(self, img_files, task_ids, class_ids, img_size=105, aug=False):
        self.img_files = img_files
        self.task_ids = torch.tensor(task_ids)
        self.class_ids = torch.tensor(class_ids)
        self.aug = aug
        print("len(self.img_files)", len(self.img_files))
        print("len(self.task_ids)", len(self.task_ids))
        print("len(self.class_ids)", len(self.class_ids))
        if self.aug:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=45),
                transforms.RandomResizedCrop(((img_size, img_size)), scale=(0.5, 1.0)),
                RandomBlur(),
                ThresholdTransform()
            ])
        


    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img = torch.tensor(np.array(Image.open(self.img_files[idx]).convert('L')), dtype=torch.float32).unsqueeze(0)
        img = 1 - (img/255)
        if self.aug:
            img = self.transform(img)
        return img, self.task_ids[idx].unsqueeze(0), self.class_ids[idx]

def load_omniglot():
    print("LOADING OMNIGLOT")
    omniglot_dir = "/media/data/datasets/omniglot/multitask"

    langs = os.listdir(omniglot_dir)

    task_ids_train = []
    task_ids_val = []
    task_ids_test = []


    dataset_files_train = []
    dataset_files_val = []
    dataset_files_test = []

    class_ids_train = []
    class_ids_val = []
    class_ids_test = []

    # Number of classes per task
    out_sizes = []

    # Iterate over the language directories
    for task_id, l in enumerate(langs):
        chars = os.listdir(os.path.join(omniglot_dir, l))
        chars = sorted(os.listdir(os.path.join(omniglot_dir, l)), key=lambda x: int(x.split("character")[1]))
        out_sizes.append(len(chars))
        # Iterate over the character directories
        for char in chars:
        
            char_dir = os.path.join(omniglot_dir, l, char)
            cur_class_id = int(char.split("character")[1]) - 1
            char_img_files = sorted(os.listdir(char_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            random.Random(seed).shuffle(char_img_files)

            # Iterate over the character images
            for i, img_file in enumerate(char_img_files):
                if i <= 9:
                    dataset_files_train.append(os.path.join(char_dir, img_file))
                    class_ids_train.append(cur_class_id)
                    task_ids_train.append(task_id)
                elif i > 9 and i <= 13:
                    dataset_files_val.append(os.path.join(char_dir, img_file))
                    class_ids_val.append(cur_class_id)
                    task_ids_val.append(task_id)
                else:
                    dataset_files_test.append(os.path.join(char_dir, img_file))
                    class_ids_test.append(cur_class_id)
                    task_ids_test.append(task_id)
    
    print("TRAIN DATASET")
    train_dataset = OmniglotDataset(dataset_files_train, task_ids_train, class_ids_train, aug=True)
    print("VAL DATASET")
    val_dataset = OmniglotDataset(dataset_files_val, task_ids_val, class_ids_val)
    print("TEST DATASET")
    test_dataset = OmniglotDataset(dataset_files_test, task_ids_test, class_ids_test)

    return train_dataset, val_dataset, test_dataset, out_sizes
