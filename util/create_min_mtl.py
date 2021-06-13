import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import random

img_dim = (84,84)
root_dir = "/media/data/code/Routing/data/ILSVRC/Data/CLS-LOC/train/"
data_dirs = os.listdir(root_dir)
out_train = "min-mtl.npy"
out_test = "min-mtl-test.npy"
num_images = 800
num_test_images = 50
num_classes = 5

# Get first 50 labels
labels_file = "/media/data/code/Routing/data/LOC_synset_mapping.txt"
labels = []
NUM_LABELS = 50
with open(labels_file) as f:
    for i, line in enumerate(f):
        if i >= NUM_LABELS:
            break
        s = line.split()
        labels.append(s[0])

# Create 10 groups with 5 labels each
with open(out_train, "wb") as f:
    for idx, data_dir in enumerate(labels):
        data_dir_path = os.path.join(root_dir, data_dir)
        img_paths = glob(data_dir_path + "/*")[:num_images]
        task_id = idx // num_classes
        label = idx % num_classes
        print("(task_id, label)", task_id, label)
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_dim)
            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :]
            img = img.astype(np.float32)

            ex = task_id, img, label
            np.save(f, ex)

with open(out_test, "wb") as f:
    for idx, data_dir in enumerate(labels):
        data_dir_path = os.path.join(root_dir, data_dir)
        img_paths = glob(data_dir_path + "/*")[num_images:num_test_images+num_images]
        print(len(img_paths))
        print("(task_id, label)", task_id, label)
        task_id = idx // num_classes
        label = idx % num_classes
        
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_dim)
            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :]
            img = img.astype(np.float32)
            ex = task_id, img, label
            np.save(f, ex)

        #print(type(img))
        #plt.imshow(img)
        #plt.show()

    #print(os.path.join(data_dir_path, os.listdir(data_dir_path)))
