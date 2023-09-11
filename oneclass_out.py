import os
import pickle
import numpy as np
from PIL import Image
import glob
from shutil import move

# Specify the path to the data batch file
root = '/home/user1/ariel/federated_learning/data/cifar/'
dest = root +'one_out/'
if not os.path.exists(dest):
    os.mkdir(dest)
lbl_batch_file = glob.glob(root+'labels/'+'*')

# Create folders to save images and labels
image_folder = dest+'images/'
label_folder = dest+'labels/'
os.makedirs(image_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

# Load the data batch using pickle
i = 0
for item in lbl_batch_file:

    with open(item, 'r') as file:
        line = file.readline().strip()
        if line=='3':
            move(item, label_folder)
            new_item = item.replace('labels', 'images')
            new_item1 = new_item.replace('txt','png')
            new_item2 = new_item1.replace('label', 'image')
            move(new_item2, image_folder)