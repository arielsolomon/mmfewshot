import os
import pickle
import numpy as np
from PIL import Image
import glob

# Specify the path to the data batch file
root = '/home/user1/ariel/federated_learning/data/cifar/cifar-10-batches-py/'
dest = '/'.join(root.split('/')[:-2])+'/'
data_batch_file = 'data_batch_1'  # Replace with the actual filename
data_batch_file = glob.glob(root+'*')

# Create folders to save images and labels
image_folder = dest+'images/'
label_folder = dest+'labels/'
os.makedirs(image_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

# Load the data batch using pickle
i = 0
for item in data_batch_file:

    with open(item, 'rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Extract images and labels
        images = data[b'data']
        labels = data[b'labels']

        # Loop through the data and save images and labels

        for (image, label) in zip(images, labels):
            # Reshape and convert the image data
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)
            image = Image.fromarray(image)

            # Save the image as a PNG file
            image_filename = os.path.join(image_folder, f'image_{i}.png')
            image.save(image_filename)

            # Save the label as a text file
            label_filename = os.path.join(label_folder, f'label_{i}.txt')
            with open(label_filename, 'w') as label_file:
                label_file.write(str(label))
            i+=1
