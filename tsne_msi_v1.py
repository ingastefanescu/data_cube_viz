# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:10:46 2024

@author: Andrei
"""

import cv2
import numpy as np
import os
from tsne import *

def read_grayscale_images_from_folder(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the image files to maintain order
    image_files.sort()

    # Initialize an empty list to store images
    images = []

    # Loop through each image file and read it using OpenCV
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # Append the image to the list
            images.append(img)

    # Convert the list of images to a NumPy array
    images_array = np.array(images)

    # Transpose the array to have dimensions (height, width, number_of_images)
    images_array = np.transpose(images_array, (1, 2, 0))

    return images_array

# Example usage:
folder_path = '/msi'
msi_images = read_grayscale_images_from_folder(folder_path)

# Print the dimensions of the resulting array
print(msi_images.shape)

# Reshape the array into the new dimensions (total_number_of_pixels, number_of_images)
new_dims = (msi_images.shape[0] * msi_images.shape[1], msi_images.shape[2])
msi_data = msi_images.reshape(new_dims)

# Print the modified dimensions of the resulting array
print("Modified dimensions:", msi_data.shape)

print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")

# X = np.loadtxt("mnist2500_X.txt")
# labels = np.loadtxt("mnist2500_labels.txt")

msi_data = msi_data.astype(float)
msi_comp = tsne(msi_data, 3, msi_data[1], 20.0)

pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.show()