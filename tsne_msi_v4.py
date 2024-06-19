# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:20:05 2024

@author: Andrei
"""

import os
import numpy as np
from skimage import io
from sklearn.manifold import TSNE

def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = io.imread(image_path, as_gray=True)  # Read images in grayscale
        images.append(image)

    return np.array(images)

def apply_tsne_to_images(images, no_dims=3, perplexity=30.0):
    n_images, X, Y = images.shape
    tsne_results = np.zeros((n_images, 1, 3))

    for i in range(n_images):
        image = images[i].reshape(X * Y, 1)  # Reshape image to a vector
        # Apply t-SNE directly to the flattened image
        tsne = TSNE(n_components=no_dims, perplexity=perplexity)
        tsne_result = tsne.fit_transform(image)
        tsne_results[i, :, :] = tsne_result.reshape(1,  -1)

    return tsne_results

# Specify the folder containing your images
folder_path = "\msi"

# Read images from the folder
msi_data = read_images_from_folder(folder_path)

# Apply t-SNE to each image
tsne_results = apply_tsne_to_images(msi_data, no_dims=3, perplexity=30.0)

# tsne_results is a 3D array where tsne_results[i, :, :] is the 1x3 t-SNE result for the i-th image
# You can access individual results like tsne_results[0, :, :] for the first image
