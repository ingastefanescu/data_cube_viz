# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:07:39 2024

@author: Andrei
"""

import os
from skimage import io
# from tsne import *

# Replace 'folder_path' with the path to your image folder
folder_path = "/msi"

# Get a list of image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Read the first image
if image_files:
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = io.imread(first_image_path, as_gray=True)
    
    # Reshape the image into a column vector
    reshaped_image = first_image.reshape(-1, 1)

    # Display the image or print its shape
    print(f"Shape of the first image: {first_image.shape}")
else:
    print("No image files found in the specified folder.")


msi_data = reshaped_image


tsne_results = tsne(msi_data, no_dims=3,initial_dims=msi_data[0: ], perplexity=30.0)
