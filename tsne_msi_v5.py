# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:34:43 2024

@author: Andrei
"""

#%% Imports
import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

#%% Load msi images

# Specify the path to your folder containing images
folder_path = '/msi'

# Initialize an empty list to store image values
msi_data = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is an image (you might want to add more sophisticated checks)
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Open the image using cv2
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was successfully loaded
        if img is not None:
            # Flatten the image array and append to the list
            # img_values = img.flatten().tolist()
            msi_data.append(img)
            
# Convert the list of 2D arrays to a 3D NumPy array
msi_data = np.array(msi_data)


#%% tsne

# Flatten each 2D array in image_values to make it compatible with t-SNE
flattened_images = msi_data.reshape((114, -1))

# Apply t-SNE to reduce dimensionality to 3D
tsne = TSNE(n_components=3, random_state=42, perplexity=20, n_iter=250)

embedded_images = tsne.fit_transform(flattened_images)

# Print or use the resulting array with dimensions (114, 3)
print(embedded_images.shape)

# Visualize the 3D embeddings (optional)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedded_images[:, 0], embedded_images[:, 1], embedded_images[:, 2])
plt.show()
