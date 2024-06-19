# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:28:18 2024

@author: Andrei
"""

# import cv2
# import os
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# def read_first_image_apply_tsne(folder_path, perplexity=0.5, random_state=42):
#     # Get the list of files in the folder
#     files = os.listdir(folder_path)

#     # Filter out non-image files (you may need to adjust this depending on your image file extensions)
#     image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     if not image_files:
#         print("No image files found in the specified folder.")
#         return

#     # Read the first image
#     first_image_path = os.path.join(folder_path, image_files[0])
#     image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale

#     # Flatten the image to use as input for t-SNE
#     flattened_image = image.flatten().reshape(1, -1)

#     # Apply t-SNE
#     tsne = TSNE(perplexity=perplexity, random_state=random_state)
#     tsne_result = tsne.fit_transform(flattened_image)

#     # Plot the t-SNE result
#     plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
#     plt.title('t-SNE Result for the First Image')
#     plt.show()

# # Example usage
# folder_path = '/msi'
# read_first_image_apply_tsne(folder_path)

#%%
# import cv2
# import matplotlib.pyplot as plt

# def visualize_pixel_values(image_path):
#     # Read the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Plot the pixel values directly
#     plt.imshow(image, cmap='gray')
#     plt.title('Pixel Values Visualization for Single Image')
#     plt.colorbar()
#     plt.show()
    
    
# # Example usage
# image_path = '/msi/image_0001.jpg'
# apply_tsne_on_single_image(image_path)


#%% 

import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# def apply_pca(image_path, n_components=3):
# Read the image
image_path = 'D:/OneDrive - Universitatea Politehnica Bucuresti/Andrei/Facultate/Master/DISSERTATION/work/msi/image_0001.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Flatten the image to a 1D array
# flattened_image = image.flatten()

# Print the number of samples and features
n_samples = image.shape[0]
n_features = image.shape[1]
print(f"n_samples: {n_samples}, n_features: {n_features}")

# Determine the valid range for n_components
valid_n_components = min(n_samples, n_features)

# Set n_components within the valid range
n_components = 3
if n_components is None or n_components > valid_n_components:
    n_components = valid_n_components

# Apply PCA
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(image)

# Reconstruct the image using the principal components
reconstructed_image = pca.inverse_transform(pca_result)

# Reshape the reconstructed image to its original shape
reconstructed_image = reconstructed_image.reshape(image.shape)

# Plot the original and reconstructed images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image (PCA)')

plt.show()

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
data = pca_result
# Extract x, y, and z coordinates from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Plot the 3D scatter plot
ax.scatter(x, y, z, c='blue', marker='o', label='Data Points')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot')

# Add a legend
ax.legend()

# Show the plot
plt.show()