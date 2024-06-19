# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:28:15 2024

@author: Andrei
"""

#%% Imports
import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from plotly.offline import plot
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import plotly.io as pio


#%% Load msi images

# Define the cropping coordinates
# x1, x2 = 500, 1000
# y1, y2 = 1950, 2300

x1, x2 = 500, 600
y1, y2 = 1950, 2100

# Specify the path to your folder containing images
work_dir = os.getcwd()
folder_path = work_dir + '/msi'

# Initialize an empty list to store image values
msi_data = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file is an image (you might want to add more sophisticated checks)
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Open the image using cv2
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = img[y1:y2, x1:x2]
        
        # Check if the image was successfully loaded
        if img is not None:
            # Flatten the image array and append to the list
            # img_values = img.flatten().tolist()
            msi_data.append(img)
            
# Convert the list of 2D arrays to a 3D NumPy array
msi_data = np.array(msi_data)

#%% Flatten each 2D array in image_values to make it compatible with t-SNE
flattened_images = msi_data.reshape((114, -1))
flattened_images = flattened_images.T

#%% Perplexity vs KL graph 2D

perplexity = np.arange(5,100, 5)
divergence = []

for i in perplexity:
    model = TSNE(n_components=2, init="pca", perplexity=i)
    reduced = model.fit_transform(flattened_images)
    divergence.append(model.kl_divergence_)
fig = px.line(x=perplexity, y=divergence, markers=True) 
fig.update_layout(title = 'MSI 2D', xaxis_title="Perplexity Values", yaxis_title="Divergence")
fig.update_traces(line_color="red", line_width=1)
plot(fig, auto_open=True)
pio.write_html(fig, file='tsne_msi_2d_perp.html', auto_open=True)

#%% Perplexity vs KL graph 3D
perplexity = np.arange(5,50, 5)
divergence = []

for i in perplexity:
    model = TSNE(n_components=3, init="pca", perplexity=i)
    reduced = model.fit_transform(flattened_images)
    divergence.append(model.kl_divergence_)
fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(title = 'MSI 3D', xaxis_title="Perplexity Values", yaxis_title="Divergence")
fig.update_traces(line_color="red", line_width=1)
plot(fig, auto_open=True)
pio.write_html(fig, file='tsne_msi_3d_perp.html', auto_open=True)

#%% tsne 2D with plotly

tsne = TSNE(n_components=2, random_state=42)
tsne_2d = tsne.fit_transform(flattened_images)
tsne.kl_divergence_

# fig = px.scatter(x=tsne_2d[:, 0], y=tsne_2d[:, 1],)
# fig.update_layout(
#     title="t-SNE 2D MSI",
#     xaxis_title="First t-SNE",
#     yaxis_title="Second t-SNE",
# )

# plot(fig, auto_open=True)
# pio.write_html(fig, file='tsne_msi_2d_data.html', auto_open=True)


data = pd.DataFrame({
        'x': tsne_2d[:, 0], 
        'y': tsne_2d[:, 1],
    })

#data['marker_size'] = 0.5

#fig = px.scatter_3d(x=tsne_3d[:, 0], y=tsne_3d[:, 1], z=tsne_3d[:, 2], opacity=0.8)
#fig = px.scatter_3d(data, x = 'x', y = 'y', z = 'z', opacity=0.8, size = 'marker_size', width= 1300, height =1300)
fig = px.scatter(data, x = 'x', y = 'y', opacity=0.8, hover_name = data.index)
fig.update_layout(autosize=True,)
fig.update_traces(marker=dict(color='red'))
plot(fig, auto_open=True)
pio.write_html(fig, file='tsne_msi_2d_data.html', auto_open=True)

#%% tsne 3D 
# Apply t-SNE to reduce dimensionality to 3D
tsne = TSNE(n_components=3, random_state=42, perplexity=25, n_iter=250)
tsne_3d = tsne.fit_transform(flattened_images)
tsne.kl_divergence_

# Print or use the resulting array with dimensions (114, 3)
print(tsne_3d.shape)

#%% tsne 3D with plt

# Visualize the 3D embeddings (optional)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2])
plt.show()

#%% tsne 3D with plotly

#fig = px.scatter_3d(x=tsne_3d[:, 0], y=tsne_3d[:, 1], z=tsne_3d[:, 2], opacity=0.8)
#plot(fig, auto_open=True)
# fig.write_html("3d.html")

data = pd.DataFrame({
        'x': tsne_3d[:, 0], 
        'y': tsne_3d[:, 1],
        'z': tsne_3d[:, 2]
    })

data['marker_size'] = 1

#fig = px.scatter_3d(x=tsne_3d[:, 0], y=tsne_3d[:, 1], z=tsne_3d[:, 2], opacity=0.8)
#fig = px.scatter_3d(data, x = 'x', y = 'y', z = 'z', opacity=0.8, size = 'marker_size', width= 1300, height =1300)
fig = px.scatter_3d(data, x = 'x', y = 'y', z = 'z', opacity=0.8, size = 'marker_size', hover_name = data.index)
fig.update_layout(autosize=True,)
fig.update_traces(marker=dict(color='red'))
plot(fig, auto_open=True)
pio.write_html(fig, file='tsne_msi_3d_data.html', auto_open=True)

#%% Get pixel row and column from flattened image

# Define original matrix dimensions
num_rows = 150
num_cols = 100

# Flattened index to locate in the original matrix
#flattened_index = 12264
flattened_index = 1999

# Calculate row and column indices in the original matrix
row_index = flattened_index // num_cols
col_index = flattened_index % num_cols

# Output the corresponding row and column indices
print(f"Flattened Index {flattened_index} corresponds to:")
print(f"Row: {row_index}, Column: {col_index}")

#%% Highlight pixel

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Load and display an image
#image = img
image = msi_data[2,:,:]
image_array = np.array(image)

# Create a figure and axis for plotting
fig, ax = plt.subplots()

# Display the image
ax.imshow(image_array, cmap = 'gray')

# Specify the row and column of the pixel to highlight (example: row=50, column=100)
# highlight_row = 122
# highlight_col = 64

highlight_row = 19
highlight_col = 99

# Highlight the specific pixel by drawing a rectangle around it
rect = patches.Rectangle((highlight_col - 0.5, highlight_row - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Show the plot with the highlighted pixel
plt.title(f"Highlighted Pixel: Row={highlight_row}, Column={highlight_col}")
plt.show()

#%% PCA information factor

n_components = 20
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(flattened_images)

# Get the explained variance ratios
explained_var = pca.explained_variance_ratio_

# Compute the cumulative variance
cumulative_var = np.cumsum(explained_var)

# Plot the information gain for each component
# plt.bar(range(1, n_components  + 1), explained_var, label='Individual Component')
plt.plot(range(1, n_components  + 1), cumulative_var, label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Information Gain for Each Principal Component')
plt.legend()
plt.show()

#%% PCA 2D msi

pca = PCA(n_components=2)
X_pca = pca.fit_transform(flattened_images)

# pca.score(flattened_images)

# fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1])
# fig.update_layout(
#     title="PCA visualization of msi cube",
#     xaxis_title="First Principal Component",
#     yaxis_title="Second Principal Component",
# )
# plot(fig, auto_open=True)
# pio.write_html(fig, file='pca_msi_2d_data.html', auto_open=True)

data = pd.DataFrame({
        'x': X_pca [:, 0], 
        'y': X_pca [:, 1],
    })

#data['marker_size'] = 0.5

#fig = px.scatter_3d(x=tsne_3d[:, 0], y=tsne_3d[:, 1], z=tsne_3d[:, 2], opacity=0.8)
#fig = px.scatter_3d(data, x = 'x', y = 'y', z = 'z', opacity=0.8, size = 'marker_size', width= 1300, height =1300)
fig = px.scatter(data, x = 'x', y = 'y', opacity=0.8, hover_name = data.index)
fig.update_layout(autosize=True,)
fig.update_traces(marker=dict(color='red'))
plot(fig, auto_open=True)
pio.write_html(fig, file='pca_msi_2d_data.html', auto_open=True)

#%%  PCA 3D

pca = PCA(n_components=3)
X_pca = pca.fit_transform(flattened_images)

data = pd.DataFrame({
        'x': X_pca[:, 0], 
        'y': X_pca[:, 1],
        'z': X_pca[:, 2]
    })

data['marker_size'] = 1

# fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], opacity=0.8)
# plot(fig, auto_open=True)
# pio.write_html(fig, file='pca_msi_3d_data.html', auto_open=True)

fig = px.scatter_3d(data, x = 'x', y = 'y', z = 'z', opacity=0.8, size = 'marker_size', hover_name = data.index)
fig.update_layout(autosize=True,)
fig.update_traces(marker=dict(color='red'))
plot(fig, auto_open=True)
pio.write_html(fig, file='pca_msi_3d_data.html', auto_open=True)
