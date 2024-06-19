# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 02:06:38 2024

@author: Andrei
"""
#%% Imports
import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go

#%% Read the image
image_path = 'D:/OneDrive - Universitatea Politehnica Bucuresti/Andrei/Facultate/Master/DISSERTATION/work/msi/image_0001.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#%% Reshape the image to a 2D array (keeping the spatial structure)
reshaped_image = image

#%% Apply t-SNE
tsne = TSNE(n_components=3, random_state=42)
tsne_result = tsne.fit_transform(reshaped_image)

# # Plot the t-SNE result
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
# plt.title('t-SNE Result for Image Pixel Values (No Flattening)')
# plt.show()


#%% Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
data = tsne_result
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


#%% Perplexity
import numpy as np

perplexity = np.arange(5,100, 5)
divergence = []

for i in perplexity:
    model = TSNE(n_components=2, init="pca", perplexity=i)
    reduced = model.fit_transform(reshaped_image)
    divergence.append(model.kl_divergence_)
fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
fig.update_traces(line_color="red", line_width=1)
plot(fig, auto_open=True)
