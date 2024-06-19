# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:10:54 2024

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

#%% Crop one msi image
image_path = 'D:/OneDrive - Universitatea Politehnica Bucuresti/Andrei/Facultate/Master/DISSERTATION/work/msi/image_0001.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(1)
plt.imshow(image, cmap = 'gray')

# Define the cropping coordinates
x1, x2 = 500, 1000
y1, y2 = 1950, 2300

# Initialize an empty array to store cropped images
cropped_img = image[y1:y2, x1:x2]

plt.figure(2)
plt.imshow(cropped_img, cmap= 'gray')

#%% Crop one ndvi image

image_path = 'D:/OneDrive - Universitatea Politehnica Bucuresti/Andrei/Facultate/Master/DISSERTATION/work/ndvi/image_0001.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(1)
plt.imshow(image, cmap = 'gray')

# Define the cropping coordinates
x1, x2 = 500, 1000
y1, y2 = 1950, 2300

# Initialize an empty array to store cropped images
cropped_img = image[y1:y2, x1:x2]

plt.figure(2)
plt.imshow(cropped_img, cmap= 'gray')