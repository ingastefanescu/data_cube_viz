# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:19:17 2024

@author: Andrei
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

def display_image_with_crop(image_path, x1, x2, y1, y2):
    # Open the image using PIL (Pillow)
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    image_gray = image.convert('L')
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Display the grayscale image with the specified crop region
    ax.imshow(image_gray, cmap='gray')

    # Define the coordinates of the crop region
    crop_x1, crop_x2 = min(x1, x2), max(x1, x2)
    crop_y1, crop_y2 = min(y1, y2), max(y1, y2)

    # Calculate the width and height of the crop region
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    # Create a rectangle patch for the crop region
    rect = patches.Rectangle((crop_x1, crop_y1), crop_width, crop_height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the rectangle patch to the plot
    ax.add_patch(rect)

    # Set the title and show the plot
    plt.title(f"Cropped Region: x1={crop_x1}, x2={crop_x2}, y1={crop_y1}, y2={crop_y2}")
    plt.show()

# Load image
work_dir = os.getcwd()
folder_path = work_dir + '/ndvi'
image_path = folder_path + '\image_0001.jpg' # Replace with the path to your image file

# Define Crop Region
x1, x2 = 500, 600
y1, y2 = 1950, 2100

# Display the image with the specified crop region
display_image_with_crop(image_path, x1, x2, y1, y2)
