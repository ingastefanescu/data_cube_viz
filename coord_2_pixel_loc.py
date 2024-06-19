# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:21:21 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

# Example coordinates (to be updated based on user input)
pixel_x = None
pixel_y = None

def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x150")
    coordinates_window.configure(bg="#9C27B0")
    
    x_label = tk.Label(coordinates_window, text="X Coordinate:", bg="#9C27B0", fg="white")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack(pady=5)
    
    y_label = tk.Label(coordinates_window, text="Y Coordinate:", bg="#9C27B0", fg="white")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack(pady=5)
    
    plot_button = tk.Button(coordinates_window, text="Plot Pixel Location", command=lambda: plot_pixel_location(x_entry.get(), y_entry.get(), coordinates_window))
    plot_button.pack(pady=10)

def plot_pixel_location(x_str, y_str, coordinates_window):
    global pixel_x, pixel_y
    
    try:
        pixel_x = int(x_str)
        pixel_y = int(y_str)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid integer coordinates.")
        return
    
    # Load the first image from the folder (change this as needed for your application)
    image_files = [f for f in os.listdir(selected_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        first_image_path = os.path.join(selected_folder, image_files[0])
        img = Image.open(first_image_path).convert('L')  # Convert to grayscale
        
        # Create matplotlib figure
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')  # Plot grayscale image
        
        # Plot a red dot on the specified pixel coordinates
        ax.scatter(pixel_x, pixel_y, color='red', s=100)
        
        # Configure plot aesthetics
        ax.set_title(f"First Image with Pixel Location ({pixel_x}, {pixel_y})")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Display the plot
        plt.show()
    else:
        messagebox.showwarning("No Images Found", "No image files found in the selected folder.")
    
    # Close the coordinates window after plotting
    coordinates_window.destroy()

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.geometry("400x300")
root.configure(bg="#9C27B0")

open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

root.mainloop()