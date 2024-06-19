# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:28:07 2024

@author: Andrei
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog, Toplevel
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
from plotly.offline import plot
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

# Global variables
flattened_images = None
selected_folder = None
selected_image = None

def load_spectral_index():
    global flattened_images, selected_folder
    try:
        # Set the root folder to the current working directory
        root_folder = os.getcwd()

        # Open a dialog to select the subfolder within the root folder containing the images
        folder_path = filedialog.askdirectory(initialdir=root_folder, title="Select Folder Containing Images")
        
        # Check if a folder was selected
        if not folder_path:
            messagebox.showwarning("No Folder Selected", "Please select a folder to proceed.")
            return

        selected_folder = folder_path  # Store selected folder globally

        # Define the cropping coordinates
        x1, x2 = 500, 600
        y1, y2 = 1950, 2100

        # Initialize an empty list to store image values
        msi_data = []

        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if the file is an image
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Open the image using cv2
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = img[y1:y2, x1:x2]

                # Check if the image was successfully loaded
                if img is not None:
                    # Append the cropped image to the list
                    msi_data.append(img)

        # Convert the list of 2D arrays to a 3D NumPy array
        msi_data = np.array(msi_data)

        # Flatten each 2D array in image_values to make it compatible with t-SNE
        flattened_images = msi_data.reshape((len(msi_data), -1)).T

        # Notify the user that the process was successful
        messagebox.showinfo("Success", "Data Cube loaded successfully!")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def select_and_display_image():
    global selected_folder, selected_image

    try:
        if selected_folder is None:
            messagebox.showwarning("No Folder Selected", "Please load a data cube first.")
            return

        # Open a dialog to select an image file from the selected folder
        file_path = filedialog.askopenfilename(initialdir=selected_folder, title="Select Image File",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        
        # Check if a file was selected
        if not file_path:
            messagebox.showwarning("No Image Selected", "Please select an image file to proceed.")
            return

        # Read the selected image
        selected_image = cv2.imread(file_path)

        # Display the image using matplotlib with zoom capabilities
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size as needed
        ax.imshow(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
        ax.set_title("Selected Image")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.format_coord = lambda x, y: f'Row={int(y)}, Col={int(x)}'
        ax.autoscale(False)
        ax.set_xlim(0, selected_image.shape[1])
        ax.set_ylim(selected_image.shape[0], 0)
        plt.connect('button_press_event', on_click_image)  # Connect mouse click event
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

def on_click_image(event):
    global selected_image

    if selected_image is not None:
        col = int(event.xdata) if event.xdata else None
        row = int(event.ydata) if event.ydata else None

        if col is not None and row is not None:
            messagebox.showinfo("Pixel Clicked", f"Clicked at (row, col): ({row}, {col})")
        else:
            messagebox.showwarning("Outside Image", "Clicked outside the image area.")

def open_tsne_window():
    # Create a new Toplevel window
    top_level_window = Toplevel(root)
    top_level_window.title("t-SNE Options")
    
    # Apply the same background color as root window
    top_level_window.configure(bg="#e0f2f1")
    
    # Set the size of the window
    top_level_window.geometry("600x400")
    
    # Function to handle closing of additional window
    def close_tsne_window():
        top_level_window.destroy()
    
    def tsne_2d():
        global flattened_images
        try:
            if flattened_images is None:
                messagebox.showwarning("No Data Loaded", "Please load a data cube first.")
                return

            # Perform t-SNE in 2D
            tsne = TSNE(n_components=2, random_state=42)
            tsne_2d = tsne.fit_transform(flattened_images)
            
            # Create a DataFrame for the t-SNE 2D results
            data = pd.DataFrame({
                'x': tsne_2d[:, 0], 
                'y': tsne_2d[:, 1],
            })

            # Create the scatter plot with Plotly
            fig = px.scatter(data, x='x', y='y', opacity=0.8, hover_name=data.index)
            fig.update_layout(title="t-SNE 2D", autosize=True)
            fig.update_traces(marker=dict(color='red'))

            # Plot the figure
            plot(fig, auto_open=True)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def tsne_3d():
        global flattened_images
        try:
            if flattened_images is None:
                messagebox.showwarning("No Data Loaded", "Please load a data cube first.")
                return

            # Perform t-SNE in 3D
            tsne = TSNE(n_components=3, random_state=42, perplexity=25, n_iter=250)
            tsne_3d = tsne.fit_transform(flattened_images)
            
            # Create a DataFrame for the t-SNE 3D results
            data = pd.DataFrame({
                'x': tsne_3d[:, 0], 
                'y': tsne_3d[:, 1],
                'z': tsne_3d[:, 2]
            })

            data['marker_size'] = 1  # Adjust marker size if needed

            # Create the 3D scatter plot with Plotly
            fig = px.scatter_3d(data, x='x', y='y', z='z', opacity=0.8, size='marker_size', hover_name=data.index)
            fig.update_layout(title="t-SNE 3D", autosize=True)
            fig.update_traces(marker=dict(color='blue'))  # Change marker color if desired

            # Plot the figure
            plot(fig, auto_open=True)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def perform_tsne_perplexity_analysis():
        if flattened_images is None:
            messagebox.showwarning("No Data", "Please load a data cube first.")
            return
        
        perplexity = np.arange(5, 100, 5)
        divergence = []
        
        for i in perplexity:
            model = TSNE(n_components=2, init="pca", perplexity=i)
            reduced = model.fit_transform(flattened_images)
            divergence.append(model.kl_divergence_)
        
        fig = px.line(x=perplexity, y=divergence, markers=True)
        fig.update_layout(title='2D t-SNE', xaxis_title="Perplexity Values", yaxis_title="Divergence")
        fig.update_traces(line_color="red", line_width=1)
        
        plot(fig, auto_open=True)
        pio.write_html(fig, file='tsne_msi_2d_perp.html', auto_open=True)
        
    
    def perform_tsne_perplexity_analysis_3D():
        if flattened_images is None:
            messagebox.showwarning("No Data", "Please load a data cube first.")
            return
        
        perplexity = np.arange(5, 100, 5)
        divergence = []
        
        for i in perplexity:
            model = TSNE(n_components=3, init="pca", perplexity=i)
            reduced = model.fit_transform(flattened_images)
            divergence.append(model.kl_divergence_)
        
        fig = px.line(x=perplexity, y=divergence, markers=True)
        fig.update_layout(title='3D t-sne', xaxis_title="Perplexity Values", yaxis_title="Divergence")
        fig.update_traces(line_color="red", line_width=1)
        
        plot(fig, auto_open=True)
        pio.write_html(fig, file='tsne_msi_2d_perp.html', auto_open=True)
    
    # Create buttons in the additional window
    tsne_2d_button = tk.Button(top_level_window, text="t-SNE 2D", command=tsne_2d, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
    
    tsne_3d_button = tk.Button(top_level_window, text="t-SNE 3D", command=tsne_3d, bg="#FF5722", fg="white", font=("Arial", 12, "bold"))
    
    close_button = tk.Button(top_level_window, text="Close Window", command=close_tsne_window, bg="#e0f2f1", font=("Arial", 12, "bold"))
    
    tsne_perplexity_2D_button = tk.Button(top_level_window, text="t-SNE 2D Perplexity Analysis", command=perform_tsne_perplexity_analysis, bg="#9C27B0", fg="white", font=("Arial", 12, "bold"))
    
    tsne_perplexity_3D_button = tk.Button(top_level_window, text="t-SNE 3D Perplexity Analysis", command=perform_tsne_perplexity_analysis_3D, bg="#9C27B0", fg="white", font=("Arial", 12, "bold"))

    
    # Pack buttons in the additional window
    tsne_2d_button.pack(pady=10)
    tsne_3d_button.pack(pady=10)
    tsne_perplexity_2D_button.pack(pady=10)
    tsne_perplexity_3D_button.pack(pady=10)
    close_button.pack(pady=20)
    
def open_pca_window():
    # Create a new Toplevel window for PCA options
    top_level_pca_window = Toplevel(root)
    top_level_pca_window.title("PCA Options")
    
    # Apply the same background color as root window
    top_level_pca_window.configure(bg="#e0f2f1")
    
    # Set the size of the window
    top_level_pca_window.geometry("600x400")
    
    # Function to handle closing of additional window
    def close_pca_window():
        top_level_pca_window.destroy()
    
    def pca_2d():
        global flattened_images
        try:
            if flattened_images is None:
                messagebox.showwarning("No Data Loaded", "Please load a data cube first.")
                return

            # Perform PCA in 2D
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(flattened_images)
            
            # Create a DataFrame for the PCA 2D results
            data = pd.DataFrame({
                'x': X_pca[:, 0], 
                'y': X_pca[:, 1],
            })

            # Create the scatter plot with Plotly
            fig = px.scatter(data, x='x', y='y', opacity=0.8, hover_name=data.index)
            fig.update_layout(title="PCA 2D", autosize=True)
            fig.update_traces(marker=dict(color='red'))

            # Plot the figure
            plot(fig, auto_open=True)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def pca_3d():
        global flattened_images
        try:
            if flattened_images is None:
                messagebox.showwarning("No Data Loaded", "Please load a data cube first.")
                return

            # Perform PCA in 3D
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(flattened_images)
            
            # Create a DataFrame for the PCA 3D results
            data = pd.DataFrame({
                'x': X_pca[:, 0], 
                'y': X_pca[:, 1],
                'z': X_pca[:, 2]
            })

            data['marker_size'] = 1  # Adjust marker size if needed

            # Create the 3D scatter plot with Plotly
            fig = px.scatter_3d(data, x='x', y='y', z='z', opacity=0.8, size='marker_size', hover_name=data.index)
            fig.update_layout(title="PCA 3D", autosize=True)
            fig.update_traces(marker=dict(color='blue'))  # Change marker color if desired

            # Plot the figure
            plot(fig, auto_open=True)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def perform_information_gain():
        if flattened_images is None:
            messagebox.showwarning("No Data", "Please load a data cube first.")
            return
        
        n_components = 20
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(flattened_images)
    
        # Get the explained variance ratios
        explained_var = pca.explained_variance_ratio_
    
        # Compute the cumulative variance
        cumulative_var = np.cumsum(explained_var)
    
        # Plot the information gain for each component
        plt.plot(range(1, n_components + 1), cumulative_var, label='Cumulative')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('Information Gain for Each Principal Component')
        plt.legend()
        plt.show()
        
    # Create buttons in the additional window
    pca_2d_button = tk.Button(top_level_pca_window, text="PCA 2D", command=pca_2d, bg="#FF9800", fg="white", font=("Arial", 12, "bold"))
    pca_3d_button = tk.Button(top_level_pca_window, text="PCA 3D", command=pca_3d, bg="#FF5722", fg="white", font=("Arial", 12, "bold"))
    pca_info_button = tk.Button(top_level_pca_window, text="Number of components for PCA", command=perform_information_gain, bg="#FF5722", fg="white", font=("Arial", 12, "bold"))
    
    # White font, font=("Arial", 12, "bold"))
    close_button = tk.Button(top_level_pca_window, text="Close Window", command=close_pca_window, bg="#e0f2f1", font=("Arial", 12, "bold"))
    
    # Pack buttons in the additional window
    pca_2d_button.pack(pady=10)
    pca_3d_button.pack(pady=10)
    pca_info_button.pack(pady=10)    
    close_button.pack(pady=20)

    # Event handler to get coordinates when clicking on the image
    global selected_image

    if selected_image is not None:
        col = int(event.xdata) if event.xdata else None
        row = int(event.ydata) if event.ydata else None

        if col is not None and row is not None:
            messagebox.showinfo("Pixel Clicked", f"Clicked at (row, col): ({row}, {col})")
        else:
            messagebox.showwarning("Outside Image", "Clicked outside the image area.")

    # Event handler to get coordinates when clicking on the image
    if selected_image is not None:
        col = int(event.xdata) if event.xdata else None
        row = int(event.ydata) if event.ydata else None

        if col is not None and row is not None:
            messagebox.showinfo("Pixel Clicked", f"Clicked at (row, col): ({row}, {col})")
        else:
            messagebox.showwarning("Outside Image", "Clicked outside the image area.")

# Create the main Tkinter window
root = tk.Tk()
root.title("Data Cube Visualization")
root.geometry("600x500")
root.configure(bg="#e0f2f1")

# Create a frame for buttons
button_frame = tk.Frame(root, bg="#e0f2f1")
button_frame.pack(pady=20)

# Create and place the buttons
load_button = tk.Button(button_frame, text="Load Data Cube", command=load_spectral_index, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
select_image_button = tk.Button(button_frame, text="Select Image", command=select_and_display_image, bg="#FF5722", fg="white", font=("Arial", 12, "bold"))
open_tsne_window_button = tk.Button(button_frame, text="Open t-SNE Options", command=open_tsne_window, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
open_pca_window_button = tk.Button(button_frame, text="Open PCA Options", command=open_pca_window, bg="#FF9800", fg="white", font=("Arial", 12, "bold"))

load_button.pack(pady=10, fill='x')
select_image_button.pack(pady=10, fill='x')
open_tsne_window_button.pack(pady=10, fill='x')
open_pca_window_button.pack(pady=10, fill='x')

# Start the Tkinter event loop
root.mainloop()