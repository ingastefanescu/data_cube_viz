#%% Imports

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
from PIL import ImageTk, Image
import seaborn as sns
import plotly.graph_objs as go
import warnings
import webbrowser 

#%% Global variables

warnings.filterwarnings("ignore", category=RuntimeWarning)    
pio.renderers.default = 'browser'  # Set the default renderer to 'browser'

image_dates = [
    "26.04.2017", "05.06.2017", "25.06.2017", "05.07.2017", "04.08.2017", "13.09.2017", "03.10.2017", "13.10.2017",
    "02.11.2017", "12.11.2017", "06.01.2018", "31.01.2018", "21.04.2018", "26.04.2018", "01.05.2018", "06.05.2018",
    "16.05.2018", "31.05.2018", "10.06.2018", "15.07.2018", "14.08.2018", "19.08.2018", "24.08.2018", "29.08.2018",
    "03.09.2018", "23.09.2018", "28.09.2018", "03.10.2018", "08.10.2018", "13.10.2018", "23.10.2018", "27.12.2018",
    "20.02.2019", "25.02.2019", "27.03.2019", "01.04.2019", "26.04.2019", "26.05.2019", "30.06.2019", "20.07.2019",
    "14.08.2019", "19.08.2019", "24.08.2019", "29.08.2019", "03.09.2019", "13.09.2019", "03.10.2019", "13.10.2019",
    "18.10.2019", "23.10.2019", "17.12.2019", "22.12.2019", "26.01.2020", "01.03.2020", "06.03.2020", "11.03.2020",
    "16.03.2020", "21.03.2020", "10.04.2020", "25.04.2020", "29.06.2020", "09.07.2020", "29.07.2020", "03.08.2020",
    "13.08.2020", "18.08.2020", "23.08.2020", "28.08.2020", "02.09.2020", "07.09.2020", "12.09.2020", "17.09.2020",
    "22.09.2020", "27.09.2020", "02.10.2020", "07.10.2020", "22.10.2020", "01.11.2020", "26.11.2020", "31.12.2020",
    "15.01.2021", "26.03.2021", "10.04.2021", "25.04.2021", "30.04.2021", "10.05.2021", "04.06.2021", "24.06.2021",
    "14.07.2021", "29.07.2021", "08.08.2021", "23.08.2021", "07.09.2021", "12.09.2021", "27.09.2021", "22.10.2021",
    "27.10.2021", "11.11.2021", "21.11.2021", "26.11.2021", "01.12.2021", "15.01.2022", "20.01.2022", "25.01.2022",
    "04.02.2022", "21.03.2022", "26.03.2022", "05.04.2022", "15.04.2022", "20.05.2022", "25.05.2022", "29.06.2022",
    "14.07.2022", "19.07.2022"
]


clicked_coordinates = None  # Store clicked (x, y) coordinates
flattened_images = None
selected_folder = None
selected_image = None
date_label = None
image_data = None
fig = go.Figure()  # Persistent Plotly figure object

# Initialize a list to store all plotted traces
all_traces = []

# HTML file name
html_filename = "Coordinates_time_series_plot.html"

# Flag to check if the HTML file has been created
html_created = False

#%% Functions

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

def load_image(index):
    global img_label, image_files, image_dates, date_label
    img_path = os.path.join(selected_folder, image_files[index])
    img = Image.open(img_path)

    img.thumbnail((600, 400), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    
    img_label.config(image=img)
    img_label.image = img
    img_label.image_index = index
    
    date_label.config(text=f"Date: {image_dates[index]}")

def open_image_navigation_window():
    global image_files, img_label, date_label
    
    if not selected_folder:
        messagebox.showwarning("No Folder Selected", "Please load the spectral index first.")
        return

    image_files = [f for f in os.listdir(selected_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        messagebox.showwarning("No Images Found", "No image files found in the selected folder.")
        return
    
    nav_window = tk.Toplevel(root)
    nav_window.title("Image Navigation")
    nav_window.geometry("800x600")
    nav_window.configure(bg="#e0f7fa")
    
    img_label = tk.Label(nav_window)
    img_label.pack(pady=20)

    date_label = tk.Label(nav_window,text="",bg="#e0f7fa", font=("Arial", 12, "bold"))
    date_label.pack(pady=10)
    
    def next_image():
        current_index = img_label.image_index
        next_index = (current_index + 1) % len(image_files)
        load_image(next_index)

    def prev_image():
        current_index = img_label.image_index
        prev_index = (current_index - 1) % len(image_files)
        load_image(prev_index)

    def go_back():
        nav_window.destroy()

    load_image(0)

    next_button = tk.Button(nav_window, text="Next", command=next_image, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
    next_button.pack(side="right", padx=20, pady=20)

    prev_button = tk.Button(nav_window, text="Previous", command=prev_image, bg="#FF5722", fg="white", font=("Arial", 12, "bold"))
    prev_button.pack(side="left", padx=20, pady=20)

    back_button = tk.Button(nav_window, text="Back", command=go_back, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
    back_button.pack(side="bottom", pady=20)

def select_and_display_image():
    global selected_folder  # Ensure selected_folder is global

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
    global clicked_coordinates

    print("Click event detected.")  # Debug statement

    if event.button == 1:  # Check if left mouse button is clicked
        x = int(event.xdata)
        y = int(event.ydata)
        clicked_coordinates = (x, y)  # Store clicked coordinates
        print(f"Clicked at (x, y) = ({x}, {y})")  # Debug statement
        plot_pixel_values(x, y)

def plot_pixel_values(x, y):
    global selected_folder, clicked_coordinates, image_dates, fig

    try:
        if clicked_coordinates is None:
            raise ValueError("No coordinates have been clicked.")

        # Initialize lists to store pixel values and image indices
        pixel_values = []

        # Extract the folder name from the selected_folder path
        folder_name = os.path.basename(selected_folder)

        # Load all images from selected_folder
        image_files = os.listdir(selected_folder)
        image_files = [os.path.join(selected_folder, file) for file in image_files]

        for file_path in image_files:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            if image is None:
                raise ValueError(f"Unable to read image from {file_path}")

            # Get pixel value at clicked coordinates
            pixel_value = image[y, x]
            pixel_values.append(pixel_value)

        # Normalize pixel values to [0, 1] range
        pixel_values = np.array(pixel_values)
        max_val = np.max(pixel_values)
        min_val = np.min(pixel_values)
        if max_val != min_val:
            normalized_values = (pixel_values - min_val) / (max_val - min_val)
        else:
            normalized_values = pixel_values  # Handle case where all values are the same

        print("Normalized Values:", normalized_values)  # Debug statement

        # Add trace to the persistent Plotly figure object
        fig.add_trace(go.Scatter(x=image_dates, y=normalized_values.ravel(),
                                 mode='lines+markers', name=f"{folder_name} - Pixel at ({x}, {y})"))
        fig.update_layout(title="Time Series Plots",
                          xaxis_title="Date", yaxis_title="Normalized Pixel Value",
                          template="plotly_white")
        fig.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"Error occurred: {e}")

def update_html_file(fig):
    global html_created
    
    if not html_created:
        # Write the initial figure to HTML file
        pio.write_html(fig, html_filename)
        html_created = True
        print(f"HTML file '{html_filename}' created.")
    else:
        # Append the new trace to the existing HTML file
        pio.write_html(fig, file=html_filename, auto_open=False)
        print(f"Updated HTML file '{html_filename}'.")

def plot_pixel_values_coord(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
        print(f"Received coordinates: ({x_coord}, {y_coord})")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid integer coordinates.")
        return
    
    # Collect data for plotting
    pixel_values = []
    for filename in sorted(os.listdir(selected_folder)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(selected_folder, filename)
            img = Image.open(image_path)
            img_array = np.array(img)
            pixel_value = img_array[y_coord, x_coord]  # Assuming (y, x) format for numpy array indexing

            # Normalize pixel value to range [0, 1]
            normalized_value = pixel_value / 255.0  # Assuming pixel values are in [0, 255] range
            pixel_values.append(normalized_value)

    # Create Plotly trace for current coordinates
    trace = go.Scatter(x=image_dates,
                       y=pixel_values,
                       mode='lines+markers',
                       name=f"{os.path.basename(selected_folder)} pixel at ({x_coord},{y_coord})")
    
    print(f"Created trace for coordinates: ({x_coord}, {y_coord})")

    # Add the trace to the list of all traces
    all_traces.append(trace)

    # Create a new Plotly figure with all traces
    fig = go.Figure(data=all_traces)

    # Update layout settings (if needed)
    fig.update_layout(
        title='Pixel Values Across Images',
        xaxis=dict(title='Image Dates'),
        yaxis=dict(title='Normalized Pixel Value (0-1)')
    )

    # Update the existing HTML file with the new figure
    update_html_file(fig)

    # Open the updated HTML file in the default web browser
    # webbrowser.open(html_filename, new=2)  # Commented out to avoid opening new tabs

def open_html_in_browser():
    webbrowser.open(html_filename)

def open_coordinates_window():
    coordinates_window = tk.Toplevel()
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x150")
    
    x_label = tk.Label(coordinates_window, text="Enter X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack(pady=5)
    
    y_label = tk.Label(coordinates_window, text="Enter Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack(pady=5)
    
    open_browser_button = tk.Button(coordinates_window, text="Open Browser", command=open_html_in_browser)
    open_browser_button.pack(pady=10)
    
    update_button = tk.Button(coordinates_window, text="Update", command=lambda: plot_pixel_values_coord(x_entry.get(), y_entry.get(), coordinates_window))
    update_button.pack(pady=10)
#%% Tk GUI

#%% Tkinter GUI

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

select_image_button = tk.Button(button_frame, text="Plot Time Series", command=select_and_display_image, bg="#FF5722", fg="white", font=("Arial", 12, "bold"))

open_tsne_window_button = tk.Button(button_frame, text="Open t-SNE Options", command=open_tsne_window, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))

open_pca_window_button = tk.Button(button_frame, text="Open PCA Options", command=open_pca_window, bg="#FF9800", fg="white", font=("Arial", 12, "bold"))

navigate_images_button = tk.Button(button_frame, text="Navigate Images", command=open_image_navigation_window, bg="#9C27B0", fg="white", font=("Arial", 12, "bold"))

enter_coord = tk.Button(button_frame, text="Enter Coordinates", command=open_coordinates_window, bg="#9C27B0", fg="white", font=("Arial", 12, "bold"))


load_button.pack(pady=10, fill='x')
select_image_button.pack(pady=10, fill='x')
navigate_images_button.pack(pady=10, fill = 'x')
open_tsne_window_button.pack(pady=10, fill='x')
open_pca_window_button.pack(pady=10, fill='x')
enter_coord.pack(pady=10, fill='x')

# Start the Tkinter event loop
root.mainloop()