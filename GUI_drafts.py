# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:34:43 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import graph_objs as go
from plotly import io as pio
import webbrowser  # Import the webbrowser module

# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

all_traces = []

def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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
    html_filename = "plot.html"
    pio.write_html(fig, html_filename)

    # Open the updated HTML file in the default web browser
    webbrowser.open_new_tab(html_filename)

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()

#%%

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:56:49 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import graph_objs as go
from plotly import io as pio
import webbrowser  # Import the webbrowser module


# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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



# Initialize a list to store all plotted traces
all_traces = []

# Flag to check if the HTML file has been created
html_created = False

# HTML file name
html_filename = "plot.html"

def update_html_file(fig):
    global html_created
    
    if not html_created:
        # Write the initial figure to HTML file
        pio.write_html(fig, html_filename)
        html_created = True
    else:
        # Append the new trace to the existing HTML file
        pio.write_html(fig, file=html_filename, auto_open=False)


def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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
    webbrowser.open(html_filename, new=2)  # new=2 opens in a new tab if possible


# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()

#%%

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:01:33 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import io as pio

# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

# Initialize a list to store all plotted traces
all_traces = []

# Create initial Plotly figure
fig = go.Figure()

# Initialize Plotly figure layout
fig.update_layout(
    title='Pixel Values Across Images',
    xaxis=dict(title='Image Dates'),
    yaxis=dict(title='Normalized Pixel Value (0-1)')
)

# Function to update Plotly figure with new coordinates
def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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

    # Add the trace to the list of all traces
    all_traces.append(trace)

    # Add the trace to the figure
    fig.add_trace(trace)

    # Update the display of the figure
    pio.show(fig, validate=False)  # Setting validate=False avoids unnecessary validation warnings


# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Function to open coordinates window
def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x200")

    x_label = tk.Label(coordinates_window, text="X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack()

    y_label = tk.Label(coordinates_window, text="Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack()

    plot_button = tk.Button(coordinates_window, text="Plot Pixel Values", command=lambda: plot_pixel_values(x_entry.get(), y_entry.get(), coordinates_window))
    plot_button.pack(pady=20)


# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()

#%%

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import io as pio

# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

# Initialize a list to store all plotted traces
all_traces = []

# Initialize Plotly figure
fig = go.Figure()

# Initialize Plotly figure layout
fig.update_layout(
    title='Pixel Values Across Images',
    xaxis=dict(title='Image Dates'),
    yaxis=dict(title='Normalized Pixel Value (0-1)')
)

# Function to update Plotly figure with new coordinates
def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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

    # Add the trace to the list of all traces
    all_traces.append(trace)

    # Clear existing traces from the figure
    fig.data = []

    # Add all traces to the figure
    for t in all_traces:
        fig.add_trace(t)

    # Serve the updated figure on a local server
    if len(all_traces) > 0:
        pio.show(fig, validate=False, url="http://localhost:8050")  # Serve on localhost at port 8050


# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Function to open coordinates window
def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x200")

    x_label = tk.Label(coordinates_window, text="X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack()

    y_label = tk.Label(coordinates_window, text="Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack()

    plot_button = tk.Button(coordinates_window, text="Plot Pixel Values", command=lambda: plot_pixel_values(x_entry.get(), y_entry.get(), coordinates_window))
    plot_button.pack(pady=20)


# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()

#%%

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import io as pio

# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

# Initialize Plotly figure
fig = go.Figure()

# Initialize a list to store all plotted traces
all_traces = []

# Function to update the Plotly figure with new coordinates
def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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

    # Add the trace to the list of all traces
    all_traces.append(trace)

    # Clear existing traces from the figure
    fig.data = []

    # Add all traces to the figure
    for t in all_traces:
        fig.add_trace(t)

    # Serve the updated figure on a local server
    if len(all_traces) > 0:
        pio.show(fig, validate=False)  # This should update the existing plot in the same tab

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Function to open coordinates window
def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x200")

    x_label = tk.Label(coordinates_window, text="X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack()

    y_label = tk.Label(coordinates_window, text="Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack()

    plot_button = tk.Button(coordinates_window, text="Plot Pixel Values", command=lambda: plot_pixel_values(x_entry.get(), y_entry.get(), coordinates_window))
    plot_button.pack(pady=20)

# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()

#%%

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:13:04 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import graph_objs as go
from plotly import io as pio
import webbrowser  # Import the webbrowser module


# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

# Initialize a list to store all plotted traces
all_traces = []

# HTML file name
html_filename = "Coordinates_time_series_plot.html"

def update_html_file(fig):
    # Write the figure to HTML file
    pio.write_html(fig, html_filename)

def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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
    webbrowser.open(html_filename, new=2)  # new=2 opens in a new tab if possible

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Function to open coordinates window
def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x150")
    coordinates_window.configure(bg="#e0f2f1")

    # Create labels and entry fields for coordinates
    x_label = tk.Label(coordinates_window, text="X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack()

    y_label = tk.Label(coordinates_window, text="Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack()

    # Create a button to plot pixel values
    plot_button = tk.Button(coordinates_window, text="Plot Pixel Values", command=lambda: plot_pixel_values(x_entry.get(), y_entry.get(), coordinates_window))
    plot_button.pack(pady=10)

# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()

#%%

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:16:02 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import graph_objs as go
from plotly import io as pio
import webbrowser  # Import the webbrowser module


# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

# Initialize a list to store all plotted traces
all_traces = []

# HTML file name
html_filename = "Coordinates_time_series_plot.html"

def update_html_file(fig):
    # Write the figure to HTML file
    pio.write_html(fig, html_filename)

def open_html_in_browser():
    # Open the HTML file in the default web browser
    webbrowser.open(html_filename, new=2)  # new=2 opens in a new tab if possible

def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Function to open coordinates window
def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x150")
    coordinates_window.configure(bg="#e0f2f1")

    # Create labels and entry fields for coordinates
    x_label = tk.Label(coordinates_window, text="X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack()

    y_label = tk.Label(coordinates_window, text="Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack()

    def enter_first_point():
        open_html_in_browser()

    def update():
        plot_pixel_values(x_entry.get(), y_entry.get(), coordinates_window)

    # Create a button to open HTML in browser
    open_first_point_button = tk.Button(coordinates_window, text="Enter First Point", command=enter_first_point)
    open_first_point_button.pack(pady=10)

    # Create a button to plot pixel values
    update_button = tk.Button(coordinates_window, text="Update", command=update)
    update_button.pack(pady=10)

# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()


#%%

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:17:09 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import graph_objs as go
from plotly import io as pio
import webbrowser  # Import the webbrowser module


# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

# Initialize a list to store all plotted traces
all_traces = []

# HTML file name
html_filename = "Coordinates_time_series_plot.html"

def update_html_file(fig):
    # Write the figure to HTML file
    pio.write_html(fig, html_filename)

def open_html_in_browser():
    # Open the HTML file in the default web browser
    webbrowser.open(html_filename, new=2)  # new=2 opens in a new tab if possible

def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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
                       name=f"Pixel at ({x_coord},{y_coord})")  # Name the trace appropriately

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

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Function to open coordinates window
def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x150")
    coordinates_window.configure(bg="#e0f2f1")

    # Create labels and entry fields for coordinates
    x_label = tk.Label(coordinates_window, text="X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack()

    y_label = tk.Label(coordinates_window, text="Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack()

    def enter_first_point():
        open_html_in_browser()

    def update():
        plot_pixel_values(x_entry.get(), y_entry.get(), coordinates_window)

    # Create a button to open HTML in browser
    open_first_point_button = tk.Button(coordinates_window, text="Enter First Point", command=enter_first_point)
    open_first_point_button.pack(pady=10)

    # Create a button to plot pixel values
    update_button = tk.Button(coordinates_window, text="Update", command=update)
    update_button.pack(pady=10)

# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()



#%%

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:25:48 2024

@author: Andrei
"""

import os
import tkinter as tk
from tkinter import messagebox
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from plotly import graph_objs as go
from plotly import io as pio
import webbrowser  # Import the webbrowser module

# Global variables
selected_folder = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"

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

# Initialize a list to store all plotted traces
all_traces = []

# HTML file name
html_filename = "Coordinates_time_series_plot.html"

def update_html_file(fig):
    # Write the figure to HTML file
    pio.write_html(fig, html_filename)

def open_html_in_browser():
    # Clear the contents of the HTML file (overwrite with an empty string)
    with open(html_filename, 'w') as f:
        f.write('')
    
    # Open the HTML file in the default web browser
    webbrowser.open(html_filename, new=2)  # new=2 opens in a new tab if possible

def plot_pixel_values(x_str, y_str, coordinates_window):
    global all_traces

    try:
        x_coord = int(x_str)
        y_coord = int(y_str)
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
                       name=f"Pixel at ({x_coord},{y_coord})")  # Name the trace appropriately

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

# Initialize tkinter window
root = tk.Tk()
root.title("Image Viewer")
root.configure(bg="#e0f2f1")
root.geometry("400x300")

# Function to open coordinates window
def open_coordinates_window():
    coordinates_window = tk.Toplevel(root)
    coordinates_window.title("Enter Coordinates")
    coordinates_window.geometry("300x150")
    coordinates_window.configure(bg="#e0f2f1")

    # Create labels and entry fields for coordinates
    x_label = tk.Label(coordinates_window, text="X Coordinate:")
    x_label.pack(pady=10)
    x_entry = tk.Entry(coordinates_window)
    x_entry.pack()

    y_label = tk.Label(coordinates_window, text="Y Coordinate:")
    y_label.pack(pady=10)
    y_entry = tk.Entry(coordinates_window)
    y_entry.pack()

    def open_browser():
        open_html_in_browser()

    def update():
        plot_pixel_values(x_entry.get(), y_entry.get(), coordinates_window)

    # Create a button to open HTML in browser
    open_browser_button = tk.Button(coordinates_window, text="Open Browser", command=open_browser)
    open_browser_button.pack(pady=10)

    # Create a button to plot pixel values
    update_button = tk.Button(coordinates_window, text="Update", command=update)
    update_button.pack(pady=10)

# Create a button to open coordinates window
open_coordinates_button = tk.Button(root, text="Open Coordinates Window", command=open_coordinates_window)
open_coordinates_button.pack(pady=50)

# Start tkinter main loop
root.mainloop()
