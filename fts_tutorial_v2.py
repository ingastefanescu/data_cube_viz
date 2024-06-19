# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:23:46 2024

@author: Andrei
"""

#%%  Imports
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#%% Generate sample time series data
np.random.seed(42)
time_steps = 80
x = np.arange(time_steps)
y = np.sin(0.1 * x) + 0.2 * np.random.randn(time_steps)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Remaining code

#%% Determine the number of classes using Sturges' formula
num_classes = int(1 + np.log2(len(x_train)))  # Using only training data for this calculation


#%% Create fuzzy sets and determine membership
# universe_of_discourse = fuzz.gaussmf(x, np.mean(x), np.std(x))

# Determine the min and max of the time series
min_value = np.min(x)
max_value = np.max(x)

# Create fuzzy sets and determine membership
universe_of_discourse = fuzz.gaussmf(x, (max_value + min_value) / 2, (max_value - min_value) / 4)

#%% Fuzzify the input data
# Fuzzify the input data
fuzzy_data = np.zeros((time_steps, num_classes))
for i in range(time_steps):
    fuzzy_data[i, :] = fuzz.gaussmf(x, (max_value + min_value) / 2, (max_value - min_value) / 4)[i]


#%% Create fuzzy logical relations (FLRs)
flrs = []
for i in range(num_classes - 1):
    flr = np.zeros((num_classes, num_classes))
    flr[i, i+1] = 1
    flrs.append(flr)

#%% Group fuzzy logical relations
num_groups = 5
flrs_flat = [flr.flatten() for flr in flrs]

# Reshape each 1D array to (num_classes, num_classes)
flrs_reshaped = [flr.reshape((num_classes, num_classes)) for flr in flrs_flat]

# # Assuming flrs_reshaped is a list of 2D arrays
# for flr in flrs_reshaped:
#     if flr.ndim == 2:
#         # Reshape flr to a 1D array
#         flr = flr.flatten()

# # Now use flrs_reshaped in np.random.choice
# grouped_flrs = [np.random.choice(flr, (num_classes, num_classes), replace=True) for flr in flrs_reshaped]


# Assuming flrs_reshaped is a list of 2D arrays
flrs_flattened = [flr.flatten() for flr in flrs_reshaped]

# Now use flrs_flattened in np.random.choice
grouped_flrs = [np.random.choice(flr, (num_classes, num_classes), replace=True) for flr in flrs_flattened]

#%% Apply fuzzy rules to generate predictions
# Convert the list of arrays to a NumPy array
grouped_flrs_array = np.array(grouped_flrs)

# Now you can use the transpose operation
predictions = np.dot(fuzzy_data, grouped_flrs_array.T)

def centroid_defuzzification(fuzzy_set):
    # Assuming fuzzy_set is a 2D array
    # Calculate the centroid for each column using the weighted average
    centroids = np.sum(fuzzy_set * np.arange(fuzzy_set.shape[1]), axis=1) / np.sum(fuzzy_set, axis=1)
    return centroids


# Perform defuzzification
crisp_predictions = np.array([centroid_defuzzification(fuzzy_set) for fuzzy_set in predictions])

#%%

def centroid_defuzzification(fuzzy_set):
    # Assuming fuzzy_set is a 2D array
    sum_values = np.sum(fuzzy_set, axis=1)
    
    # Check for zero sum to avoid division by zero
    zero_sum_indices = np.where(sum_values == 0)
    non_zero_sum_indices = np.where(sum_values != 0)
    
    # Calculate centroids only for non-zero sum rows
    centroids = np.zeros_like(sum_values, dtype=float)
    centroids[non_zero_sum_indices] = np.sum(fuzzy_set[non_zero_sum_indices] * np.arange(fuzzy_set.shape[1]), axis=1) / sum_values[non_zero_sum_indices]

    return centroids

crisp_predictions = np.array([centroid_defuzzification(fuzzy_set) for fuzzy_set in predictions])

#%%
def centroid_defuzzification(fuzzy_values):
    centroid = np.sum(fuzzy_values * np.arange(len(fuzzy_values))) / np.sum(fuzzy_values)
    return centroid

# Apply defuzzification to each set of predictions
crisp_predictions = np.array([centroid_defuzzification(fuzzy_set) for fuzzy_set in predictions])

# Reshape crisp_predictions to match the shape of y_train
crisp_predictions_reshaped = crisp_predictions.reshape(y_train.shape)

