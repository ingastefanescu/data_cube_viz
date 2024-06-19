# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:29:26 2024

@author: Andrei
"""
#%% Imports
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#%% Generate sample time series data
np.random.seed(42)
time_steps = 100
x = np.arange(time_steps)
y = np.sin(0.1 * x) + 0.2 * np.random.randn(time_steps)

#%% Determine the number of classes using Sturges' formula
num_classes = int(1 + np.log2(len(x)))

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

#%% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fuzzy_data[:-1, :], y[1:], test_size=0.2, random_state=42)

#%% Build a simple neural network
model = Sequential()
model.add(Dense(10, input_dim=num_classes, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

#%% Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

#%% Make predictions on the test set using the neural network
nn_predictions = model.predict(X_test).flatten()

#%% Plot the results
# Assuming y_test is a 1D array
y_test_1d = y_test if y_test.ndim == 1 else y_test[:, 0]

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(y_test_1d, label='True values')
plt.plot(predictions[1,1,1], label='Fuzzy Time Series Prediction')  # Assuming you want to plot only the first column of predictions
plt.legend()
plt.title('Fuzzy Time Series Predictions')


plt.subplot(2, 1, 2)
plt.plot(y_test_1d, label='True values')
plt.plot(nn_predictions, label='Neural Network Predictions')
plt.legend()
plt.title('Neural Network Predictions')

plt.tight_layout()
plt.show()
