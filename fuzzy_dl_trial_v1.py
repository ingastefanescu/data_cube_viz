# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:58:34 2024

@author: Andrei
"""

import numpy as np
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Generate sample time series data
np.random.seed(42)
time_steps = 100
x = np.arange(time_steps)
y = np.sin(0.1 * x) + 0.2 * np.random.randn(time_steps)

# Define fuzzy sets
fuzzy_input = np.arange(0, 1.00, 0.01)
fuzzy_membership = fuzz.gaussmf(fuzzy_input, np.mean(x), np.std(x))

# Fuzzify the input data
fuzzy_data = np.zeros((time_steps, len(fuzzy_input)))
for i in range(time_steps):
    fuzzy_data[i, :] = fuzz.interp_membership(x, fuzzy_membership, i)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fuzzy_data[:-1, :], y[1:], test_size=0.2, random_state=42)

# Build a simple neural network
model = Sequential()
model.add(Dense(10, input_dim=len(fuzzy_input), activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

# Make predictions on the test set
predictions = model.predict(X_test).flatten()

# Plot the results
plt.plot(y_test, label='True values')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
