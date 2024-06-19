# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:55:46 2024

@author: Andrei
"""

#%% Imports

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#%% Read CSV

# Path to csv
file_path = 'timeline SITS.xlsx'

# Read Excel file into a DataFrame
df = pd.read_excel(file_path)

# Drop the NDVI column
df = df.drop('Avg_NDVI',
             axis=1)

# Convert to datetime format with specified date format
df['Data'] = pd.to_datetime(df['Data'],
                            format='%d.%m.%Y') 

# Set the first column ('Date') as the index
df = df.set_index('Data')

# msi_array = df.values
#msi_array = df

#%% Extract features and target

x = np.arange(len(df))
y = df['Avg_MSI'].values

#%% Define fuzzy sets
fuzzy_input = np.arange(0, 1.14, 0.01)
fuzzy_membership = fuzz.gaussmf(fuzzy_input, np.mean(x), np.std(x))

#%% Fuzzify the input data
fuzzy_data = np.zeros((len(df), len(fuzzy_input)))
for i in range(len(df)):
    fuzzy_data[i, :] = fuzz.interp_membership(x, fuzzy_membership, i)

#%% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fuzzy_data[:-1, :], y[1:], test_size=0.2, random_state=42)

#%% Build the neural network model

model = Sequential()

# First layer + Input layer
model.add(Dense(100, input_dim=len(fuzzy_input), activation='relu'))

#%% Other hidden layers
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))

# Output layer
model.add(Dense(1, activation='linear'))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')


#%% Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

#%% Make predictions on the test set
predictions = model.predict(X_test).flatten()

#%% Plot the results
plt.plot(y_test, label='True values')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()