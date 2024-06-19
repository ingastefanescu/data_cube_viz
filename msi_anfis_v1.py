# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 21:19:06 2024

@author: Andrei
"""

#%% Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#%% Read CSV

file_path = 'timeline SITS.xlsx'

# Read Excel file into a DataFrame
df = pd.read_excel(file_path)

# Drop the NDVI column
df = df.drop('Avg_NDVI', axis=1)

# Convert to datetime format with specified date format
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y') 

# Set the first column ('Date') as the index
df = df.set_index('Data')

# msi_array = df.values
#msi_array = df

#%% Train-Test Split

feature_column = 'Avg_MSI'
feature_vector = df[feature_column]

# Assuming 'msi_array' is your DataFrame with the first column as date and the second column as the feature vector
train_size = 0.8

# Splitting the dataset into training and test sets
# train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)
train_vector, test_vector = train_test_split(feature_vector, train_size=train_size, shuffle=False)

# # Separate features and target variable in the training data
# X_train = train_data.iloc[:, 1:]  # Assuming the second column onwards are features
# y_train = train_data.iloc[:, 0]   # Assuming the first column is the target variable

# # Separate features and target variable in the test data
# X_test = test_data.iloc[:, 1:]    # Assuming the second column onwards are features
# y_test = test_data.iloc[:, 0]     # Assuming the first column is the target variable

#%% ANFIS Model

min_output = min(df['Avg_MSI'])  # Minimum value of the output variable
max_output = max(df['Avg_MSI'])  # Maximum value of the output variable


# Create input and output universes for fuzzy sets
# input_universe = [np.min(X_train), np.max(X_train)]
input_universe = [0, 1]
# output_universe = [np.min(y_train), np.max(y_train)]
output_universe = [0, 1]

# Generate fuzzy input and output variables
input_var = ctrl.Antecedent(input_universe, 'input')
output_var = ctrl.Consequent(output_universe, 'output')

# Generate fuzzy membership functions for the input and output
input_var['low'] = fuzz.trimf(input_var.universe, [input_var.universe[0], np.percentile(input_var.universe, 25), input_var.universe[1]])
input_var['medium'] = fuzz.trimf(input_var.universe, [np.percentile(input_var.universe, 25), np.percentile(input_var.universe, 75), np.percentile(input_var.universe, 75)])
input_var['high'] = fuzz.trimf(input_var.universe, [np.percentile(input_var.universe, 75), input_var.universe[1], input_var.universe[1]])

output_var['low'] = fuzz.trimf(output_var.universe, [output_var.universe[0], np.percentile(output_var.universe, 25), output_var.universe[1]])
output_var['medium'] = fuzz.trimf(output_var.universe, [np.percentile(output_var.universe, 25), np.percentile(output_var.universe, 75), np.percentile(output_var.universe, 75)])
output_var['high'] = fuzz.trimf(output_var.universe, [np.percentile(output_var.universe, 75), output_var.universe[1], output_var.universe[1]])

# Generate fuzzy rules
rules = [
    ctrl.Rule(input_var['low'], output_var['low']),
    ctrl.Rule(input_var['medium'], output_var['medium']),
    ctrl.Rule(input_var['high'], output_var['high']),
    # Add more rules as needed
]

# Create the fuzzy system
fuzzy_system = ctrl.ControlSystem(rules)
anfis_model = ctrl.ControlSystemSimulation(fuzzy_system)

#%% Train the ANFIS model using the training data
for xi, yi in zip(X_train, y_train):
    anfis_model.input['input'] = xi
    anfis_model.compute()


#%% Use the trained model to make predictions on the test data
y_pred = []
for xi in X_test:
    anfis_model.input['input'] = xi
    fuzzy_system.compute(anfis_model)
    y_pred.append(anfis_model.output['output'])

# Display the predicted values
print("Predicted values:", y_pred)
