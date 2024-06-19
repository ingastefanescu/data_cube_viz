# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:11:54 2024

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

#%% Train-Test Split

feature_column = 'Avg_MSI'
feature_vector = df[feature_column]

# Assuming 'msi_array' is your DataFrame with the first column as date and the second column as the feature vector
train_size = 0.8

# Splitting the dataset into training and test set
# train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)
train_vector, test_vector = train_test_split(feature_vector,
                                             train_size = train_size,
                                             shuffle = False)

#%% ANFIS Model

# Create input and output universes for fuzzy sets
input_universe = [0, 1]
output_universe = [0, 1]

# Generate fuzzy input and output variables (antecedent and consequent variables)
input_var = ctrl.Antecedent(np.arange(*input_universe, step=0.1),
                            'input')

output_var = ctrl.Consequent(np.arange(*output_universe, step=0.1),
                             'output')

# Generate fuzzy membership functions for the input and output
input_var['low']     = fuzz.trimf(input_var.universe,
                                  [input_universe[0],
                                   np.percentile(input_universe, 25),
                                   input_universe[1]]
                                  )

input_var['medium']  = fuzz.trimf(input_var.universe,
                                  [np.percentile(input_universe, 25),
                                   np.percentile(input_universe, 75),
                                   np.percentile(input_universe, 75)])

input_var['high']    = fuzz.trimf(input_var.universe,
                                  [np.percentile(input_universe, 75),
                                   input_universe[1],
                                   input_universe[1]])

output_var['low']     = fuzz.trimf(output_var.universe,
                                   [output_universe[0],
                                    np.percentile(output_universe, 25),
                                    output_universe[1]])

output_var['medium']  = fuzz.trimf(output_var.universe,
                                   [np.percentile(output_universe, 25),
                                    np.percentile(output_universe, 75),
                                    np.percentile(output_universe, 75)])

output_var['high']    = fuzz.trimf(output_var.universe,
                                   [np.percentile(output_universe, 75), 
                                    output_universe[1],
                                    output_universe[1]])

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
for xi, yi in zip(train_vector.index, train_vector.values):
    anfis_model.input['input'] = yi
    anfis_model.compute()

# #%% Test: Predict next value
# # Your model is now trained, and you can use it to make predictions on test data
# # For example, to predict the next value in the time series:
# predicted_value = anfis_model.output['output']
# print("Predicted Value:", predicted_value)

#%% Test: Predict values from test 

# Initialize an empty list to store predicted values
predictions = []

# Test the ANFIS model using the test data
for xi, yi in zip(test_vector.index, test_vector.values):
    anfis_model.input['input'] = yi
    anfis_model.compute()
    
    # Get the predicted output value and append it to the list
    predicted_output = anfis_model.output['output']
    predictions.append(predicted_output)

# Convert the list of predictions to a NumPy array or a pandas Series if needed
predictions_array = np.array(predictions)

# Now, 'predictions_array' contains the predicted output values for the test data

#%% Test Metrics

# Calculate regression metrics
mse = mean_squared_error(test_vector.values, predictions_array)
mae = mean_absolute_error(test_vector.values, predictions_array)
r2 = r2_score(test_vector.values, predictions_array)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')

# #%% MSE for each date from the test array

# # Create a DataFrame with the predictions and the corresponding dates
# predictions_df = pd.DataFrame(data={'Predictions': predictions_array}, index=test_vector.index)

# # Ensure the indices are aligned
# test_vector = test_vector.loc[predictions_df.index]

# # Print the values of date and test_vector[date] to identify the issue
# for date, prediction in zip(test_vector.index, predictions_df['Predictions']):
#     print(f'Date: {date}, Actual Values: {test_vector[date]}, Prediction: {prediction}')

# # Calculate MSE for each date
# mse_values = [mean_squared_error(test_vector[date], prediction) for date, prediction in zip(test_vector.index, predictions_df['Predictions'])]

# # Plot the MSE values for each date
# plt.figure(figsize=(10, 6))
# plt.plot(test_vector.index, mse_values, label='MSE', color='green', marker='o')

# plt.xlabel('Date')
# plt.ylabel('MSE Value')
# plt.legend()
# plt.title('Mean Squared Error (MSE) for Each Date (Test Data)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# #%% Calculate MSE for each day
# mse_values = [mean_squared_error(test_vector['Avg_MSI'], predictions_array[:i+1]) for i in range(len(predictions_array))]

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(test_vector.index, mse_values, marker='o', label='MSE')
# plt.title('Mean Squared Error (MSE) for Each Day')
# plt.xlabel('Date')
# plt.ylabel('MSE')
# plt.legend()
# plt.grid(True)
# plt.show()

#%% Scatter Plot

# Plot the test_vector versus the predictions
plt.scatter(test_vector.index, test_vector.values, label='Actual', color='blue')
plt.scatter(test_vector.index, predictions_array, label='Predicted', color='red')
plt.xlabel('Index')  # Change this to the appropriate label for your x-axis
plt.ylabel('Value')  # Change this to the appropriate label for your y-axis
plt.legend()
plt.title('Actual vs. Predicted')
plt.show()

#%%  Plot the test_vector versus the predictions using a line plot

plt.plot(test_vector.index, test_vector.values, label='Actual', color='blue', marker='o')
plt.plot(test_vector.index, predictions_array, label='Predicted', color='red', marker='x')
plt.xlabel('Time')  # Change this to the appropriate label for your x-axis
plt.ylabel('Value')  # Change this to the appropriate label for your y-axis
plt.legend()
plt.title('Actual vs. Predicted Time Series')
plt.show()

#%% Plot train predictions

# Initialize an empty list to store predicted values for training data
train_predictions = []

# Predict on the training data
for xi, yi in zip(train_vector.index, train_vector.values):
    anfis_model.input['input'] = yi
    anfis_model.compute()
    
    # Get the predicted output value and append it to the list
    predicted_output = anfis_model.output['output']
    train_predictions.append(predicted_output)


# Convert the list of predictions for training data to a NumPy array
train_predictions_array = np.array(train_predictions)
train_vector1 = train_vector[:80]

# Calculate regression metrics
mse = mean_squared_error(train_vector1.values, train_predictions)
mae = mean_absolute_error(train_vector1.values, train_predictions)
r2 = r2_score(train_vector1.values, train_predictions)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')


# Plot the training data versus the predicted training data using a line plot
plt.plot(train_vector1.index, train_vector1.values, label='Actual (Train)', color='blue', marker='o')
plt.plot(train_vector1.index, train_predictions_array, label='Predicted (Train)', color='green', marker='x')
plt.xlabel('Time')  # Change this to the appropriate label for your x-axis
plt.ylabel('Value')  # Change this to the appropriate label for your y-axis
plt.legend()
plt.title('Actual vs. Predicted Time Series (Training Data)')
plt.show()

