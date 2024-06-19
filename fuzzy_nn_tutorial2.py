# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:32:52 2024

@author: Andrei
"""

import numpy as np
import skfuzzy as fuzz
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y_true = 2 * np.sin(X) + np.random.normal(scale=0.5, size=len(X))

# Create fuzzy sets
def create_fuzzy_sets(data, num_sets):
    fuzzy_sets = []
    for i in range(num_sets):
        fuzzy_set = fuzz.gaussmf(data, np.mean(data), np.std(data))
        fuzzy_sets.append(fuzzy_set)
    return np.vstack(fuzzy_sets)

# Fuzzify the input data
fuzzy_inputs = create_fuzzy_sets(X, num_sets=5)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fuzzy_inputs.T, y_true, test_size=0.2, random_state=42)

# Create and train the fuzzy neural network
model = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.scatter(X_test[:, 0], y_test, label='True values')
plt.scatter(X_test[:, 0], y_pred, label='Predicted values')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Fuzzy Neural Network Prediction')
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y_true = 2 * np.sin(X) + np.random.normal(scale=0.5, size=len(X))

# Fuzzify the input data
def create_fuzzy_sets(data, num_sets):
    fuzzy_sets = []
    for i in range(num_sets):
        fuzzy_set = np.exp(-((data - np.mean(data)) / np.std(data))**2)
        fuzzy_sets.append(fuzzy_set)
    return np.vstack(fuzzy_sets)

fuzzy_inputs = create_fuzzy_sets(X, num_sets=5)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = fuzzy_inputs[:, :split_index], fuzzy_inputs[:, split_index:]
y_train, y_test = y_true[:split_index], y_true[split_index:]

# Neural Network Implementation
class FuzzyNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.rand(hidden_size, input_size)
        self.weights_hidden_output = np.random.rand(output_size, hidden_size)

        self.bias_hidden = np.zeros((hidden_size, 1))
        self.bias_output = np.zeros((output_size, 1))

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden_inputs = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        final_outputs = final_inputs  # Identity activation for regression

        return hidden_outputs, final_outputs

    def train(self, inputs, targets):
        hidden_outputs, final_outputs = self.forward(inputs)

        # Backpropagation
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # Update weights and biases
        self.weights_hidden_output += self.learning_rate * np.dot(output_errors, hidden_outputs.T)
        self.bias_output += self.learning_rate * output_errors.sum(axis=1, keepdims=True)

        self.weights_input_hidden += self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)
        self.bias_hidden += self.learning_rate * hidden_errors.sum(axis=1, keepdims=True)

    def predict(self, inputs):
        _, final_outputs = self.forward(inputs)
        return final_outputs

# Training the Fuzzy Neural Network
input_size = fuzzy_inputs.shape[0]
hidden_size = 10
output_size = 1

fnn = FuzzyNeuralNetwork(input_size, hidden_size, output_size)

epochs = 5000
for epoch in range(epochs):
    fnn.train(X_train, y_train.reshape(1, -1))

# Testing the trained model
y_pred = fnn.predict(X_test)

# Plot the results
plt.scatter(X_test[0], y_test, label='True values')
plt.scatter(X_test[0], y_pred.flatten(), label='Predicted values')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Fuzzy Neural Network Prediction')
plt.show()

