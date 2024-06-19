# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:32:45 2024

@author: Andrei
"""

import numpy as np

class FuzzyLayer:
    def __init__(self, num_nodes, input_size):
        self.weights = np.random.rand(num_nodes, input_size)
        self.membership_functions = []

    def forward(self, x):
        # Apply membership functions
        self.membership_functions = np.exp(-((x[:, np.newaxis] - self.weights) ** 2))
        return self.membership_functions

class MultiplyLayer:
    def __init__(self):
        pass

    def forward(self, layer1_output, layer2_output):
        return layer1_output * layer2_output[:, np.newaxis]

class NormalizationLayer:
    def __init__(self):
        pass

    def forward(self, x):
        return x / np.sum(x, axis=1, keepdims=True)

class DefuzzificationLayer:
    def __init__(self, num_rules, num_features):
        self.weights = np.random.rand(1, num_rules, num_features)

    def forward(self, x):
        return np.sum(x * self.weights, axis=2).squeeze()




class OutputLayer:
    def __init__(self):
        pass

    def forward(self, x):
        return x

class ANFIS:
    def __init__(self, num_rules, num_features):
        self.layer1 = FuzzyLayer(num_rules, num_features)
        self.layer2 = MultiplyLayer()
        self.layer3 = NormalizationLayer()
        self.layer4 = DefuzzificationLayer(num_rules, num_features)
        self.layer5 = OutputLayer()


    def train(self, input_data, target_output, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            layer1_output = self.layer1.forward(input_data)
            layer2_output = self.layer2.forward(layer1_output, input_data)
            layer3_output = self.layer3.forward(layer2_output)
            layer4_output = self.layer4.forward(layer3_output)
            output = self.layer5.forward(layer4_output)

            # Backpropagation (Not implemented in this example)

            # Print or store metrics for analysis (e.g., mean squared error)
            mse = np.mean((output - target_output) ** 2)
            print(f"Epoch {epoch + 1}/{epochs}, Mean Squared Error: {mse}")

# Example usage
num_rules = 5
num_features = 2
anfis_model = ANFIS(num_rules, num_features)

# Dummy data for training (replace with your actual data)
input_data = np.random.rand(100, num_features)
target_output = np.random.rand(100)

# Training the ANFIS model
anfis_model.train(input_data, target_output, epochs=100)
