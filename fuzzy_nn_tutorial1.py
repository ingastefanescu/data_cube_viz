# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:56:09 2024

@author: Andrei
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model

# Generate synthetic data
np.random.seed(0)
X = np.random.uniform(-1, 1, (100, 2))  # 100 samples, 2 features
y = X[:, 0]**2 + X[:, 1]**2 + np.random.normal(0, 0.1, 100)  # Quadratic function with noise

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Custom Fuzzy Layer
class FuzzyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, x):
        return tf.math.sigmoid(tf.matmul(x, self.kernel))

# Neural Network Model
input_layer = Input(shape=(2,))
fuzzy_layer = FuzzyLayer(10)(input_layer)
hidden_layer = Dense(10, activation='relu')(fuzzy_layer)
output_layer = Dense(1)(hidden_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Evaluate the model
predictions = model.predict(X_test)
mse = np.mean((y_test - predictions.flatten())**2)
print("Mean Squared Error:", mse)

# Plotting results
plt.scatter(X_test[:,0], y_test, label='True')
plt.scatter(X_test[:,0], predictions, label='Predicted')
plt.legend()
plt.title("Fuzzy Neural Network Predictions")
plt.show()
