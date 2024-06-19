# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:03:55 2024

@author: Andrei
"""

#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

#%% Read data frame, select msi, create train and test arrays
df = pd.read_excel('timeline SITS.xlsx')
msi = df.drop('Avg_NDVI', axis = 1)

train_size = int(len(msi) * 0.8)  # Use 80% for training
train, test = msi.iloc[:train_size], msi.iloc[train_size:]

#%% Linear Regressor training
X_train, y_train = train.index.values.reshape(-1, 1), train['Avg_MSI']
X_test, y_test = test.index.values.reshape(-1, 1), test['Avg_MSI']

regressor = LinearRegression()
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

#%% Plot the predicted and real values for the Linear Regressor
plt.plot(test.index, test['Avg_MSI'], label='Actual Values')
plt.plot(test.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

#%% Suppor Vectorm Machine Regressor

regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot only the test values and predictions
plt.plot(test.index, test['Avg_MSI'], label='Actual Values')
plt.plot(test.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

#%% ARIMA

msi['Data'] = pd.to_datetime(msi['Data'], format='%d.%m.%Y')
msi_arima = msi.set_index("Data")
ts = msi_arima["Avg_MSI"]

# Step 2: Split into Train and Test Sets
train_size = int(len(ts) * 0.8)  # Use 80% for training
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

# Fit ARIMA model on the training set
order = (1, 1, 1)  # Example order, you may need to tune this
model = ARIMA(train, order=order)
results = model.fit()# Make predictions

#forecast_steps = len(test)  # Number of steps to forecast into the future
forecast = results.get_forecast(steps=len(test), index=test.index)

# Plotting
plt.plot(test.index, test, label='Actual Values')
plt.plot(forecast.predicted_mean, label='ARIMA Forecast', color='red')
# plt.fill_between(forecast, forecast.conf_int()['lower Value'], forecast.conf_int()['upper Value'], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

#%% Auto ARIMA

test['Avg_MSI'] = test['Avg_MSI'].astype(float)

# Fit automated arima model
model = auto_arima(train, seasonal=True, suppress_warnings=True)
model.fit(train)
forecast_steps = len(test)
forecast, conf_int = model.predict(n_periods=forecast_steps, return_conf_int=True)

forecast_dates = pd.date_range(start=df.index[-1] + pd.to_timedelta(1, unit='D'), periods=forecast_steps, freq='D')

# Plotting
plt.plot(df.index, df['Avg_MSI'], label='Actual Values')
plt.plot(forecast, forecast, label='ARIMA Forecast', linestyle='dashed', color='red')
plt.fill_between(forecast, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
