# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:36:54 2024

@author: Andrei
"""

#%%

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%%

df = pd.read_excel("timeline SITS.xlsx")

#%%

# Assuming you have a timestamp column, replace 'timestamp_column' with the actual name
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df.set_index('Data', inplace=True)

#%%

# Assuming you have 80% of data for training
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

#%%

# Assuming you want to use RandomForestRegressor, you can choose a different model if needed
model = RandomForestRegressor()

# Training on Avg_MSI
model.fit(train.index.values.reshape(-1, 1), train['Avg_MSI'])

# Predicting on the test set
predictions_msi = model.predict(test.index.values.reshape(-1, 1))

# Training on Avg_NDVI
model.fit(train.index.values.reshape(-1, 1), train['Avg_NDVI'])

# Predicting on the test set
predictions_ndvi = model.predict(test.index.values.reshape(-1, 1))

#%%

# Calculate Mean Squared Error for Avg_MSI
mse_msi = mean_squared_error(test['Avg_MSI'], predictions_msi)
print(f'Mean Squared Error for Avg_MSI: {mse_msi}')

# Calculate Mean Squared Error for Avg_NDVI
mse_ndvi = mean_squared_error(test['Avg_NDVI'], predictions_ndvi)
print(f'Mean Squared Error for Avg_NDVI: {mse_ndvi}')


#%%

plt.figure(figsize=(12, 6))

# Plot the actual vs predicted for Avg_MSI
plt.subplot(1, 2, 1)
plt.plot(test.index, test['Avg_MSI'], label='Actual')
plt.plot(test.index, predictions_msi, label='Predicted')
plt.title('Avg_MSI Forecasting')
plt.legend()

# Plot the actual vs predicted for Avg_NDVI
plt.subplot(1, 2, 2)
plt.plot(test.index, test['Avg_NDVI'], label='Actual')
plt.plot(test.index, predictions_ndvi, label='Predicted')
plt.title('Avg_NDVI Forecasting')
plt.legend()

plt.show()

#%%

import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with timestamp_column, Avg_MSI, and Avg_NDVI columns
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
df.set_index('Data', inplace=True)

# Decompose the time series for Avg_MSI
stl_msi = STL(df['Avg_MSI'])
result_msi = stl_msi.fit()
trend_msi, seasonal_msi, residual_msi = result_msi.trend, result_msi.seasonal, result_msi.resid

# Decompose the time series for Avg_NDVI
stl_ndvi = STL(df['Avg_NDVI'])
result_ndvi = stl_ndvi.fit()
trend_ndvi, seasonal_ndvi, residual_ndvi = result_ndvi.trend, result_ndvi.seasonal, result_ndvi.resid

# Visualize the decomposition
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(df['Avg_MSI'], label='Original')
plt.title('Avg_MSI - Original')

plt.subplot(2, 3, 2)
plt.plot(trend_msi, label='Trend')
plt.title('Avg_MSI - Trend')

plt.subplot(2, 3, 3)
plt.plot(seasonal_msi, label='Seasonal')
plt.title('Avg_MSI - Seasonal')

plt.subplot(2, 3, 4)
plt.plot(df['Avg_NDVI'], label='Original')
plt.title('Avg_NDVI - Original')

plt.subplot(2, 3, 5)
plt.plot(trend_ndvi, label='Trend')
plt.title('Avg_NDVI - Trend')

plt.subplot(2, 3, 6)
plt.plot(seasonal_ndvi, label='Seasonal')
plt.title('Avg_NDVI - Seasonal')

plt.show()

#%%

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Train ARIMA model for Avg_MSI
model_msi = ARIMA(df['Avg_MSI'], order=(1, 1, 1))  # Adjust order as needed
results_msi = model_msi.fit()

# Train ARIMA model for Avg_NDVI
model_ndvi = ARIMA(df['Avg_NDVI'], order=(1, 1, 1))  # Adjust order as needed
results_ndvi = model_ndvi.fit()

# Forecast future values
forecast_steps = 12  # Adjust as needed
forecast_msi = results_msi.get_forecast(steps=forecast_steps)
forecast_ndvi = results_ndvi.get_forecast(steps=forecast_steps)

# Create a new time index for the forecast starting from the last date in the original data
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq=df.index.freq)[1:]

# Set the new time index for the forecasted values
forecast_msi = pd.Series(forecast_msi.predicted_mean.values, index=forecast_index)
forecast_ndvi = pd.Series(forecast_ndvi.predicted_mean.values, index=forecast_index)

# Get confidence intervals using the get_forecast object
conf_int_msi = forecast_msi.get_forecast(steps=forecast_steps).conf_int()
conf_int_ndvi = forecast_ndvi.get_forecast(steps=forecast_steps).conf_int()

# Visualize results
plt.figure(figsize=(12, 6))

# Plot Avg_MSI
plt.subplot(1, 2, 1)
plt.plot(df['Avg_MSI'], label='Actual')
plt.plot(forecast_msi, label='Forecasted')
plt.fill_between(conf_int_msi.index, conf_int_msi.iloc[:, 0], conf_int_msi.iloc[:, 1], color='gray', alpha=0.2, label='Confidence Interval')
plt.title('Avg_MSI Forecasting')
plt.legend()

# Plot Avg_NDVI
plt.subplot(1, 2, 2)
plt.plot(df['Avg_NDVI'], label='Actual')
plt.plot(forecast_ndvi, label='Forecasted')
plt.fill_between(conf_int_ndvi.index, conf_int_ndvi.iloc[:, 0], conf_int_ndvi.iloc[:, 1], color='gray', alpha=0.2, label='Confidence Interval')
plt.title('Avg_NDVI Forecasting')
plt.legend()

plt.show()


