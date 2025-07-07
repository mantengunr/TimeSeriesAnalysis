# Import necessary libraries
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate
import matplotlib.pyplot as plt

# Suppressing warnings
warnings.filterwarnings("ignore")

# Initializing hex_forecasts as an empty list
hex_forecasts = []

# Step 1: Loading the data
df = pd.read_csv('sr_hex.csv')

# Removing requests that don't have a valid geolocation
df = df[df['h3_level8_index'] != "0"]

# Converting 'creation_timestamp' to datetime
df['creation_timestamp'] = pd.to_datetime(df['creation_timestamp'])

# Setting 'creation_timestamp' as the index
df.set_index('creation_timestamp', inplace=True)

# Ensuring 'h3_level8_index' is treated as a string
df['h3_level8_index'] = df['h3_level8_index'].astype(str)

# Aggregating the data to get weekly counts of service requests
weekly_data = df.groupby('h3_level8_index').resample('W').size().reset_index(name='notification_number')

# Extracting week and year from the timestamp for grouping
weekly_data['week'] = weekly_data['creation_timestamp'].dt.isocalendar().week
weekly_data['year'] = weekly_data['creation_timestamp'].dt.year

# Creating Lagged Features
weekly_data['lag_1'] = weekly_data['notification_number'].shift(1)
weekly_data['lag_2'] = weekly_data['notification_number'].shift(2)
weekly_data['lag_3'] = weekly_data['notification_number'].shift(3)

# Dropping rows with NaN values created by shifting
weekly_data.dropna(inplace=True)

# Making Predictions for the next 4 weeks for each hex
unique_hexes = weekly_data['h3_level8_index'].unique()

for hex_id in unique_hexes:
    hex_data = weekly_data[weekly_data['h3_level8_index'] == hex_id]
    
    if len(hex_data) < 3:
        print(f"Not enough data to forecast for hex {hex_id}")
        continue
    
    # Performing the ADF test on the notification counts for stationarity
    if len(hex_data) > 12:  # Ensure there are enough data points
        result = adfuller(hex_data['notification_number'])
        print(f'ADF Statistic for hex {hex_id}:', result[0])
        print(f'p-value for hex {hex_id}:', result[1])

        # If not stationary, difference the data
        if result[1] > 0.05:
            print(f"The series for hex {hex_id} is not stationary. Differencing the data.")
            hex_data['notification_number'] = hex_data['notification_number'].diff().dropna()
    else:
        print(f"Not enough data points to perform ADF test for hex {hex_id}")
        continue
    
    # Preparing the data for modeling
    y = hex_data['notification_number']
    
    # Fitting the ARIMA model
    p, d, q = 1, 1, 1
    model = ARIMA(y, order=(p, d, q))
    model_fit = model.fit()
    
    # Making predictions
    forecast = model_fit.forecast(steps=4)
    hex_forecasts.append([hex_id] + list(forecast))

# Converting the forecasted values to a DataFrame
forecast_df = pd.DataFrame(hex_forecasts, columns=["Hex ID", "Week 1", "Week 2", "Week 3", "Week 4"])

# Saving the DataFrame to a CSV file
forecast_df.to_csv('forecasted_values.csv', index=False)

# Printing the forecasted values in a structured table
headers = ["Hex ID", "Week 1", "Week 2", "Week 3", "Week 4"]
print(tabulate(hex_forecasts, headers=headers, tablefmt="grid"))

# Visualizing the forecasted values
for hex_id in unique_hexes:
    hex_forecast = forecast_df[forecast_df['Hex ID'] == hex_id]
    if not hex_forecast.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(['Week 1', 'Week 2', 'Week 3', 'Week 4'], hex_forecast.iloc[0, 1:], marker='o')
        plt.title(f'Forecast for Hex ID: {hex_id}')
        plt.xlabel('Week')
        plt.ylabel('Forecasted Notification Number')
        plt.grid(True)
        plt.show()