
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# https://medium.com/datainc/time-series-analysis-and-forecasting-with-arima-in-python-aa22694b3aaa
# https://medium.com/nlplanet/bert-finetuning-with-hugging-face-and-training-visualizations-with-tensorboard-46368a57fc97

with open("/home/bitz/voice_recognition/training_log.json", 'r') as fp:
    dat = json.loads(fp.read())

data = pd.DataFrame(dat['training'])
data['initime'] = pd.to_datetime(data['init'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
data['exitime'] = pd.to_datetime(data['exit'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
data = data.drop(columns=['learning_rate', 'init', 'exit'])
data = data.rename(columns={"initime": "init"})
data = data.rename(columns={"exitime": "exit"})
data.set_index('init', inplace=True)
# print(data.head())

plt.plot(data['grad_norm'], marker='o', markerfacecolor='yellow', markeredgecolor='red', markersize=8)
plt.title('Gradient Normalization')
plt.xlabel('Time')
plt.ylabel('GradNorm Predictions')
plt.xlim(data.index.min(), data.index.max())
# Defining and displaying all time axis ticks
plt.figure(figsize=(20,8))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator())

plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gcf().autofmt_xdate()

plt.xticks(rotation=90)
plt.show()

"""
Testing for Stationarity
Next, we use the Augmented Dickey-Fuller (ADF) test to check for stationarity:
"""
adf_test = adfuller(data['grad_norm'])
# Output the results
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])
"""
A p-value below 0.05 indicates stationarity, and our data meets this criterion, 
so we do not need to difference it
"""

"""
Finding ARIMA Parameters
We use the Autocorrelation Function (ACF) and Partial Autocorrelation Function 
(PACF) plots to find the ARIMA parameters.
"""
plot_acf(data['grad_norm'], lags=40)
plot_pacf(data['grad_norm'], lags=40, method='ywm')
plt.show()

"""
Building the ARIMA Model
"""
# model = ARIMA(data['grad_norm'], order=(1, 0, 1), freq=None)
# model_fit = model.fit()

"""
Training and Forecasting
We train the model on the data and perform a forecast.
"""

# forecast = model_fit.get_forecast(steps=30)

"""
Model Evaluation
To assess the model, we perform a retrospective forecast.
"""

print("Split the data into train and test")
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]

print("Fit the ARIMA model on the training dataset")
model_train = ARIMA(train['grad_norm'], order=(5, 0, 1))
model_train_fit = model_train.fit()

"""
Running the example prints a summary of the fit model. This summarizes the 
coefficient values used as well as the skill of the fit on the on the in-
sample observations.
"""
# print(model_train_fit.summary())
print("line plot of residuals")
residuals = pd.DataFrame(model_train_fit.resid)
residuals.plot()
plt.show()

print("density plot of residuals")
residuals.plot(kind='kde')
plt.show()
print("summary stats of residuals")
print(residuals.describe())

print("Forecast on the test dataset")
test_forecast = model_train_fit.get_forecast(steps=len(test))
test_forecast_series = pd.Series(test_forecast.predicted_mean, index=test.index)

print("Calculate the mean squared error")
# mse = mean_squared_error(test['grad_norm'], test_forecast_series)
# rmse = mse**0.5

print("Create a plot to compare the forecast with the actual test data")
plt.figure(figsize=(14,7))
plt.plot(train['grad_norm'], label='Training Data')
plt.plot(test['grad_norm'], label='Actual Data', color='orange')
plt.plot(test_forecast_series, label='Forecasted Data', color='green')
plt.fill_between(test.index, 
                 test_forecast.conf_int().iloc[:, 0], 
                 test_forecast.conf_int().iloc[:, 1], 
                 color='k', alpha=.15)
plt.title('ARIMA Model Evaluation')
plt.xlabel('Time')
plt.ylabel('GradNorm Predictions')
plt.legend()
plt.show()

# print('RMSE:', rmse)
