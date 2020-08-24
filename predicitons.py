from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf

# ----------------------------------------------------------
# printing the graph of the stock: Adj Close Values over Time
series = read_csv('stock_dfs\GOOGL.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0)
print(series.head())
series.plot()
pyplot.show()

# ---------------------------------------------
# fitting and evaluating an autoregression model
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy
from math import sqrt

# checking for autocorrelation
lag_plot(series)
pyplot.show()

# checking for lags
plot_acf(series, lags=29)
pyplot.show()

# --------------------------------------------
# create a difference transform of the dataset
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return numpy.array(diff)


# make a prediction give regression coefficients and lag obs
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


series = read_csv('stock_dfs\GOOGL.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0, parse_dates=True,
                  squeeze=True)
# split dataset
X = difference(series.values)
size = int(len(X)*0.15)
train, test = X[0:size], X[size:]

# train autoregression
window = 3
model = AutoReg(train, lags=50)
model_fit = model.fit()
coef = model_fit.params

# walk forward over time series in test
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
    yhat = predict(coef, history)
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plotting
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# ----------------------------
# building a persistence model
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error

series = read_csv('stock_dfs\GOOGL.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0)

# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']

# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]


# persistence model
def model_persistence(x):
    return x


# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions vs. expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()


# -------------------------------------------------
# create and evaluate a static autoregressive model
# load dataset
series = read_csv('stock_dfs\GOOGL.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0,
                  parse_dates=True, squeeze=True)

# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]

# train autoregression
model = AutoReg(train, lags=30)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)


predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# ------------------
# create and evaluate an updated autoregressive model
# load dataset
series = read_csv('stock_dfs\GOOGL.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0,
                  parse_dates=True, squeeze=True)

# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]

# train autoregression
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot
pyplot.plot(test, label='test values')
pyplot.plot(predictions, color='red', label='predicted values')
pyplot.title('Predictions vs Test Values')
pyplot.xlabel('Time')
pyplot.ylabel('Stock Price (USD)')
pyplot.legend()
pyplot.show()
