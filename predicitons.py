import bs4 as bs
import datetime as dt
from matplotlib import pyplot
import os
from pandas import read_csv
from pandas.plotting import lag_plot
import pandas_datareader.data as web
import pickle
import requests
from statsmodels.graphics.tsaplots import plot_acf
import tkinter as tk
from tkinter import *
from tqdm import tqdm

# ----------------------------
# creating a GUI

# creating a Tk() object
win = tk.Tk()
# creating a title for the window
win.title("Stock Market Predictor Input")
# creating the geometry of the main root window
win.geometry("600x600")


# making the string global so it can be printed and accessed in the code
# plotting the data for the requested stock market company
def passing_user_entry():
    global e1
    global user_input
    user_input = e1.get()
    e1.insert(INSERT, user_input)

    win.destroy()


# creating the first GUI window
label = tk.Label(win, text="Stock Market Predictor")
label.pack(pady=3)

# creating the entry label for the suer to enter in the name of the desired stock company
label2 = tk.Label(win, text="Please enter the name of the stock")
label2.pack(pady=10, padx=10)
e1 = tk.Entry(win)
e1.pack(pady=10, padx=10)

# button to close the window and execute the user entry
btn2 = tk.Button(win, text="Close Application", command=passing_user_entry)
btn2.pack(pady=12, padx=12)

mainloop()

print(user_input)


# -------------------
# saving ticker names
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.find_all('td')[0].text.strip()
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
        print(tickers)

    return tickers


save_sp500_tickers()


# ------------------------------------------------
# getting the stock information from yahoo finance
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    # use the past 5 years worth of data
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    for ticker in tqdm(tickers):
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker.replace('.', '-'), 'yahoo', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker), header=True)
            except Exception as ex:
                print('Error:', ex)
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo(True)

# ----------------------------------------------------------
# printing the graph of the stock: Adj Close Values over Time
series = read_csv(f'stock_dfs/{user_input}.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0)
print(series.head())
series.plot()
pyplot.title('Stock Adj Closing Values over Time')
pyplot.xlabel('Time')
pyplot.ylabel('Stock Price (USD)')
pyplot.show()

# ---------------------------------------------
# fitting and evaluating an autoregression model
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy
from math import sqrt

# checking for autocorrelation
lag_plot(series)
pyplot.title('Plotting t vs. t+1 to determine correlation significance')
pyplot.xlabel('Current stock price at time t (USD)')
pyplot.ylabel('Predicted stock price as time t+1 (USD)')
pyplot.show()

# checking for lags
plot_acf(series, lags=29)
pyplot.title('Checking for lag significance with lag window = 29')
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


series = read_csv(f'stock_dfs/{user_input}.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0, parse_dates=True,
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
pyplot.plot(test, label='real stock values')
pyplot.plot(predictions, label='predicted stock values', color='red')
pyplot.title('Real Stock Values vs. Predicted Stock Values')
pyplot.xlabel('Time')
pyplot.ylabel('Stock Price (USD)')
pyplot.legend()
pyplot.show()

# ----------------------------
# building a persistence model
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error

series = read_csv(f'stock_dfs/{user_input}.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0)

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
pyplot.plot(test_y, label='expected stock values')
pyplot.plot(predictions, label='predicted stock values', color='red')
pyplot.title('Predicted Stock Values vs. Expected Stock Values')
pyplot.xlabel('Time')
pyplot.ylabel('Stock Price (USD)')
pyplot.legend()
pyplot.show()


# -------------------------------------------------
# create and evaluate a static autoregressive model
# load dataset
series = read_csv(f'stock_dfs/{user_input}.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0,
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
pyplot.plot(test, label='test values')
pyplot.plot(predictions, label='predicted values', color='red')
pyplot.title('Real Stock Values vs. Predicted Values (Static AR Model)')
pyplot.xlabel('Time')
pyplot.ylabel('Stock Price (USD)')
pyplot.legend()
pyplot.show()

# ------------------
# create and evaluate an updated autoregressive model
# load dataset
series = read_csv(f'stock_dfs/{user_input}.csv', usecols=['Date', 'Adj Close'], header=0, index_col=0,
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
pyplot.title('Real Stock Values vs. Predicted Values (Updated AR Model)')
pyplot.xlabel('Time')
pyplot.ylabel('Stock Price (USD)')
pyplot.legend()
pyplot.show()