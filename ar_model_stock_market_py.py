import bs4 as bs
import csv
import datetime as dt
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from numpy import mean
import os
import pandas_datareader.data as web
from pandas import read_csv
from pandas.plotting import lag_plot, autocorrelation_plot
import pickle
import requests
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import *
from tqdm import tqdm

style.use('ggplot')


# ------------------------------------
# saving the names of S&P 500 companies
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

    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    for ticker in tqdm(tickers):
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker.replace('.', '-'), 'yahoo', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker), header=False)
            except Exception as ex:
                print('Error:', ex)
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo(True)


# ------------------------------------------------------------------------
# creating a GUI for the user to input a desired stock market company name

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


# creating a GUI window to accept user input
label = tk.Label(win, text="Stock Market Predictor")
label.pack(pady=3)

# creating the entry label for the user to enter in the name of the desired stock company
label2 = tk.Label(win, text="Please enter the name of the stock")
label2.pack(pady=10, padx=10)
e1 = tk.Entry(win)
e1.pack(pady=10, padx=10)

# button to close the window and execute the user entry
btn2 = tk.Button(win, text="Close Application", command=passing_user_entry)
btn2.pack(pady=12, padx=12)

mainloop()

# --------------------------------------------------
# create and evaluate an updated autoregressive model

# load dataset
series = read_csv(f'stock_dfs\{user_input}.csv', usecols=[0, 6], header=0, index_col=0,
                  parse_dates=True, squeeze=True)

# checking and evaluating correlation and autocorrelation in the dataset
# checking for autocorrelation with a scatter plot
lag_plot(series)
plt.title('Plotting t vs. t+1 to determine correlation significance')
plt.xlabel('Current stock price at time t (USD)')
plt.ylabel('Predicted stock price as time t+1 (USD)')
plt.show()

# plotting the correlation significance with respect to the number of lags
# this plot is used to determine what number of lags is important
autocorrelation_plot(series)
plt.title('Correlation significance with respect to a number of lags')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# this plots the autocorrelation function (ACF)
# ACF = direct and indirect relationship between an observation and another observation at a prior time step
# confidence intervals are drawn as a cone (95% confidence interval)
# correlation values outside the cone are probably a correlation - not a statistical fluke
plot_acf(series, lags=10)
plt.title('Checking for autocorrelation with lags')
plt.show()

# this plots the partial autocorrelation function (PACF)
# PACF = direct relationship between an observation and its lag
# confidence intervals are drawn as a cone (95% confidence interval)
# correlation values outside the cone are probably a correlation - not a statistical fluke
plot_pacf(series, lags=50)
plt.title('Checking for partial autocorrelation with lags')
plt.show()

# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]

# train the autoregression
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()

# write to CSV
with open('output_file.csv', 'a', newline='') as csv_file:
    # header field names
    fields = ['Stock Name', 'Closing Price', 'Predicted Price', 'Prediction Error']
    # creating a csv writer object
    csv_writer = csv.writer(csv_file)
    # writing the fields
    csv_writer.writerow(fields)

    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window, length)]
        yhat = coef[0]

        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        print('Predicted Values = %f, Expected Values = %f' % (yhat, obs))
        difference = yhat - obs
        # defining the rows
        rows = [f'{user_input}', obs, yhat, difference]
        # writing the data rows
        csv_writer.writerow(rows)


# printing the root mean square error (rmse) for the mode
# a very small rmse = very little error
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)

# printing the prediction errors for the predicted values based on the expected values
prediction_errors = [test[i]-predictions[i] for i in range(len(test))]
print('Prediction Errors: %s' % prediction_errors)

# printing the mean prediction error
mean_forecast_error = mean(prediction_errors)
print('Mean Prediction Error: %f' % mean_forecast_error)

# printing the bias of the model
# prediction bias = 0 or ~ 0 is an unbiased model
# prediction bias is negative means an over prediction
bias = sum(prediction_errors) * 1.0/len(test)
print('Model Bias: %f' % bias)

# plot
plt.plot(test, color='blue', label='test values')
plt.plot(predictions, color='red', label='predicted values')
plt.title(f'Autoregression Results for {user_input}: Predictions vs Test Values')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()