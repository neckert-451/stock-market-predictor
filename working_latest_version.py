import bs4 as bs
from collections import deque
import datetime as dt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import random
import requests
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import time
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tqdm import tqdm

style.use('ggplot')


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

# ----------------------------
# creating a GUI

# creating a Tk() object
win = tk.Tk()
# creating a title for the window
win.title("Stock Market Predictor Input")
# creating the geometry of the main root window
win.geometry("600x600")


# opening a new window with the list of stock names
def open_new_window():

    new_window = tk.Toplevel(win)
    new_window.title("List of Stocks")
    new_window.geometry("600x600")

    label3 = tk.Label(new_window, text="Search by stock name")
    label3.pack(pady=10)


# making the string global so it can be printed and accessed in the code
# plotting the data for the requested stock market company
def passing_user_entry():
    global e1
    global user_input
    user_input = e1.get()
    e1.insert(INSERT, user_input)

    # new_window2 = tk.Toplevel(win)
    # new_window2.title(f'Visualizing {user_input} Data')
    # new_window2.geometry("600x600")

    # --------------------------------------
    # creating a graph and plotting the data
    headers = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
    df = pd.read_csv(f'stock_dfs\{user_input}.csv', names=headers)

    # end = datetime.now()
    # start = end - timedelta(days=10)

    print(df)

    x = df['Date']
    y = df['Adj Close']

    # plotting the historical data for desired stock company
    plt.plot(x, y, color='blue', linestyle='solid')
    plt.title(f'{user_input} Closing Stock Prices')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Time')
    plt.ylabel('Stock Prices')
    plt.show()


# terminating the window when the user wants to exit the application
def terminate_window():
    win.destroy()


# creating the first GUI window
label = tk.Label(win, text="Stock Market Predictor")
label.pack(pady=3)

# creating the entry label for the suer to enter in the name of the desired stock company
label2 = tk.Label(win, text="Please enter the name of the stock")
label2.pack(pady=10, padx=10)
e1 = tk.Entry(win)
e1.pack(pady=10, padx=10)

# button to execute the user entry window
btn1 = tk.Button(win, text="Next", command=passing_user_entry)
btn1.pack(pady=10, padx=10)

btn2 = tk.Button(win, text="Close Application", command=terminate_window)
btn2.pack(pady=12, padx=12)

label4 = tk.Label(win, text="==================================================================")
label4.pack(pady=10)

# creating a way for the user to search for a stock company by the stock name (in a list)
#label3 = tk.Label(win, text="If you don't know the name of the stock, please click below to search for it")
#label3.pack(pady=10, padx=10)
# creating a button which, when clicked, opens a new window (list of stock names)
#btn = Button(win, text="Click here to search by stock name", command=open_new_window)
#btn.pack(pady=10, padx=10)

mainloop()

print(user_input)

# ----------------------------------------------
# creating the machine learning part of the code
SEQ_LEN = 60  # how long of a preceding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = user_input


# ---------------------------------------------------------------------
# classifying the current and future values for training and validation
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


# ---------------------
# pre-processing the df
def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)


    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    print(df.values)

    # iterate over the values
    for i in df.values:
        # store all but the target values
        prev_days.append([n for n in i[:-1]])
        # make sure we have 60 sequences
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    # shuffle all
    random.shuffle(sequential_data)

    # list that will store buy sequences and targets
    buys = []
    # list that will store sell sequences and targets
    sells = []

    # iterate over the sequential data
    for seq, target in sequential_data:
        # if it's not a "not buy"
        if target == 0:
            # appends to the sells list
            sells.append([seq, target])
        # otherwise if the target is a 1 then its a buy
        elif target == 1:
            buys.append([seq, target])

    # shuffle the buys and sells
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    # making sure both lists are only up to the shortest length
    buys = buys[:lower]
    sells = sells[:lower]

    # add the lists together
    sequential_data = buys + sells
    # shuffle the data so the model doesn't get confused
    random.shuffle(sequential_data)

    X = []
    y = []

    # going over the new sequential data
    for seq, target in sequential_data:
        # X is the sequences
        X.append(seq)
        # y is the targets/labels
        y.append(target)

    # return X and y and make X as a numpy array
    return np.array(X), y


# ------------------------
# setting up the data frame
main_df = pd.DataFrame()
# the ratios are the stocks that are used for training and validation
# this list needs to be all the csv files or a random bunch of them
ratios = ["A", "AAL", "AAP", "AAPL", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE"]

for ratio in ratios:
    #print(ratio)
    dataset = f'stock_dfs/{ratio}.csv'
    df = pd.read_csv(dataset, names=['date', 'high', 'low', 'open', 'close', 'volume', 'adj close'])

    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("date", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# ----------------------------------------------------------
# if there are gaps in the data, use previously known values
main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

times = sorted(main_df.index.values)
last_15pct = sorted(main_df.index.values)[-int(0.15 * len(times))]

validation_main_df = main_df[(main_df.index >= last_15pct)]
main_df = main_df[(main_df.index < last_15pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)


print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Don't buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


# --------------------------------
# number of passes through the data
# use 10 for CPU and 100 for GPU
EPOCHS = 10
# number of batches
# use smaller batches if getting OOM errors
BATCH_SIZE = 64
# unique name for the model
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
# normalizing the activation outputs
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# ------------------
# compiling the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

train_y = np.array(train_y)
validation_y = np.array(validation_y)

# training the model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y)
)

# scoring the  model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# printing a summary of the model
model.summary()

# --------------
# finding the prediction values based on training and validation
trainPredict = model.predict(train_x)
testPredict = model.predict(validation_x)

# --------
# plotting
# plotting the training and testing predictions
plt.plot(trainPredict, color='Red', label='Training Predictions')
plt.plot(testPredict, color='Blue', label='Testing Predictions')
plt.title(f'LSTM Results for {user_input}: Predictions vs Test Values')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

# --------------------------------------------------
# create and evaluate an updated autoregressive model
from math import sqrt
import pandas as pd
from pandas import read_csv
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


# load dataset
series = read_csv(f'stock_dfs\{user_input}.csv', usecols=[0, 6], header=0, index_col=0,
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
    lag = [history[i] for i in range(length-window, length)]
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
plt.plot(test, color='blue', label='test values')
plt.plot(predictions, color='red', label='predicted values')
plt.title(f'Autoregression Results for {user_input}: Predictions vs Test Values')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()