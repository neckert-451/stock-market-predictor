import bs4 as bs
import csv
import datetime as dt
from math import sqrt
from matplotlib import pyplot
import os
import pandas as pd
from pandas import read_csv
import pandas_datareader.data as web
import pickle
import requests
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


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
                df.to_csv('stock_dfs/{}.csv'.format(ticker), header=True)
            except Exception as ex:
                print('Error:', ex)
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo(True)


# getting the predictions for stock prices
def get_predictions():
    # ------------------
    # create and evaluate an updated autoregressive model
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    for ticker in tickers:
        try:
            # load dataset
            series = read_csv(f'stock_dfs\{ticker}.csv', usecols=[0, 6], header=0, index_col=0, parse_dates=True, squeeze=True)
        except Exception:
            pass

        # split dataset
        X = series.values
        train, test = X[1:len(X)-7], X[len(X)-7:]

        try:
            # train autoregression
            window = 29
            model = AutoReg(train, lags=29)
            model_fit = model.fit()
            coef = model_fit.params

            # walk forward over time steps in test
            history = train[len(train)-window:]
            history = [history[i] for i in range(len(history))]
            predictions = list()
        except Exception:
            pass

        # creating a directory for the stock predictions
        if not os.path.exists('stock_predictions'):
            os.makedirs('stock_predictions')
        # creating a header for each stock prediction csv
        header = [f'{ticker}_Close', f'{ticker}_Predicted', f'{ticker}_Prediction_Error']

        with open(f'stock_predictions/{ticker}.csv', mode='a', newline='\n') as csv_file:
            # writing the header
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)

            for t in range(len(test)):
                length = len(history)
                lag = [history[i] for i in range(length-window,length)]
                yhat = coef[0]

                for d in range(window):
                    yhat += coef[d+1] * lag[window-d-1]
                obs = test[t]
                predictions.append(yhat)
                # print(predictions)
                history.append(obs)

                print('predicted=%f, expected=%f' % (yhat, obs))

                difference = yhat - obs

                # writing the date index into the csv files
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([obs, yhat, difference])

        #rmse = sqrt(mean_squared_error(test, predictions))
        #print('Test RMSE: %.3f' % rmse)

        # plot
        # pyplot.plot(test, label='test values')
        # pyplot.plot(predictions, color='red', label='predicted values')
        # pyplot.title(f'Real Stock Values vs. Predicted Values (Updated AR Model) for {ticker}')
        # pyplot.xlabel('Time')
        # pyplot.ylabel('Stock Price (USD)')
        # pyplot.legend()
        # pyplot.show()


get_predictions()


# compiling all the predicted and closing stock values
def compile_data():
    # finding all csv files in the stock predictions directory
    for file in os.listdir('stock_predictions'):
        if file.endswith('.csv'):
            print(os.path.join('stock_predictions', file))
            combined_csv = pd.concat([pd.read_csv(f'stock_predictions/{file}')
                                      for file in os.listdir('stock_predictions')], join='outer', axis=1)
            combined_csv.to_csv('combined_prediction_data.csv')


compile_data()

def adding_index():
    # creating the date index
    current_date = dt.datetime.today()
    current_date_formatted = dt.datetime.today().strftime('%m-%d-%Y')
    date_index = []
    i = 1

    main_df = pd.DataFrame()

    for i in range(7):
        i = i + 1
        previous_dates = dt.datetime.today() - dt.timedelta(days=i)
        date = previous_dates.strftime('%m-%d-%Y')

        print('date=%s' % date)
        date_index.append(date)

    print(date_index)

    df = pd.read_csv('combined_prediction_data.csv')
    print(df)

    main_df.insert(loc=0, column='Date', value=date_index)
    main_df = main_df.join(df, how='outer')
    del main_df['Unnamed: 0']
    main_df.to_csv('final_table.csv')

    print(main_df)


adding_index()

