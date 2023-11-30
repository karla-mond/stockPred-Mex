# Import libraries
import os
import sys
import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, classification_report

from pathlib import Path

# Grabbing Historical Price Data

def grab_price_data():
    
    # Supports more than 1 ticker.
    # S&P500
    tickerStrings = ['AAPL', 'MSFT']
    
    for ticker in tickerStrings:
        
        # Last 2 days, with daily frequency
        # Candles in yf.dowload - Date,Open,High,Low,Close,Adj Close,Volume,ticker
        
        data = yf.download(ticker, group_by="Ticker", period='2y', interval='1d')
        data['ticker'] = ticker  # add this column because the dataframe doesn't contain a column with the ticker
        data.to_csv(f'randomForest/csvDataFrames/ticker_{ticker}.csv')
        
    # Read in multiple files saved with the previous section and create a single dataframe
    p = Path('/Users/mariafernandadeleon/src/stockPred/stockPred-Mex/randomForest/csvDataFrames')

    # Find the files; this is a generator, not a list
    files = (p.glob('ticker_*.csv'))
    
    # read the files into a dataframe
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    df.to_csv('randomForest/csvDataFrames/price_data.csv', index=False)

    
if os.path.exists('/randomForest/csvDataFrames/price_data.csv'):

    # Load the data
    price_data = pd.read_csv('/randomForest/csvDataFrames/price_data.csv')

else:

    # Grab the data and store it.
    grab_price_data()

    # Load the data
    price_data = pd.read_csv('/randomForest/csvDataFrames/price_data.csv')

# Display the head before moving on.
print(price_data.head())
    
# Data Processing


