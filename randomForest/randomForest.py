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

def grab_price_data():

    # Define tickers. Supports more than 1 ticker
    tickers_list = ['^GSPC']
    
    full_price_history = []

    for ticker in tickers_list:

        # Grab the daily price history for 1 year
        stock_data = yf.Ticker(ticker)
        price_history = stock_data.history(period='1y', interval='1d')

        # Grab just the candles, and add them to the list.
        for index, row in price_history.iterrows():
            candle = {
                'close': row['Close'],
                'datetime': int(index.timestamp() * 1000),  # Convert timestamp to milliseconds
                'high': row['High'],
                'low': row['Low'],
                'open': row['Open'],
                'symbol': ticker,
                'volume': row['Volume']
            }
            full_price_history.append(candle)

    # Save the data to a CSV file, don't have an index column
    price_data = pd.DataFrame(full_price_history)
    price_data.to_csv('price_data.csv', index=False, columns=['close', 'datetime', 'high', 'low', 'open', 'symbol', 'volume'])

# Call the function
grab_price_data()
