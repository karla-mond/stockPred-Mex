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


def get_price_data():
    # Grabbing Historical Price Data

    # Supports more than 1 ticker.
    # S&P500 - ^GSPC
    tickerStrings = ['AAPL', 'MSFT']
    
    for ticker in tickerStrings:
        # Last 2 days, with daily frequency
        # Candles in yf.dowload - Date,Open,High,Low,Close,Adj Close,Volume,ticker
        
        df = yf.download(ticker, group_by="Ticker", period='2y', interval='1d')
        df['Symbol'] = ticker  # add this column because the dataframe doesn't contain a column with the ticker
        df.to_csv(f'randomForest/csvDataFrames/ticker_{ticker}.csv')
        
    return clean_data()

def clean_data():
    # This isolation prevents NaN at the time of calculating Difference
    
    # Read in multiple files saved with the previous section    
    p = Path('randomForest/csvDataFrames')

    # Find the files; this is a generator, not a list
    files = (p.glob('ticker_*.csv'))
    
    for file in files:
        # Read the file
        df = pd.read_csv(file)
        
        # Remove unwanted columns and re-organize data
        df = df[['Symbol','Date','Close','High','Low', 'Open', 'Volume']]

        # It should be already be sorted by symbol and Date
        # Sort by Symbol - name = df.sort_values(by = ['Symbol','Date'], inplace = True)

        # Calculate the change in price
        df['Difference'] = df['Close'].diff()
        
        df.to_csv(file, index=False)
        
    # read the files to create a single dataframe
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    
    return df

def calculate_momentum(df):
    # Calculate momentum since we want to predict if the stock goes up and down, not the price itself
    # Relative Strength Index
    # RSI > 70 - overbought
    # RSI < 30 - oversold
    
    # Calculate the 14 day RSI
    rsi_period = 14
    
    # Separate data frames into average change in price up and down
    up_df = df.dropna()[['Symbol', 'Difference']][df['Difference'] > 0]
    down_df = df.dropna()[['Symbol', 'Difference']][df['Difference'] < 0]
    
    # Absolute values for down average change in price
    down_df['Difference'] = down_df['Difference'].abs()
    
    # Calculate the EWMA (Exponential Weighted Moving Average), older values are given less weight compared to newer values
    # Relative strenth formula
    ewma_up = up_df.groupby('Symbol').group().transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df.groupby('Symbol')['Difference'].transform(lambda x: x.ewm(span = n).mean())

    # Calculate the Relative Strength
    relative_strength = ewma_up / ewma_down

    # Calculate the Relative Strength Index
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    # Add the info to the data frame.
    df['down_days'] = down_df['Difference']
    df['up_days'] = up_df['Difference']
    price_data['RSI'] = relative_strength_index

    # Display the head.
    price_data.head(30)
        
    print(up_df)
    print(down_df)

    return 0

def main():
    data_folder = 'randomForest/csvDataFrames/price_data.csv'
    
    # Prevent re-pulling data
    if os.path.exists(data_folder):
        # Load the data
        df = pd.read_csv(data_folder)
    else:
        # Grab the data and store it.
        df = get_price_data()
        df.to_csv(data_folder, index=False)
        
        # Load the data
        df = pd.read_csv(data_folder)
        
    # Display the head
    print(df.head())
    
    # Display NaN
    print(df[df.isna().any(axis = 1)])
    
    # Calculate momentum
    calculate_momentum(df)
        
if __name__ == "__main__":
    main()

