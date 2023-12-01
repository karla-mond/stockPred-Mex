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

'''
# read the files to create a single dataframe
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    
'''
def get_price_data():
    # Grabbing Historical Price Data

    # Supports more than 1 ticker.
    # S&P500 - ^GSPC
    tickerStrings = ['AAPL', 'MSFT']
    
    for ticker in tickerStrings:
        # Last 2 days, with daily frequency
        # Candles in yf.dowload - Date,Open,High,Low,Close,Adj Close,Volume
        
        df = yf.download(ticker, group_by="Ticker", period='2y', interval='1d')
        
        # add this column because the dataframe doesn't contain a column with the ticker
        df['Symbol'] = ticker  
        df.to_csv(f'randomForest/csvDataFrames/ticker_{ticker}.csv')
        
    return
    
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
        
        df.to_csv(file, index=False)
        
        return

def add_data():
    # Read in multiple files saved with the previous section    
    p = Path('randomForest/csvDataFrames')

    # Find the files; this is a generator, not a list
    files = (p.glob('ticker_*.csv'))
    
    for file in files:
        # Read the file
        df = pd.read_csv(file)
        
        # Calculate the change in price
        delta = df['Close'].diff().dropna()
                
        # Calculate momentum since we want to predict if the stock goes up and down, not the price itself
        # Relative Strength Index
        # RSI > 70 - overbought
        # RSI < 30 - oversold
        
        # Calculate the 14 day RSI
        rsi_period = 14
        
        # Separate data frames into average change in price up and down
        # Absolute values for down average change in price
        
        up_df = delta.clip(lower=0)
        down_df = delta.clip(upper=0).abs()
    
        # Calculate the EWMA (Exponential Weighted Moving Average), older values are given less weight compared to newer values
        # Relative strenth formula
        # Calculate the exponential moving average (EMA) of the gains and losses over the time period
        
        ewma_gain = up_df.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        ewma_loss = down_df.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        
        # Calculate the Relative Strength
        relative_strength = ewma_gain / ewma_loss

        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

        # Add the info to the data frame.
        df['Delta'] = delta
        df['Down_price'] = down_df
        df['Up_price'] = up_df
        df['RSI'] = relative_strength_index
        
        print(relative_strength_index.tail(10))
        
        df.to_csv(file, index=False)
        
         # Display the head.
        print(df.head())
        
        '''
            
        print(up_df)
        print(down_df)
            
        return
        '''
    

def main():
    data_folder = 'randomForest/csvDataFrames/price_data.csv'
    
    # Prevent re-pulling data
    if os.path.exists(data_folder):
        # Load the data
        df = pd.read_csv(data_folder)
    else:
        # Grab the data and store it.
        get_price_data()
        clean_data()
        add_data()
        
        # df.to_csv(data_folder, index=False)
        
        # Load the data
        # df = pd.read_csv(data_folder)
        
    # Display the head
    # print(df.head())
    
    # Display NaN
   #  print(df[df.isna().any(axis = 1)])
        
if __name__ == "__main__":
    main()

