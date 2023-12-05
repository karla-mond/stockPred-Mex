import yfinance as yf
import os
import pandas as pd

def get_price_data():
    # Grabbing Historical Price Data

    # Supports more than 1 ticker.
    # S&P500 - ^GSPC
    tickerStrings = ['GOOG', 'MSFT', 'V']
    
    for ticker in tickerStrings:
        # Last 2 days, with daily frequency
        # Candles in yf.dowload - Date,Open,High,Low,Close,Adj Close,Volume
        
        df = yf.download(ticker, group_by="Ticker", period='2y', interval='1d')
        
        # add this column because the dataframe doesn't contain a column with the ticker
        df['Symbol'] = ticker  
        df.to_csv(f'/stockPred-Mex/dataBase/ticker_{ticker}.csv')
        
        print(df.head())
        
def main():
        get_price_data()
        
if __name__ == "__main__":
    main()

