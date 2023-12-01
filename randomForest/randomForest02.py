import yfinance as yf
import pandas as pd
from pathlib import Path
import os

def grab_price_data(tickers):
    for ticker in tickers:
        data = yf.download(ticker, group_by="Ticker", period='2y', interval='1d')
        data['Symbol'] = ticker
        data.to_csv(f'randomForest/csvDataFrames/ticker_{ticker}.csv')

def calculate_differences(file_path):
    df = pd.read_csv(file_path)
    df['Difference'] = df['Close'].diff()
    return df[['Symbol', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Difference']]

def load_and_combine_data(folder_path):
    p = Path(folder_path)
    files = p.glob('ticker_*.csv')
    dfs = [calculate_differences(file) for file in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df.sort_values(by=['Symbol', 'Date'])

def main():
    tickers = ['AAPL', 'MSFT']
    data_folder = 'randomForest/csvDataFrames'
    combined_file_path = 'randomForest/csvDataFrames/price_data_combined.csv'

    if os.path.exists(combined_file_path):
        df = pd.read_csv(combined_file_path)
    else:
        grab_price_data(tickers)
        df = load_and_combine_data(data_folder)
        df.to_csv(combined_file_path, index=False)

    print("Combined DataFrame:")
    print(df.head())

if __name__ == "__main__":
    main()
