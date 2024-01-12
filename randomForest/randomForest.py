# Import libraries
import os
import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
from sklearn.metrics import RocCurveDisplay

from pathlib import Path

'''
# read the files to create a single dataframe
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    
'''
def get_price_data():
    # Grabbing Historical Price Data

    # Supports more than 1 ticker.
    # S&P500 - ^GSPC
    tickerStrings = ['GOOG', 'MSFT', 'V']
    
    for ticker in tickerStrings:
        # Last 2 years, with daily frequency
        # Candles in yf.dowload - Date,Open,High,Low,Close,Adj Close,Volume
        
        df = yf.download(ticker, group_by="Ticker", period='2y', interval='1d')
        
        # add this column because the dataframe doesn't contain a column with the ticker
        df['Symbol'] = ticker  
        df.to_csv(f'randomForest/csvDataFrames/ticker_{ticker}.csv')
    
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

def add_data():
    # Read in multiple files saved with the previous section    
    p = Path('randomForest/csvDataFrames')

    # Find the files; this is a generator, not a list
    files = (p.glob('ticker_*.csv'))
    
    def relative_strength_index(df):
        # Calculate the change in price
        delta = df['Close'].diff().dropna()
       
        ''' 
        # define the number of days out you want to predict
        days_out = 30

        # Calculate ema to give more weight to recent values while smoothing out fluctuation
        price_data_smoothed = df[['Close', 'Low', 'High', 'Open', 'Volume']].ewm(span=days_out).mean()

        # Join the smoothed columns with the symbol and datetime column
        smoothed_df = pd.concat([df[['Symbol', 'Date']], price_data_smoothed], axis=1)
        
        # Calculate the flag, and calculate the diff compared to 30 days ago
        smoothed_df['Signal_Flag'] = np.sign(df['Close'].diff(days_out))

        # print the first 50 rows
        print(smoothed_df.head(50))
        '''
        
        # Calculate momentum since we want to predict if the stock goes up and down, not the price itself
        # Momentum indicator Relative Strength Index
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
        
        ewma_gain = up_df.ewm(span=rsi_period).mean()
        ewma_loss = down_df.ewm(span=rsi_period).mean()
        
        # Calculate the Relative Strength
        relative_strength = ewma_gain / ewma_loss
        
        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

        # Store the data in the data frame.
        df['Delta'] = delta
        df['Down_price'] = down_df
        df['Up_price'] = up_df
        df['RSI'] = relative_strength_index
    
    def stochastic_oscillator(df): 
        
        so_period = 14
        
        # Apply the rolling function and grab the Min and Max
        low_low = df["Low"].rolling(window = so_period).min()
        high_high =df["High"].rolling(window = so_period).max()

        # Calculate the momentum indicator Stochastic Oscillator. Relation to the lowest price
        stochastic_oscillator = 100 * ((df['Close'] - low_low) / (high_high - low_low))

        # Store the data in the data frame.
        df['Lowest_low'] = low_low
        df['Highest_high'] = high_high
        df['SO'] = stochastic_oscillator
        
    def williams_r(df):
        
        # WR > -20 is sell signal
        # WR < -80 is buy signal
        
        # William R period depends on SO period

        # Calculate the momentum indicator Williams %R. Relation to the highest price
        r_percent = ((df['Highest_high'] - df['Close']) / (df['Highest_high'] - df['Lowest_low'])) * - 100

        # Store the data in the data frame.
        df['R_percent'] = r_percent
        
    def macd(df):
        
        # MACD goes below the SingalLine -> sell signal. Above the SignalLine -> buy signal.
        
        # Calculate the MACD
        ema_26 = df['Close'].ewm(span=26).mean()
        ema_12 = df['Close'].ewm(span=12).mean()
        macd = ema_12 - ema_26

        # Calculate the EMA of the MACD
        ema_9_macd = macd.ewm(span=9).mean()

        # Store the data in the data frame.
        df['MACD'] = macd
        df['MACD_EMA'] = ema_9_macd
        
    def price_rate_change(df):
        
        # Measures the most recent change in price with respect to the price in n days ago.
        
        # Standard window of 9.
        n = 9

        # Calculate and store the Rate of Change in the Price
        df['Price_Rate_Of_Change'] = df['Close'].pct_change(periods=n)
        
    def obv(df):
        
        # Uses changes in volume to estimate changes in stock prices by considering the cumulative volume.
        
        df['Delta'] = df['Close'].diff().fillna(0)
        obv_values = (df['Delta'] > 0).astype(int) * df['Volume'] - (df['Delta'] < 0).astype(int) * df['Volume']
        
        # Store the data in the data frame.
        df['OBV'] = obv_values.cumsum()

    def direction_prediction(df):
        
        # Predict closing direction
        # -1.0 for negative values (down days)
        # 1.0 for postive values
        # 0.0 for no change (flat days)

        direction_predictions = np.sign(df['Delta'])
        
        direction_predictions[direction_predictions==0.0] = 1.0
        
        # Store the data in the data frame.
        df['Direction'] = direction_predictions
        
    for file in files:     
        df = pd.read_csv(file)
               
        relative_strength_index(df)
        stochastic_oscillator(df)
        williams_r(df)
        macd(df)
        price_rate_change(df)
        obv(df)
        direction_prediction(df)
        
        df.to_csv(file, index=False)

def predict():
    
    # Read in multiple files saved with the previous section    
    p = Path('randomForest/csvDataFrames')

    # Find the files; this is a generator, not a list
    files = (p.glob('ticker_*.csv'))
    
    def preprocess_data(df):
        
        # Drop al NaN values before feeding it to the model
        df = df.dropna()
        return df
        
    def split_data(df):
        # Split into training and test set
        X_cols = df[['RSI', 'SO', 'R_percent', 'MACD', 'Price_Rate_Of_Change', 'OBV']]
        Y_cols = df['Direction']
        
        X_train, X_test, y_train, y_test = train_test_split(X_cols, Y_cols, random_state=0)
        
        return X_train, X_test, y_train, y_test, X_cols
    
    def train_model(X_train, y_train):
        # Random Forest Classifier
        # 100 trees
        # 00B used later
        rand_frst_clf = RandomForestClassifier(bootstrap=True, n_estimators=100, oob_score=True, criterion="gini", random_state=0)
        
        # Fit the data to the model
        rand_frst_clf.fit(X_train, y_train)
        return rand_frst_clf
    
    def evaluate_model(file, clf, X_test, y_test, X_cols):
        y_pred = clf.predict(X_test)
        
        # Save results for market simulation
        testing_indices = X_test.index
        close_values = df.loc[testing_indices, 'Close']
        result_df = pd.DataFrame({'Direction_prediction': y_pred, 'Close': close_values}, index=testing_indices)
        result_df.to_csv(f"{file}_results", header=True, index=True)
                
        accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100.0
        print('Correct Prediction (%): ', accuracy)

        # Use F-score to measure Recall and Precision at the same time through Harmonic Mean.
        report = classification_report(y_true=y_test, y_pred=y_pred, target_names=['Down Day', 'Up Day'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        print('Classification Report: ', report_df)
        
        rf_matrix = confusion_matrix(y_test, y_pred)

        true_negatives = rf_matrix[0][0]
        false_negatives = rf_matrix[1][0]
        true_positives = rf_matrix[1][1]
        false_positives = rf_matrix[0][1]

        accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
        percision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)

        print('Accuracy: {}'.format(float(accuracy)))
        print('Percision: {}'.format(float(percision)))
        print('Recall: {}'.format(float(recall)))
        print('Specificity: {}'.format(float(specificity)))
        
        # Normalized confusion matrix
        ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, normalize='true', display_labels=['Down Day', 'Up Day'], cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Normalized')
        plt.show()
        
        # Gini impurity. Misclassified result.
        feature_imp = pd.Series(clf.feature_importances_, index=X_cols.columns).sort_values(ascending=False)
        print(feature_imp)
        
        # Feature Importance Graph
        x_values = list(range(len(clf.feature_importances_)))
        cumulative_importances = np.cumsum(feature_imp.values)
        plt.plot(x_values, cumulative_importances, 'g-')

        # Draw line at 95% of importance retained
        plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')
        plt.xticks(x_values, feature_imp.index, rotation = 'vertical')
        plt.xlabel('Variable')
        plt.ylabel('Cumulative Importance')
        plt.title('Random Forest: Feature Importance Graph')

        # ROC Curve to select optimal model, far from 45 degrees diagonal of ROC space
        RocCurveDisplay.from_estimator(clf, X_test, y_test)
        plt.show()
        
        # Parameter ideally similar to accuracy score 
        print('Random Forest Out-Of-Bag Error Score: {}'.format(clf.oob_score_))

    def train_model_enhanced():
        
        # Tree Number
        n_estimators = list(range(200, 2000, 200))
        
        # Number of features to consider at every split
        max_features = [None, 'sqrt', 'log2']
        
        # Max tree level number 
        max_depth = list(range(10, 110, 10))
        max_depth.append(None)
        
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 20, 30, 40]
        
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 7, 12, 14, 16, 20]
        
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
    
        # Find the best parameters
        rf_random_clf = RandomizedSearchCV(estimator = RandomForestClassifier(oob_score=True), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

        # Fit the random search model
        rf_random_clf.fit(X_train, y_train)
        
        return rf_random_clf
    
    for file in files:     
        df = pd.read_csv(file)
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test, X_cols = split_data(df)
        # print(file)
        # clf = train_model(X_train, y_train)
        # evaluate_model(file, clf, X_test, y_test, X_cols)
        print(file)
        clf_enhanced = train_model_enhanced()
        evaluate_model(file, clf_enhanced.best_estimator_, X_test, y_test, X_cols)
        
        df.to_csv(file, index=False)    

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
        predict()
        
        # Load the data
        # df = pd.read_csv(data_folder)
        
    # Display the head
    # print(df.head())

if __name__ == "__main__":
    main()