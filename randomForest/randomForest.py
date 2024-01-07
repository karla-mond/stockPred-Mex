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

        # Add the info to the data frame.
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

        # Add the info to the data frame
        df['Lowest_low'] = low_low
        df['Highest_high'] = high_high
        df['SO'] = stochastic_oscillator
        
    def williams_r(df):
        
        # WR > -20 is sell signal
        # WR < -80 is buy signal
        
        # William R period depends on SO period

        # Calculate the momentum indicator Williams %R. Relation to the highest price
        r_percent = ((df['Highest_high'] - df['Close']) / (df['Highest_high'] - df['Lowest_low'])) * - 100

        # Add the info to the data frame
        df['R_percent'] = r_percent
        
    def macd(df):
        
        # MACD goes below the SingalLine -> sell signal. Above the SignalLine -> buy signal.
        
        # Calculate the MACD
        ema_26 = df['Close'].ewm(span = 26).mean()
        ema_12 = df['Close'].ewm(span = 12).mean()
        macd = ema_12 - ema_26

        # Calculate the EMA of the MACD
        ema_9_macd = macd.ewm(span = 9).mean()

        # Store the data in the data frame.
        df['MACD'] = macd
        df['MACD_EMA'] = ema_9_macd
        
    def price_rate_change():
        
        # Read updated file 
        df = pd.read_csv(file)
        
        # Calculate the Price Rate of Change
        n = 9

        # Calculate the Rate of Change in the Price, and store it in the Data Frame.
        df['Price_Rate_Of_Change'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(periods = n))
        
        df.to_csv(file, index=False)
        
    def obv(group):
        
        # Read updated file 
        df = pd.read_csv(file)

        # Grab the volume and close column.
        volume = group['volume']
        change = group['close'].diff()

        # intialize the previous OBV
        prev_obv = 0
        obv_values = []

        # calculate the On Balance Volume
        for i, j in zip(change, volume):

            if i > 0:
                current_obv = prev_obv + j
            elif i < 0:
                current_obv = prev_obv - j
            else:
                current_obv = prev_obv

            # OBV.append(current_OBV)
            prev_obv = current_obv
            obv_values.append(current_obv)
        
        # Return a panda series.
        return pd.Series(obv_values, index = group.index)
            

        # apply the function to each group
        obv_groups = df.groupby('symbol').apply(obv)

        # add to the data frame, but drop the old index, before adding it.
        df['On Balance Volume'] = obv_groups.reset_index(level=0, drop=True)
        
        df.to_csv(file, index=False)
    
    def pred_column():
        
        # Read updated file 
        df = pd.read_csv(file)
        
        # Create a column we wish to predict


        # Group by the `Symbol` column, then grab the `Close` column.
        close_groups = df.groupby('symbol')['close']

        # Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
        close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

        # add the data to the main dataframe.
        df['Prediction'] = close_groups

        # for simplicity in later sections I'm going to make a change to our prediction column. To keep this as a binary classifier I'll change flat days and consider them up days.
        df.loc[df['Prediction'] == 0.0] = 1.0
    
    def split_data():
        # Read updated file 
        df = pd.read_csv(file)
        
        # Grab our X & Y Columns.
        X_Cols = df[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD','On Balance Volume']]
        Y_Cols = df['Prediction']

        # Split X and y into X_
        X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)

        # Create a Random Forest Classifier
        rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

        # Fit the data to the model
        rand_frst_clf.fit(X_train, y_train)

        # Make predictions
        y_pred = rand_frst_clf.predict(X_test)
    
    def interpret():
        # Define the traget names
        target_names = ['Down Day', 'Up Day']

        # Build a classifcation report
        report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)

        # Add it to a data frame, transpose it for readability.
        report_df = pd.DataFrame(report).transpose()
        report_df
        
    def confusion_matrix():
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

        disp = plot_confusion_matrix(rand_frst_clf, X_test, y_test, display_labels = ['Down Day', 'Up Day'], normalize = 'true', cmap=plt.cm.Blues)
        disp.ax_.set_title('Confusion Matrix - Normalized')
        plt.show()
    
    def split_data():
        # Read updated file 
        df = pd.read_csv(file)
        
        # Grab our X & Y Columns.
        X_Cols = df[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD','On Balance Volume']]
        Y_Cols = df['Prediction']

        # Split X and y into X_
        X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)

        # Create a Random Forest Classifier
        rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

        # Fit the data to the model
        rand_frst_clf.fit(X_train, y_train)

        # Make predictions
        y_pred = rand_frst_clf.predict(X_test)
        
        # Print the Accuracy of our Model.
        print('Correct Prediction (%): ', accuracy_score(y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0)
        
        # Define the traget names
        target_names = ['Down Day', 'Up Day']

        # Build a classifcation report
        report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)

        # Add it to a data frame, transpose it for readability.
        report_df = pd.DataFrame(report).transpose()
        report_df
        
        # store the values in a list to plot.
        x_values = list(range(len(rand_frst_clf.feature_importances_)))

        # Cumulative importances
        cumulative_importances = np.cumsum(feature_imp.values)

        # Make a line graph
        plt.plot(x_values, cumulative_importances, 'g-')

        # Draw line at 95% of importance retained
        plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')

        # Format x ticks and labels
        plt.xticks(x_values, feature_imp.index, rotation = 'vertical')

        # Axis labels and title
        plt.xlabel('Variable')
        plt.ylabel('Cumulative Importance')
        plt.title('Random Forest: Feature Importance Graph')
        
        # Create an ROC Curve plot.
        rfc_disp = plot_roc_curve(rand_frst_clf, X_test, y_test, alpha = 0.8)
        plt.show()
        
        print('Random Forest Out-Of-Bag Error Score: {}'.format(rand_frst_clf.oob_score_))

    def randomized_search():
        # Number of trees in random forest
        n_estimators = list(range(200, 2000, 200))

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', None, 'log2']

        # Maximum number of levels in tree
        max_depth = list(range(10, 110, 10))
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 20, 30, 40]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 7, 12, 14, 16 ,20]

        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        print(random_grid)
        
        # New Random Forest Classifier to house optimal parameters
        rf = RandomForestClassifier()

        # Specfiy the details of our Randomized Search
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

        # Fit the random search model
        rf_random.fit(X_train, y_train)
        
        # With the new Random Classifier trained we can proceed to our regular steps, prediction.
        rf_random.predict(X_test)

        # Accuracy
        print('Correct Prediction (%): ', accuracy_score(y_test, rf_random.predict(X_test), normalize = True) * 100.0)


        # CLASSIFICATION REPORT
        
        target_names = ['Down Day', 'Up Day']

        report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)

        report_df = pd.DataFrame(report).transpose()
        display(report_df)
        print('\n')

        # Feature importance
        feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)
        display(feature_imp)
        
        # Roc Curve

        fig, ax = plt.subplots()

        rfc_disp = plot_roc_curve(rand_frst_clf, X_test, y_test, alpha = 0.8, name='ROC Curve', lw=1, ax=ax)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve Random Forest")

        ax.legend(loc="lower right")

        plt.show()

        
    for file in files:     
        df = pd.read_csv(file)
               
        relative_strength_index(df)
    
        stochastic_oscillator(df)
        
        williams_r(df)
        
        macd(df)
        
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
        
        # df.to_csv(data_folder, index=False)
        
        # Load the data
        # df = pd.read_csv(data_folder)
        
    # Display the head
    # print(df.head())
    
    # Display NaN
   #  print(df[df.isna().any(axis = 1)])
        
if __name__ == "__main__":
    main()

