import pandas as pd
from pathlib import Path

# Function to simulate trading for a specific stock
def simulate_trading(file_path, initial_money, days_to_simulate):
    data = pd.read_csv(file_path)

    money = initial_money
    shares = 0

    # Ensure the input number of days is within the available data range
    days_to_simulate = min(days_to_simulate, len(data))

    # Simulate trading based on predictions
    for index, row in data.iterrows():
        if index >= days_to_simulate:
            break

        if row['Direction_prediction'] == 1.0:
            # Buy shares if prediction is positive
            if money > 0:
                shares_to_buy = money / row['Close']
                shares += shares_to_buy
                money = 0
        elif row['Direction_prediction'] == -1.0:
            # Sell shares if prediction is negative
            if shares > 0:
                money += shares * row['Close']
                shares = 0
        elif row['Direction_prediction'] == 0.0:
            # No action for flat days, maintain the current position
            pass
    
    # Calculate final value of the investment
    final_value = money + (shares * data.iloc[days_to_simulate - 1]['Close'])
    return final_value