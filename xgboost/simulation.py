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

        if row['Direction_prediction'] == 2:
            # Buy shares if prediction is positive
            if money > 0:
                shares_to_buy = money / row['Close']
                shares += shares_to_buy
                money = 0
        elif row['Direction_prediction'] == 0:
            # Sell shares if prediction is negative
            if shares > 0:
                money += shares * row['Close']
                shares = 0
        elif row['Direction_prediction'] == 1:
            # No action for flat days, maintain the current position
            pass
    
    # Calculate final value of the investment
    final_value = money + (shares * data.iloc[days_to_simulate - 1]['Close'])
    return final_value

# Read in multiple files saved with the previous section
p = Path('xgboost/csvDataFrames')
files = p.glob('ticker_*.csv')

# Input number of days to simulate
days_to_simulate = 100 # Replace with days to simulate

# Initialize variables for global sum
global_initial_money = 10000 # Replace with your initial investment amount
global_final_value = 0

# Simulate trading for each stock
for file in files:
    print(f"Processing file: {file}")
    stock_final_value = simulate_trading(file, global_initial_money, days_to_simulate)
    global_final_value += stock_final_value
    print(f"Final value after {days_to_simulate} days: ${stock_final_value:.2f}")

# Print the global sum
print(f"\nInitial investment: ${global_initial_money:.2f}")
print(f"Global final value for all stocks: ${global_final_value:.2f}")
