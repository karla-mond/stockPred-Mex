import pandas as pd

# Load the CSV file containing the predictions
file_path = 'randomForest/csvDataFrames/ticker_AAPL.csv'
data = pd.read_csv(file_path)

# Initialize variables
initial_money = 10000  # Replace with your initial investment amount
money = initial_money
shares = 0
days = 30 # Replace with days to simulate

# Ensure the input number of days is within the available data range
days_to_simulate = min(days, len(data))

# Simulate trading based on predictions
for index, row in data.iterrows():
    if row['Direction_prediction'] == 1.0:
        # Buy shares if prediction is positive
        if money > 0:
            shares_to_buy = money / row['Close']
            shares += shares_to_buy
            money = 0
    elif row['Direction_prediction'] == 0.0:
        # No action for flat days
        pass
    else:
        # Sell shares if prediction is negative
        if shares > 0:
            money += shares * row['Close']
            shares = 0

# Calculate final value of the investment
final_value = money + (shares * data.iloc[days_to_simulate - 1]['Close'])

print(f"Initial investment: ${initial_money:.2f}")
print(f"Final value after {days_to_simulate} days: ${final_value:.2f}")
