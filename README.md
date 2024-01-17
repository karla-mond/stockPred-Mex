# stockPred-Mex
Stock prediction (S&amp;P 100) with Random Forest and XGBoost algortihms

Simulate trading based on stock prediction models using CSV files for multiple stocks.

Key functionalities include:

Reading multiple CSV files containing stock prediction data for different stocks.
A function to simulate trading for a specific stock based on predictions (positive, negative, or flat).
Accumulating the final values of each stock's simulated trading into a global sum.
Prompting users to input the number of days they want to simulate for all stocks.

### Literature

Predicting the direction of stock market prices using tree-based classifiers. (Basak, Kar, Saha, Khaidem, & Dey)

### Objective

- Implement: Random Forest, XGBoost, LSTM
- Apply: Google, Microsoft, VISA
- Simulate: Stock market and simulate asset returns
- Compare: Performance

### Challenge

- 3 algorithms

Random Forest
XGBoost
LSTM

- 3 stocks

Google
Microsoft
VISA

- $10,000 USD of initial investment

Assess the most profitable model

### Data Overview

1. Data source:
   - Yahoo Finance

3. Dataset variables:
   - Date, Open, High, Low, Close, Adj Close, Volume

4. Daily closing stock price:
   - Google, Microsoft, VISA

5. Time period:
   - 01-01-2021 until 05-12-2023

### Scheme Flowchart

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/06d9ca10-b5e0-4d61-93df-6fcfb7e41c59)


### Stock Market Prediction

1. Model
   - Random Forest
   - XGBoost
   - LSTM

2. Features
   - Closing Price
   - Direction Prediction

| Variable           | Input            |
| -----------        | -----------      |
| Days to simulate   | 7, 30, 60 or 90  |
| Initial investment | $10000 per stock |


| Direction Prediction  | Class       | Action       | Calculation                    | 
| -----------           | ----------- | -----------  | -----------                    |
| 1                     | Positive    | Buy          | shares = money / Closing Price |
| -1                    | Negative    | Sell         | money = shares * Closing Price |
| 0                     | Flat day    | None         | pass                           |

### Feature Engineering: XGBoost, Random Forest & LSTM
