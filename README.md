# stockPred-Mex
Stock prediction (S&amp;P 100) with Random Forest and XGBoost algortihms

Simulate trading based on stock prediction models using CSV files for multiple stocks.

Key functionalities include:

Reading multiple CSV files containing stock prediction data for different stocks.
A function to simulate trading for a specific stock based on predictions (positive, negative, or flat).
Accumulating the final values of each stock's simulated trading into a global sum.
Prompting users to input the number of days they want to simulate for all stocks.

## Literature

Predicting the direction of stock market prices using tree-based classifiers. (Basak, Kar, Saha, Khaidem, & Dey)

## Objective

- Implement:
  - Random Forest
  - XGBoost
  - LSTM
- Apply:
  - Google
  - Microsoft
  - VISA
- Simulate:
  - Stock market
  - Simulate asset returns
- Compare:
  - Performance

## Challenge

- 3 algorithms
  - Random Forest
  - XGBoost
  - LSTM
   
- 3 stocks
  - Google
  - Microsoft
  - VISA

- $10,000 USD of initial investment
  - Assess the most profitable model

## Data Overview

- Data source:
  - Yahoo Finance
  
- Dataset variables:
  - Date, Open, High, Low, Close, Adj Close, Volume

- Daily closing stock price:
  - Google, Microsoft, VISA

- Time period:
  - 01-01-2021 until 05-12-2023

## Scheme Flowchart

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/06d9ca10-b5e0-4d61-93df-6fcfb7e41c59)


## Stock Market Prediction

- Model
  - Random Forest
  - XGBoost
  - LSTM

- Features
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

## Feature Engineering: XGBoost, Random Forest & LSTM

Model and Features used

- Random Forest & XGBoost
  - Relative Strength Index
  - Stochastic Oscillator
  - Williams Percentage Rate
  - On Balance Volume
  - Price Rate Of Change
  - Moving Average Convergence Divergence

- LSTM
  - Closing Price

Features correspond to the values in the “Y” axis and date for each row in the datasets works as an index and “X” axis

### Feature Engineering: Relative Strength Index

- Determines whether the stock is over-purchased or over-sold.
- Range 0-100
  - RSI > 70 stock overbought (overvalued)
  - RSI < 30 stock oversold (panic selling)

$$ RSI=100-\frac{100}{1+RS} $$

$$ RS=\frac{AverageGainOverPast14Days}{AverageLossOverPast14Days} $$

- RS:  Relative Strength

### Feature Engineering: Stochastic Oscillator

- Measures level of the closing price relative to lo-high range
- Range 0 - 100
  - 0 : Price is trading near the lowest low
  - 100 : Price is trading near the highest high
  
$$ \%K=100\times\frac{\left(C-L_{14}\right)}{\left(H_{14}-L_{14}\right)} $$

- C: Current closing price
- L_14:  Lowest price within the last 14 days
- H_14:  Highest price within the past 14 days

### Feature Engineering: Williams Percentage Rate

-  Inverse of Stochastic Oscillator
-  Measures closing price related to the highest high
-  Ranges -100 to 0
  - PR > -20: Sell signal
  - WPR < -80: Buy signal

$$ \%R=-100\times\frac{\left(H_{14}-C\right)}{\left(H_{14}-L_{14}\right)} $$

- C: Current closing price
- L_14: Lowest price within the las 14 days
- H_14: Highest price within the pas 14 days

### Feature Engineering: Moving Average Convergence Divergence

- Compares two moving averages; 26 day EMA and 12 day EMA
- 9 day EMA is signal line
   - MACD < signal indicates a sell signal
   - MACD > signal indicates a buy signal
  
$$ MACD=EMA_{12}\left(C\right)-EMA_{26}\left(C\right) $$
$$ SignalLine=EMA_9\left(MACD\right) $$

- C: closing price
- EMA_n: n-day exponential moving average

### Feature Engineering: Price Rate Of Change 

- Indicates the percentage change in price between the current price and the price over the time window
- How far and fast prices move

$$ PROC_{t}=\frac{C_{t}-C_{t-n}}{C_{t-n}} $$

- PROC(t):  price rate of change at time t
- C(t): closing price at time t
- C(t-n): closing price at time t-n

### Feature Engineering: On Balance Volume

- Uses changes in volume to estimate change in prices.
- Buying and selling trends using the cumulative volume.
  - Cumulative add if prices go up. 
  - Subtracts volume when prices go down.

$$ OBV\left(t\right)=\left\lbrace\begin{matrix}{OBV\left(t-1\right)}+Vol\left(t\right)...ifC\left(t\right)>C\left(t-1\right)\\ OBV\left(t-1\right)-Vol\left(t\right)...ifC\left(t\right)<C\left(t-1\right)\\ OBV\left(t-1\right)...ifC\left(t\right)=C\left(t-1\right)\end{matrix}\right\rbrace $$

- C(t):  Closing time at time t
- Vol(t): Trading volume at time t
- OBV(t): On balance volume at time t

# Random Forest Algorithm

## Methodology Decision Trees

- CART algorithm does recursive binary splitting
- Child node is split until only pure leaf nodes remain (single class)
  - Test sample is assigned the class label of the training samples of leaf node it arrives at. 
- Decision trees tend to over-fit the training sets.

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/ea606207-bdc0-4b9b-adcb-15ed30cc0716)

## Methodology: Split

### Gini index

- Likelihood randomly selected example would be incorrectly classified
- How impure or pure that split will be
- Ranges
  - 0: all elements belong to the same class
  - 1: only one class exists
 
$$ Gini=1-\overset{{C}}{\underset{i=1}{\Sigma}}p{\left(i^{}\left|t\right.\right)}^2 $$

$$ 
Gini(t) = 1 - \left[p(0 | t)^2 + p(1 | t)^2\right]
$$

- p(i|t) : Probability of choosing an element of class i given that the element is in node t.
- C :  Number of classes.


