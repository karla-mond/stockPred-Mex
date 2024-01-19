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

$$ OBV(t) = \begin{cases}
    OBV(t-1) + \text{Vol}(t) & \text{if } C(t) > C(t-1) \\
    OBV(t-1) - \text{Vol}(t) & \text{if } C(t) < C(t-1) \\
    OBV(t-1) & \text{if } C(t) = C(t-1)
\end{cases}
$$

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

### Entropy 

- Measure randomness or uncertainty

- Information gain
  - Difference in entropy before and after the split
  - Low entropy: Low randomness
  - High entropy: High randomness

$$ H(t) = -\sum_{i=1}^{c} p(i | t) \cdot \log_2(p(i | t)) $$
$$ H(t) = -[p(0 | t) \cdot \log_2(p(0 | t)) + p(1 | t) \cdot \log_2(p(1 | t))] $$

- p(i|t) : probability of choosing an element of class i given that the element is in node t.
- C : number of classes.

## Random Forest Methodology

- Training multiple decision trees
- Randomly selecting m = √ M features out of M and n out of N samples
- Bagging (Bootstrap Aggregating)
  - Training each tree on a random subset of the training data with replacement
- Increasing the number of trees stabilizes results by continuous re-sampling of data
- OOB: unused data for performance evaluation

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/03aab2b7-bafa-4939-ac52-aa19f3a07058)

## Evaluation

- How well the final predicted classes align with the actual classes in the dataset
- EnhancementRandomizedSearchCV()
  - Best set of hyper parameters which gives the best score
  - Parameters are optimized by cross-validation

$$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$

Portion of all testing samples classified correctly

$$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}} $$

Not label as positive, a negative sample

$$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}} $$

Correctly identify positive labels 

$$ \text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives + False Positives}} $$

Correctly identify negative labels

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}} $$

Harmonic mean of precision and recall

## Model Evaluation

### Google

Original

Confusion Matrix           |  ROC Cruve               | Feature Importance         |
:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/0fd714bc-a221-4b84-bb5a-719e38cdc69b)  | ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/23e8bac9-4f9b-4f30-9edf-bb861408c60e) | ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/af2e0e27-dce3-4ec6-8b06-a9a35aaf1281)

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/4f8d3fd0-630b-4127-b3fa-f2d0faf98673)

Enhanced

Confusion Matrix           |  ROC Cruve               | Feature Importance         |
:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/0050f68a-0270-49a2-ad75-f4bd2dcebac6) | ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/0e239513-f5da-487e-95b7-b6f685f38521) | ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/d4d39f35-95dc-41a8-b89f-bebe4103b859)

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/14dd4670-6725-4d24-a740-574c578994f2)

### MSFT

Original

Confusion Matrix           |  ROC Cruve               | Feature Importance         |
:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/c77978cf-b8f1-4451-9a12-fc515eb17043)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/a5bdfffd-37ff-4bab-bcf1-038430797d78)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/6f65113b-139b-46e7-af2c-211646e7df65)|

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/800e121c-7a16-47df-ac56-632649630ff1)


Enhanced

Confusion Matrix           |  ROC Cruve               | Feature Importance         |
:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/4c89c81e-652e-413a-a312-752589154925)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/27381037-bc25-419c-b44f-d5b913a2a557)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/cac3aef3-3ffb-4e97-9df1-692810b70c75)|

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/e89d19ad-2165-43b2-90e8-1ca4ec3613e1)

### V

Original

Confusion Matrix           |  ROC Cruve               | Feature Importance         |
:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/fe711789-2095-4bfa-9ac7-e6f0685a9678)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/06cb585b-289e-4122-9b92-1108d4f7e46b)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/bda689e3-9771-4db8-afda-97ac64960883)|

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/4bec3601-c6d9-445d-83e6-ffbc6409059f)

Enhanced

Confusion Matrix           |  ROC Cruve               | Feature Importance         |
:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/adf80a4e-2508-4974-998b-0ea51c0a4c7e)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/42ed5cfa-819b-4561-9940-bb6f7f9c7b53)| ![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/3a08c041-b438-47e7-90eb-ee660242349d)|

![image](https://github.com/karla-mond/stockPred-Mex/assets/78885738/e537cfed-a81b-49f6-a0e5-534c4e6f4efc)

### Comparison

| Metrics            | GOOG        | MSFT        | V           | 
| -----------        | ----------- | ----------- | ----------- |
| Correct Prediction | **69.10**   | 67.47       | 65.85       |
| Accuracy           | **0.69**    | 0.67        | 0.65        |
| Out-Of-Bag         | 0.71        | **0.74**    | 0.66        |

## Simulation

Results of the simulation with a starting balance of $10,000 for each stock

Original 

| Days        | GOOG          | MSFT         | V           | 
| ----------- | -----------   | -----------  | ----------- |
| 7           | $7504.29      | $9088.34     | $7124.58    |
| 30          | **$20428.04** | $19963.9     | $9032.84    |
| 60          | **$43432.95** | $11934.86    | $4585.1     |
| 90          | $33186.53     | **$3837.84** | $3091.25    |


Enhanced

| Days        | GOOG          | MSFT         | V           | 
| ----------- | -----------   | -----------  | ----------- |
| 7           | $7504.29      | $7185.86     | $7411.16    |
| 30          | **$20608.03** | $15023.71    | $7506.98    |
| 60          | **$59505.92** | $9204.86     | $3845.99    |
| 90          | **$67393.19** | $4193.66     | $3059.73    |

## XGBoost: Extreme Gradient Boosting

## LSTM

