## Devloped by: LOKESH KUMAR P
## Register Number: 212222240054
## Date: 

# Ex.No: 07                         AUTO-REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM :

### Step 1 :

Import necessary libraries.

### Step 2 :

Read the CSV file into a DataFrame.

### Step 3 :

Perform Augmented Dickey-Fuller test.

### Step 4 :

Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags.

### Step 5 :

Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF).

### Step 6 :

Make predictions using the AR model.Compare the predictions with the test data.

### Step 7 :

Calculate Mean Squared Error (MSE).Plot the test data and predictions.

### PROGRAM

#### Import necessary libraries :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```

#### Read the CSV file into a DataFrame :

```python
data = pd.read_csv('JPMorgan case.csv', parse_dates=['date'], index_col='date')
```
```
data = data[['close']]
````

```
result = adfuller(data['close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
![image](https://github.com/user-attachments/assets/3253126f-19bb-450f-ad0a-192d3fb18a49)

### Step 4: Split into train/test
```
x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]
```
### # Step 5: Fit AutoRegressive model with 13 lags
```
lag_order = 13
model = AutoReg(train_data['close'], lags=lag_order)
model_fit = model.fit()
```

```
plt.figure(figsize=(10, 6))
plot_acf(data['close'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Close Price')
plt.show()
```
![image](https://github.com/user-attachments/assets/8a24fc58-11ba-4570-b9cb-56e1d8a72c9a)

```
plt.figure(figsize=(10, 6))
plot_pacf(data['close'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Close Price')
plt.show()
mse = mean_squared_error(test_data['close'], predictions)
print('Mean Squared Error (MSE):', mse)
```
![image](https://github.com/user-attachments/assets/e3a838b7-955a-4343-8236-a2b29851b7ed)

```
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)
```
### Step 9: Plot actual vs predicted
```
plt.figure(figsize=(12, 6))
plt.plot(test_data['close'], label='Test Data - Close Price')
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
plt.xlabel('date')
plt.ylabel('Close Price')
plt.title('AR Model Predictions vs Test Data (Stock Price)')
plt.legend()
plt.grid()
plt.show()
```
![image](https://github.com/user-attachments/assets/a6f50e32-d5c7-44cd-9e0a-7296b5a08c96)
### RESULT:
Thus we have successfully implemented the auto regression function using python.


