#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


def preprocess_data(data, time_step):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main():
    
    time_step = 60 

    # Load the dataset
    print("Loading stock data...")
    data = pd.read_csv("Accenture_stock_history.csv")
    data['Date'] = pd.to_datetime(data['Date'])  
    data.set_index('Date', inplace=True)         
    close_prices = data['Close']                


    X, y, scaler = preprocess_data(close_prices, time_step)

    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

   
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

   
    print("Building and training the model...")
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

  
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

 
    print("Plotting the results...")
    plt.figure(figsize=(12, 6))
    plt.plot(close_prices.index[-len(y_test):], y_test, label='True Price', color='blue')
    plt.plot(close_prices.index[-len(predictions):], predictions, label='Predicted Price', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()







