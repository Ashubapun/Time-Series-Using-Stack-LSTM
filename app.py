import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
# import pandas_datareader as pdr
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

start = '2013-01-01'
end = '2023-01-01'

st.title('Stock Market Prediction')

user_input = st.text_input('Enter the Stock Ticker ', 'AAPL')
df = yf.download(user_input, start = start, end = end)
df = df.round(2)

st.subheader('Data from 2013 - 2023')
st.write(df.describe())

# Visualizations

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (10,10))
plt.plot(df.Close)
st.pyplot(fig)

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

st.subheader('Closing Price vs Time chart with 100 days Moving Average')
fig = plt.figure(figsize = (10,10))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 days and 200 days Moving Averages')
fig = plt.figure(figsize = (10,10))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

# Split data into training and testing
data_train = pd.DataFrame(df['Close'][ : int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7) : ])


# Scaling down the data using MinMax Scaler

scaler = MinMaxScaler(feature_range = (0, 1))   # which is default parameter
data_training = scaler.fit_transform(data_train)

# Load Model
model = load_model('keras_lstm.h5')

#testing

past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Prediction
y_pred = model.predict(x_test)

scale = scaler.scale_
y_pred = y_pred * (1/scale[0])
y_test = y_test * (1/scale[0])

# Plotting

st.subheader("Original vs Prediction")
fig2 = plt.figure(figsize = (10,10))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_pred, 'g', label = 'Prediced Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)



