from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd 
import os
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt 

DATA_DIR = 'data'
location = 'korsnas'
data_path = os.path.join(DATA_DIR, '{}_2017.xlsx'.format(location))

df = pd.read_excel(data_path, index_col='Date')
abr = {'Pressure': 'P', 'Wind_speed': 'Wnd_s', 'Air_temperature':'Tmp'}
df = df.rename(abr, axis='columns')

features = ['P', 'Wnd_s', 'Tmp']
target = 'Tmp'
train_size = 0.8
epochs = 50
steps = [1,2,3]

#def add_shifted_features(df, features, start=1, end=2, step=1, forecast_step=1, dropnan=True):
def add_shifted_features(df, features, steps, forecast_step=1, dropnan=True):
    
    df_new = df[features].copy()
    for feat in features:
        for i in steps:
            newCol = feat + '(t-{})'.format(str(i))
            df_new[newCol] = df[feat].shift(i)
            
    if dropnan:
        df_new.dropna(axis=0, inplace=True)
    
    return df_new

df = add_shifted_features(df, features, steps)
print(df.head(5))
y_data = df[target]
X_data = df.drop(features, axis=1)
print('X data shape: {0}\n'.format(X_data.shape))
print(X_data.head())
print('Y data shape: {0}\n'.format(y_data.shape))
print(y_data.head())

# Need some scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data.values)
y_scaled = scaler.fit_transform(y_data.values.reshape(-1,1))

# Splitting into train and test data
n = int(train_size * len(X_data))
X_train, X_test = X_scaled[:n, :], X_scaled[n:, :]
y_train, y_test = y_scaled[:n], y_scaled[n:]

# LSTM needs a 3-D shape
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


print('X_train reshaped: ', X_train.shape)
print('X_test reshape: ', X_test.shape)

# Defining the model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=epochs, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# # make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# # invert scaling for forecast
inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y[-100:], label='Test Values')
plt.plot(inv_yhat[-100:], label='Predictions')
plt.title(target + ' predictions in ' + location)
plt.legend()
plt.show()