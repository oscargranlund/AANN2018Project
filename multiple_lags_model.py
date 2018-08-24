from math import sqrt
import pandas as pd 
import matplotlib.pyplot as plt 
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

DATA_DIR = 'data'
location = 'korsnas_2017'
data_path = os.path.join(DATA_DIR, '{}.xlsx'.format(location))

df = pd.read_excel(data_path, index_col='Date')
abr = {'Pressure': 'P', 'Wind_speed': 'Wnd_s', 'Air_temperature':'Tmp'}
df = df.rename(abr, axis='columns')

features = ['P', 'Wnd_s', 'Tmp']
target = 'Tmp'
train_size = 0.8
epochs = 50
do_scaling = True
steps = [3, 6, 12, 24]

plot_start = '2017-10-25 00:00:00'
plot_end = '2017-11-01 00:00:00'

n_features = len(features)
n_hours = len(steps)
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
print(df.head(5), '\n')
df_y = df[target].to_frame()
df_X = df.drop(features, axis=1)
print('X data shape: {0}\n'.format(df_X.shape))
print(df_X.head())
print('Y data shape: {0}\n'.format(df_y.shape))
print(df_y.head())

# Need some scaling
if do_scaling:
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_data = scaler.fit_transform(df_X.values)
    y_data = scaler.fit_transform(df_y.values)
else:
    X_data = df_X.values
    y_data = df_y.values
# Splitting into train and test data
n = int(train_size * len(X_data))
X_train, X_test = X_data[:n, :], X_data[n:, :]
y_train, y_test = y_data[:n], y_data[n:]


# LSTM needs a 3-D shape
X_train = X_train.reshape((X_train.shape[0], n_hours, n_features))
X_test = X_test.reshape((X_test.shape[0], n_hours, n_features))

print('X_train reshaped: ', X_train.shape)
print('X_test reshape: ', X_test.shape)

# Defining the model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='MSE', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=epochs, batch_size=72, validation_data=(X_test, y_test), verbose=1, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
 
# # make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], n_hours*n_features))
# # invert scaling for forecast
#inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
if do_scaling:
    yhat = scaler.inverse_transform(yhat)
#inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
#inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
if do_scaling:
    y_test = scaler.inverse_transform(y_test)
#inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(y_test, yhat))
print('Test RMSE: %.3f' % rmse)

y_test = pd.Series(y_test.flatten(), index = df_y.iloc[n:, :].index)
yhat = pd.Series(yhat.flatten(), index = df_y.iloc[n:, :].index)

def plot_predictions(y_test, yhat, start, end):
    y_test[start: end].plot(label = 'Test Values')
    yhat[start: end].plot(label = 'Predictions')
    #plt.plot(inv_y[-100:], label='Test Values')
    #plt.plot(inv_yhat[-100:], label='Predictions')
    plt.title(target + ' predictions in ' + location)
    plt.legend()
    plt.show()


plot_predictions(y_test, yhat, plot_start, plot_end)