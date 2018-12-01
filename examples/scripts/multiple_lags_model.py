from math import sqrt
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from utils import load_data, add_shifted_features

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM


loc_target = 'kuopio-maaninka_2017'
loc_neighbors = [] #['kasko_2017', 'hango-russaro_2017', 'utsjoki-nuorgam_2017'] # leave empty when no neighbours in model
has_winddir=False
abr = {'Pressure': 'P', 'Wind_speed': 'Wnd_s', 'Air_temperature':'Tmp', 'Wind_direction':'Wnd_dir', 'Cloud_amount': 'Clouds'}

features = ['P', 'Tmp', 'Wnd_s']
target = 'Tmp'
if 'Wnd_h' in features and 'Wnd_v' in features: # = Wind east-west, Wind north-south
    has_winddir = True

train_size = 0.8
epochs = 30
do_scaling = True
steps = [3]

plot_start = '2017-10-28 00:00:00'
plot_end = '2017-11-01 00:00:00'

n_features = len(features * (1 + len(loc_neighbors)))
n_hours = len(steps)

#data_path = os.path.join(DATA_DIR, '{}.xlsx'.format(loc_target))
df = load_data(loc_target + '.xlsx')#, rename_abr=abr)
if has_winddir:
    df['Wnd_v'] = np.sin(df['Wnd_dir'])
    df['Wnd_h'] = np.cos(df['Wnd_dir'])
df = add_shifted_features(df, features, steps)

# Adding data for neighbors
for n in loc_neighbors:
    data_path = os.path.join(DATA_DIR, '{}.xlsx'.format(n))
    df_neighbor = load_data(data_path, rename_abr=abr)
    if has_winddir:
        df_neighbor['Wnd_v'] = np.sin(df_neighbor['Wnd_dir'])
        df_neighbor['Wnd_h'] = np.cos(df_neighbor['Wnd_dir'])
    
    df_neighbor = add_shifted_features(df_neighbor, features, steps)
    df_neighbor = df_neighbor.drop(features, axis=1)
    df = df.join(df_neighbor, on='Date', rsuffix='_' + n[:4])

df = df.dropna()
print(df.head(5), '\n')
print(df.describe())
df_y = df[target].to_frame()
df_X = df.drop(features, axis=1)
#if 'Wnd_dir' in features:
#    df_X = df_X.drop(['Wnd_v', 'Wnd_h'])

print('X data shape: {0}\n'.format(df_X.shape))
print(df_X.head())
print('Y data shape: {0}\n'.format(df_y.shape))
print(df_y.head())

# Scaling the data to (0,1)
if do_scaling:
    scaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(0, 1))
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
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
#model.add(Dense(1))
model.add(Dropout(0.2))

model.add(LSTM(15, return_sequences=True))

model.add(Dropout(0.2))
    
model.add(LSTM(20, return_sequences=False))
model.add(Dropout(0.2))
    
model.add(Dense(1, activation="linear"))
    
model.compile(loss="mse", optimizer='rmsprop')
    
print(model.summary())

#model.compile(loss='MSE', optimizer='rmsprop')
# fit network
history = model.fit(X_train, y_train, epochs=epochs, batch_size=72, validation_data=(X_test, y_test), verbose=1, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
 
# # make a prediction
yhat = model.predict(X_test)

if do_scaling:
    yhat = scaler.inverse_transform(yhat)

y_test = y_test.reshape((len(y_test), 1))

if do_scaling:
    y_test = scaler.inverse_transform(y_test)

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
    plt.title(target + ' predictions in ' + loc_target)
    plt.legend()
    plt.show()


plot_predictions(y_test, yhat, plot_start, plot_end)
