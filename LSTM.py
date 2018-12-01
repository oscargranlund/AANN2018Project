import keras
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd 
import re
import sklearn.metrics
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop

# Data source: https://en.ilmatieteenlaitos.fi/download-observations#!/

DATA_DIR = "data"

def load_data(name, rename=True):
    path = os.path.join(DATA_DIR, name)
    df = pd.read_excel(path, index_col='Date')
    
    if rename:
        abr = {'Pressure': 'P', 'Wind_speed': 'Wnd_s', 
               'Air_temperature':'Tmp', 'Wind_direction':'Wnd_dir', 
               'Cloud_amount': 'Clouds'}
        df = df.rename(abr, axis='columns')
    return df


def build_dataframe(locations, features, features_d, target, steps):
    dfs = []
    for location in locations:
        df_tmp = load_data(location + '.xlsx')
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    df = add_difference(df, features_d)

    df = df[features_d + features]
    df = add_prediction_column(df, target, steps)
    return df.dropna(axis=0)
    
def add_difference(df, features, step=1):
    for feature in features:
        df[feature + '_d'] = df[feature].diff(step)
    return df

def add_prediction_column(df, feature, steps):
    for step in steps:
        df[feature + '_diff{}'.format(str(step))] = df[feature].diff(step).shift(-step)
        df['target'] = df[feature].shift(-step)
    return df

def plot_predictions(predictions, true_values, start,end, title):
    plt.figure()
    predictions[start:end].plot(label='Predictions')
    true_values[start:end].plot(label='True Values')
    plt.legend()
    plt.title(title)


def make_train_test(df, features, predictions, sequence_len, tr_size):
    """Splits data into train and test."""
    num_features = len(features)
    num_predictions = len(predictions)
    columns = features + predictions
    df = df[columns].values
    result = []
    print('Number of features: {0}\nHorizon length: {1}'.format(num_features, sequence_len))
    for index in range(len(df) - sequence_len + 1):
        result.append(df[index: index + sequence_len])
    result = np.array(result)
    print('Shape of data: {}'.format(result.shape))
    n = int(tr_size * result.shape[0])
    X_test,y_test = None, None
    X_train = result[:n, :, :-num_predictions]
    y_train = result[:n, -1, -num_predictions:]
    if tr_size < 1:
        X_test = result[n:, :, :-num_predictions]
        y_test = result[n:, -1, -num_predictions:]
        print('X_train: {0}\ny_train: {1}\nX_test: {2}\ny_test: {3}'.format(X_train.shape, y_train.shape,
                                                                       X_test.shape, y_test.shape))
    return X_train, y_train, X_test, y_test, n

def build_model(layers):
    model = Sequential()
    model.add(LSTM(
            layers[1],
            input_shape=(None, layers[0]),
            return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[3], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(layers[4], activation="linear"))
    model.compile(loss="mae", optimizer='rmsprop')
    return model

def train_model(model, X_train, y_train, epochs=20, batch_size=64):
    print(model.summary())
    history = model.fit(
                X_train, y_train, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_split=0.1,
                shuffle = True,
                callbacks = [keras.callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=0, 
                                                            save_best_only=True, save_weights_only=False, 
                                                            mode='min', period=1
                                                            ),
                            keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, 
                                                          patience=6, verbose=1, mode='min', 
                                                          )
                            ]
                )
    return history

def evaluate_predictions(predictions, true_values):
    mean_abs_error = sklearn.metrics.mean_absolute_error(true_values, predictions)
    mean_sqrd_error = sklearn.metrics.mean_squared_error(true_values, predictions)

    print('Mean absolute error: {}'.format(mean_abs_error))
    print('Mean squared error: {}'.format(mean_sqrd_error))
    

def run_training(data, model, features, target_d, look_back, train_size, epochs=10):
    X_train, y_train, X_test, y_test, split_point = make_train_test(data, features, target_d, look_back, train_size)
    
    true_values = data['target'][split_point + look_back - 1:] #Future values we want to predict
    orig_values = data[target][split_point + look_back - 1:]   #Values at t=0

    history = train_model(model, X_train, y_train, epochs=epochs)
    predicted_difference = model.predict(X_test).flatten() # Predicted differences
    predictions = orig_values + predicted_difference
    
    print('*'*50, 'Test errors model', sep='\n')
    evaluate_predictions(predictions, true_values)
    print('*'*50, 'Test errors naive model', sep='\n')
    evaluate_predictions(orig_values, true_values)
    return history, predictions, true_values, predicted_difference

def run_validation(df, features, target_d, look_back, tr_size=1, model=None):
    """Tests a model on unseen data."""
    reshaped_data = make_train_test(df, features, target_d, look_back, tr_size)
    X_data, y_data = reshaped_data[0], reshaped_data[1]

    if model is None:
        model = keras.models.load_model('model.HDF5')

    true_values = df['target'][look_back - 1:]
    orig_values = df[target][look_back - 1:]

    predicted_difference = model.predict(X_data).flatten()
    predictions = orig_values + predicted_difference 
    print('*'*30, 'Errors model, unseen data', sep='\n')
    evaluate_predictions(predictions, true_values)
    print('*'*30, 'Errors naive model, unseen data', sep='\n')
    evaluate_predictions(orig_values, true_values)
    
    return predictions, true_values, predicted_difference
    
train_location = ['kuopio-maaninka_2016']#, 'kasko_2016']
val_location = ['kuopio-maaninka_2017']
epochs = 20
train_size = 0.80
features_d = ['Wnd_s', 'Tmp']       # features to differentiate
features = ['Wnd_s_d', 'Tmp_d']     # features to use in model
look_back = 6                       # 
target = 'Wnd_s'                    # feature to predict
horizons = [6]                      # how far in the future to predict
target_d = [target + '_diff' + str(h) for h in horizons] # 
model_path = 'model.hdf5'
model_layers=[len(features), 40, 30, 40, len(horizons)] 
plt_start = 200 # plots all timestamps if set to 'None'
plt_end = 400

data_train = build_dataframe(train_location, features, features_d, target, horizons)
print(data_train.describe())
print(data_train.head())

model = build_model(model_layers)

history_train, predictions_test, true_values_test, pred_diff_train = run_training(data_train, model, features, target_d, 
                                                                   look_back, train_size, epochs)
plot_predictions(predictions_test, true_values_test, plt_start,plt_end, 'Training '+train_location[0])

data_val = build_dataframe(val_location, features, features_d, target, horizons)
predictions_val, true_values_val, pred_diff_val = run_validation(data_val, features, target_d, look_back, tr_size=1, model=model)

plot_predictions(predictions_val, true_values_val, plt_start,plt_end, 'Validation unseen data')
plt.show()


#model.save('model.HDF5')
#model = keras.models.load_model('model.HDF5')