import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import os
import sys
import re
import sklearn.metrics
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from utils import load_data

# Data source: https://en.ilmatieteenlaitos.fi/download-observations#!/

def build_dataframe(locations, features, features_d, target, steps):
    dfs = []
    for location in locations:
        df_tmp = load_data(location + '.xlsx')
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    df = add_difference(df, features_d)

    df = df[features_d + features]
    df = add_prediction_column(df, target, steps)
    df.dropna(axis=0, inplace=True)
    return df 

def add_difference(df, features, step=1):
    for feature in features:
        df[feature + '_d'] = df[feature].diff(step)

    return df

def add_prediction_column(df, feature, steps):
    for step in steps:
        df[feature + '_diff{}'.format(str(step))] = df[feature].diff(step).shift(-step)
        #df[feature + '_{}'.format(str(step))] = df[feature].shift(-step)
        df['target'] = df[feature].shift(-step)
    return df


def make_train_test(df, features, predictions, sequence_len, tr_size):
    num_features = len(features)
    num_predictions = len(predictions)
    columns = features + predictions
    df = df[columns].values
    result = []
    print('Number of features: {0}\nNumber of horizons: {1}'.format(num_features, num_predictions))
    for index in range(len(df) - sequence_len + 1):
        result.append(df[index: index + sequence_len])
    result = np.array(result)
    print('Shape of data: {}'.format(result.shape))

    n = int(tr_size * result.shape[0])
    #train = result[:n]
    X_train = result[:n, :, :-num_predictions]
    y_train = result[:n, -1, -num_predictions:]
    X_test = result[n:, :, :-num_predictions]
    y_test = result[n:, -1, -num_predictions:]
    print('X_train: {0}\ny_train: {1}\nX_test: {2}\ny_test: {3}'.format(X_train.shape, y_train.shape,
                                                                       X_test.shape, y_test.shape))
    #X_train = np.reshape(X_train, ())
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
    model.compile(loss="mse", optimizer='rmsprop')
    return model

def train_model(model, X_train, y_train, epochs=20, batch_size=64):
    print(model.summary())
    history = model.fit(
                X_train, y_train, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_split=0.15,
                shuffle = True,
                callbacks = [keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0, 
                                                            save_best_only=True, save_weights_only=False, 
                                                            mode='min', period=1
                                                            ),
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                          patience=6, verbose=1, mode='min', 
                                                          )
                            ]
                )
    return history

def evaluate_predictions(predictions, test_values):
    mean_abs_error = sklearn.metrics.mean_absolute_error(test_values, predictions)
    mean_sqrd_error = sklearn.metrics.mean_squared_error(test_values, predictions)

    print('Mean absolute error: {}'.format(mean_abs_error))
    print('Mean squared error: {}'.format(mean_sqrd_error))
    
def run_training():
    df = build_dataframe(train_location, features, features_d, target, horizons)
    print(df.describe())
    print(df.head())
    X_train, y_train, X_test, y_test, split_point = make_train_test(df, features, target_d, look_back, train_size)
    test_values = df['target'][split_point + look_back - 1:] #Future values we want to predict
    orig_values = df[target][split_point + look_back - 1:]   #Values at t=0

    #model_layers=[X_train.shape[2], 20, 15, 20, y_train.shape[1]]
    model_layers=[X_train.shape[2], 40, 30, 40, y_train.shape[1]]
    model = build_model(model_layers)

    history = train_model(model, X_train, y_train, epochs=epochs)

    pred_diff = model.predict(X_test).flatten() # Predicted differences
    predictions = orig_values + pred_diff
    print('*' * 40)
    print('Test errors model')
    evaluate_predictions(predictions, test_values)
    #plot_predictions(predictions, test_values, 'Training ' + target + ' ' + train_location[0])

    print('Test errors naive model')
    evaluate_predictions(orig_values, test_values)

    return model, history, predictions, pred_diff, test_values, df

def plot_predictions(predictions, true_values, title):
    plt.figure()
    predictions.plot(label='Predictions')
    true_values.plot(label='True Values')
    plt.legend()
    plt.title(title)

def run_validation(model=None):
    """Tests a model on unseen data."""
    df = build_dataframe(val_location, features, features_d, target, horizons)
    X, y = make_samples(df, features, target_d, look_back)
    
    if model is None:
        model = keras.models.load_model('my_model.HDF5')

    true_values = df['target'][look_back - 1:]
    orig_values = df[target][look_back - 1:]

    diff_preds = model.predict(X).flatten()
    predictions = orig_values + diff_preds 
    print('Errors model, unseen data')
    evaluate_predictions(predictions, true_values)
    #plot_predictions(predictions, true_values, 'Validation ' + target + ' ' + val_location[0])
    print('Errors naive model, unseen data')
    evaluate_predictions(orig_values, true_values)
    print('*' * 40)
    #return predictions, true_values, diff_preds

def make_samples(df, features, predictions, sequence_len):
    num_features = len(features)
    num_predictions = len(predictions)
    columns = features + predictions
    df = df[columns].values
    result = []
    for index in range(len(df) - sequence_len + 1):
        result.append(df[index: index + sequence_len])

    result = np.array(result)
    X = result[:, :, :-num_predictions]
    y = result[:, -1, -num_predictions:]
    return X, y

    
train_location = ['kasko_2016', 'kasko_2017']
val_location = ['kasko_2015']
epochs = 25 
train_size = 0.80
features_d = ['P','Wnd_s', 'Tmp']      # features to differentiate
features = ['P_d', 'Wnd_s_d', 'Tmp_d']  # features to use in modelling
#features = ['P', 'Wnd_s', 'Tmp']
look_back = 48               # 
target = 'Tmp'               # feature to predict
horizons = [3]               # prediction horizon
target_d = [target + '_diff' + str(h) for h in horizons] # 
best_model_path = 'best_model.hdf5'
model, history, predictions_train, pred_diff, true_values_train, data = run_training()
#model.save('my_model.HDF5')
#model = keras.models.load_model('my_model.HDF5')
run_validation(model)

best_model = keras.models.load_model(best_model_path)

print('Best Model Evaluation')
run_validation(best_model)