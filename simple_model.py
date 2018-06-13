import pandas as pd
import keras as kr
import numpy as np
import matplotlib.pyplot as plt
from utils import load_table
from sklearn.model_selection import train_test_split 

train_size = 0.80
start = 1
end = 5
step = 1
forecast_step = 1

n_epochs = 75
n_preds = 40

def add_features(df, start=1, end=5, step=1, forecast_step=1):

    for i in range(start, end, step):
        newColumnTemp = 'Temp-' + str(i)
        newColumnWind = 'Wind-' + str(i)
        df[newColumnTemp] = df.Temp.shift(i)
        df[newColumnWind] = df.Wind.shift(i)
    
    df['Wind_Target'] = df['Wind'].shift(-1 * forecast_step)

    df.dropna(axis=0, inplace=True)
    
    return df

# Lets the last 'n' samples be test data
def split_data2(df, n=20):
    train, test = df.iloc[:-n, :], df.iloc[-n:, :]
    return train.iloc[:,:-1], train.iloc[:, -1], test.iloc[:,:-1], test.iloc[:, -1]

def split_data(df):
    train, test = train_test_split(df, train_size=train_size, random_state=2018)
    return train.iloc[:,:-1], train.iloc[:, -1], test.iloc[:,:-1], test.iloc[:, -1]

def get_model(hid_nodes=16):
    #X_train, y_train, X_test, y_test = split_data(df)

    model = kr.models.Sequential()
    model.add(kr.layers.Dense(hid_nodes, input_dim=2 * (end - start + 1), activation='relu'))
    model.add(kr.layers.Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model



def plot_predictions(preds, y_test):
    plt.figure()
    
    plt.plot(y_test.index, y_test, label='True Wind')
    plt.plot(y_test.index, preds, label='Predicted Wind')
    plt.xticks(rotation=45)
    plt.legend()

def plot_costs(costs):
    plt.figure()
    plt.plot(np.arange(len(costs)), costs)
    plt.title('Costs mean squared error')
    plt.xlabel('Epochs'), plt.ylabel('Cost')


# Kumpula
df_K = load_table('kumpula')
df_K = add_features(df_K, start, end, step, forecast_step)

X_train, y_train, X_test, y_test = split_data2(df_K, n_preds)

model = get_model(hid_nodes=32)

history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=64, shuffle=True)

costs = history.history['mean_squared_error']
scores = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)


print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))

plot_predictions(predictions.flatten(), y_test)
plot_costs(costs)
plt.show()

#df_S = load_table('sodankyla')
#df_S = add_features(df_S, start, end, step, forecast_step)

#X_train, y_train, X_test, y_test = split_data(df_K)

#def run(do_shuffle=True):