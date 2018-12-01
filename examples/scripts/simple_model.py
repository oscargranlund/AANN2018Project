import pandas as pd
import keras as kr
import numpy as np
import matplotlib.pyplot as plt
from utils import load_table
from sklearn.model_selection import train_test_split 

train_size = 0.80
start = 3
end = 5
step = 1
forecast_step = 1
neighbor = 'kasko'
target_area = 'korsnas'
target_feat = 'Air_temperature'
features = ['Air_temperature', 'Wind_speed']
n_epochs = 75
n_preds = 10
size = 500

def add_shifted_features(df, features, start=1, end=5, step=1, forecast_step=1):

    for feat in features:
        for i in range(start, end, step):
            newCol = feat + str(i)
            df[newCol] = df[feat].shift(i)
            
    #df['Wind_Target'] = df['Wind'].shift(-1 * forecast_step)

    df.dropna(axis=0, inplace=True)
    
    return df.drop(features, axis=1), df[features] 

# Lets the last 'n' samples be test data
def split_data(X_data, y_data, n=20):
    i = X_data.index.intersection(y_data.index)
    X_data = X_data[X_data.index == i]
    y_data = y_data[i]
    return X_data.iloc[:-n], y_data.iloc[:-n], X_data.iloc[-n:], y_data.iloc[-n:]

def split_data_random(df):
    train, test = train_test_split(df, train_size=train_size, random_state=2018)
    return train.iloc[:,:-1], train.iloc[:, -1], test.iloc[:,:-1], test.iloc[:, -1]

def get_model(input_dim, hid_nodes=16):
    #X_train, y_train, X_test, y_test = split_data(df)

    model = kr.models.Sequential()
    #model.add(kr.layers.Dense(hid_nodes, input_dim=2 * (end - start + 1), activation='relu'))
    model.add(kr.layers.Dense(hid_nodes, input_dim=input_dim, activation='relu'))
    
    model.add(kr.layers.Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model


# Kumpula
df_n = load_table(neighbor, 2017)[features]
X_n, y_n = add_shifted_features(df_n, features, start=start, end=end, step=step, forecast_step=forecast_step)
#df_n = df_n.drop(features, axis=1)

df_t = load_table(target_area, 2017)[features]
X_t, y_t = add_shifted_features(df_t, features, start=start, end=end, step=step, forecast_step=forecast_step)
y = y_t[target_feat]
#df_t = df_t.drop(features, axis=1)

X = pd.merge(X_t, X_n, on='Date', suffixes=['_'+neighbor, '_'+target_area])

#X_data = df_All
X_train, y_train, X_test, y_test = split_data(X, y, n=n_preds)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



model = get_model(input_dim=X_train.shape[1], hid_nodes=32)

history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=64, shuffle=True)

costs = history.history['mean_squared_error']
scores = model.evaluate(X_test, y_test)

predictions = model.predict(X_test)

predictions = pd.Series(predictions.flatten(), index=y_test.index)

ax = y_test.plot(label='True values')
predictions.plot(label='Predictions', ax=ax)
ax.legend()
plt.show()
    
plt.figure()
plt.plot(costs)
plt.title('Costs')
plt.show()

#model.save('super_model.h5')
#model= kr.models.load_model('super_model.h5')