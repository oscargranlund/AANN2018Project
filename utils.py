import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import os
import sys
import re

DATA_DIR = "data"

def load_data(path, rename_abr=None, dropna=True):
    df = pd.read_excel(path, index_col='Date')
    
    if rename_abr is not None:
        df = df.rename(rename_abr, axis='columns')
    
    return df

def add_shifted_features(df, features, steps, forecast_step=1, dropnan=True):
    df_new = df[features].copy()
    for feat in features:
        for i in steps:
            newCol = feat + '(t-{})'.format(str(i))
            df_new[newCol] = df[feat].shift(i)
            
    if dropnan:
        df_new.dropna(axis=0, inplace=True)
    
    return df_new

def load_table(name, year=2008):
    filename = '_'.join([name, str(year)]) + '.xlsx'
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_excel(path)
    df.set_index('Date', inplace=True)
    return df


def plot_data(name, y='Air_temperature', year=2017, hour= 12, start_time='2017-01-01'):
    plt.figure()
    
    df = load_table(name, year = year)
    df = df.loc[df.index.hour == hour][start_time:]
    
    plt.plot(df.index, df[y], label=name.title())
    
    plt.title(y)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_data('korsnas')