import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import os
import sys
import re

DATA_DIR = "data"

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