import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import os
import sys
import re

DATA_DIR = "data"

def load_data(name, year=2017, datetime=True):
    """Takes in name and year and returns a Pandas dataframe. 
        If datetime is True, dates are parsed into datetime64 format and set as index."""
    file_name = "{name}{year}.csv".format(name=name, year=year)
    path = os.path.join(DATA_DIR, file_name)
    
    df = pd.read_csv(path)
    if datetime:
        df = make_datetime(df)

    df = df.rename(lambda x: prettify(x), axis='columns')
    
    return df

def make_datetime(df):
    df.rename({'d': 'Day', 'm':'Month'}, axis='columns', inplace=True)
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.Time = pd.to_timedelta(df.Time + ':00', unit='h') #  + ':00' = adding the seconds
    df.Date = df.Date + df.Time
    df.index = df.Date
    df.drop(['Time', 'Year', 'Month', 'Day', 'Date'], axis=1, inplace=True)
    return df
    
def prettify(aString):
    return re.sub('\(.*\)', '', aString).strip().replace(' ', '_')


def plot_data(name, y='Air_temperature', year=2017, time= 12):
    plt.figure()
    
    df = load_data(name, year = year)
    df = df[df.index.hour == time]
    
    plt.plot(df.index, df[y], label=name.title())
    
    plt.title(y)
    plt.grid(True)
    plt.legend()
    

if __name__ == '__main__':

    df = load_data('kasko')
    df2 = load_data('kasko', datetime=False)
