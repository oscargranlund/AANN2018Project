import pandas as pd 
import numpy as np 
import re 
import os 
import sys 
SOURCE = 'rawdata'
DATA_OUT = 'data'

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

def clean_data(file_name):
    if file_name not in os.listdir(SOURCE):
        print('No such file')
        return
    loc_name = file_name.split('.')[0]
    #file_name, file_type = file.split('.')
    df = pd.read_csv(os.path.join(SOURCE, file_name))
    df = make_datetime(df)
    df = df.rename(lambda x: prettify(x), axis='columns')
    df.to_excel(os.path.join(DATA_OUT, loc_name + '.xlsx'))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        name = sys.argv[-1]
        clean_data(name)
