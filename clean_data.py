import pandas as pd 
import numpy as np 
import re 
import os 

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

def clean_data():
    for file in os.listdir(SOURCE):
        file_name, file_type = file.split('.')
        if file_type == 'csv':
            print(file)
            df = pd.read_csv(os.path.join(SOURCE, file))
            df = make_datetime(df)
            df = df.rename(lambda x: prettify(x), axis='columns')
            df.to_excel(os.path.join(DATA_OUT, file_name + '.xlsx'))

if __name__ == '__main__':
    clean_data()
