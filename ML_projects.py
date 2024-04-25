import numpy as numpy
import pandas as pd
import random

# read raw data

#choose columns to read
selected_cols = 'a,b'.split(',')

#all samples or just a few? --> select randomly 1% of samples
sample_fraction = 0.01
random.seed(42)
def skip_row(row_idx):
    if row_idx == 0:
        return False
    return random.random() > sample_fraction
#define dtypes for less RAM use
dtypes = {
        'a': 'float32',
        'b': 'float32',
        'c': 'float32',
        'd': 'float32',
        'e': 'float32',
        'f': 'uint8'
}


df = pd.read_csv('Taxi Fare Prediction/train.csv',
            usecols=selected_cols,
            parse_dates=['date'],
            dtype=dtypes,
            skiprows=skip_row)


