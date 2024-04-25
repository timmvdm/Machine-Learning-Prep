import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 

selected_cols = 'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')

print(selected_cols)
dtypes = {
        'fare_amount': 'float32',
        'pickup_longitude': 'float32',
        'pickup_latitude': 'float32',
        'dropoff_longitude': 'float32',
        'dropoff_latitude': 'float32',
        'passenger_count': 'uint8'
}
sample_fraction = 0.01

def skip_row(row_idx):
    if row_idx == 0:
        return False
    return random.random() > sample_fraction

random.seed(42)
df = pd.read_csv('Taxi Fare Prediction/train.csv',
            usecols=selected_cols,
            parse_dates=['pickup_datetime'],
            dtype=dtypes,
            skiprows=skip_row)

print(df.shape)

test_df = pd.read_csv('Taxi Fare Prediction/test.csv', dtype=dtypes, parse_dates=['pickup_datetime'])

from sklearn.model_selection import train_test_split
train_df , val_df = train_test_split(df, test_size=0.2, random_state=42)

print(train_df.shape, val_df.shape)

train_df = train_df.dropna()
val_df = val_df.dropna()


input_cols = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
target_cols = 'fare_amount'

train_inputs = train_df[input_cols]
train_targets = train_df[target_cols]

val_inputs = val_df[input_cols]
val_targets = val_df[target_cols]

test_inputs = test_df[input_cols]

class MeanRegressor:
    def fit(self,inputs, targets):
        self.mean = targets.mean()
    
    def predict(self,inputs):
        return np.full(inputs.shape[0], self.mean)


mean_model = MeanRegressor()

mean_model.fit(train_inputs, train_targets)
train_preds = mean_model.predict(train_inputs)
val_preds = mean_model.predict(val_inputs)


from sklearn.metrics import mean_squared_error
def rmse(targets, preds):
    return mean_squared_error(targets, preds, squared= False)

train_rmse = rmse(train_targets, train_preds)
val_rmse = rmse(val_targets, val_preds)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(train_inputs, train_targets)

train_preds = linear_model.predict(train_inputs)

rmse(train_targets, train_preds)

def add_dateparts(df, col):
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day
    df[col + '_weekday' ] = df[col].dt.weekday
    df[col +'_hour'] = df[col].dt.hour

add_dateparts(train_df,'pickup_datetime')
add_dateparts(val_df,'pickup_datetime')
add_dateparts(test_df,'pickup_datetime')

import numpy as np

def haversine_np(lon1, lat1, lon2, lat2):

    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length."""

    lon1, lat1, lon2, lat2=map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0) ** 2 +np.cos(lat1) * np.cos(lat2) *np.sin(dlon/2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def add_trip_distance(df):
    df['trip_distance'] = haversine_np(df['pickup_longitude'],
    df['pickup_latitude'],
    df['dropoff_longitude'],
    df['dropoff_latitude'])

add_trip_distance(train_df)
add_trip_distance(val_df)
add_trip_distance(test_df)


jfk_lonlat = -73.7781, 40.6413
lga_lonlat = -73.8740, 40.7769
ewr_lonlat = -74.1745, 40.6895
met_lonlat = -73.9632, 40.7794
wtc_lonlat = -74.0099, 40.7126

def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = haversine_np(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])

def add_landmarks(a_df):
    landmarks = [('jfk', jfk_lonlat), ('lga', lga_lonlat), ('ewr', ewr_lonlat), ('met', met_lonlat), ('wtc', wtc_lonlat) ]
    for name, lonlat in landmarks:
        add_landmark_dropoff_distance(a_df, name, lonlat)

add_landmarks(train_df)
add_landmarks(val_df)
add_landmarks(test_df)

def remove_outliers (df):
    return df[(df['fare_amount'] >= 1.) &
        (df['fare_amount'] <= 500.) &
        (df['pickup_longitude'] >= -75) &
        (df['pickup_longitude'] <= -72) &
        (df['dropoff_longitude'] >= -75) &
        (df['dropoff_longitude'] <= -72) &
        (df['pickup_latitude'] >= 40) &
        (df['pickup_latitude'] <= 42) &
        (df['dropoff_latitude'] >=40) &
        (df['dropoff_latitude'] <= 42) &
        (df['passenger_count'] >= 1) &
        (df['passenger_count'] <= 6)]

train_df = remove_outliers(train_df)
val_df = remove_outliers(val_df)

train_df.to_parquet('train.parquet')
val_df.to_parquet('val.parquet')
test_df.to_parquet('test.parquet')

