import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_parquet('Taxi Fare Prediction/train.parquet')
val_df = pd.read_parquet('Taxi Fare Prediction/val.parquet')
test_df = pd.read_parquet('Taxi Fare Prediction/test.parquet')

# print(train_df.columns)

input_cols = ['pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
        'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day',
        'pickup_datetime_weekday', 'pickup_datetime_hour', 'trip_distance',
        'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
        'met_drop_distance', 'wtc_drop_distance']

target_col = 'fare_amount'

train_inputs = train_df[input_cols]
train_targets = train_df[target_col]

val_inputs = val_df[input_cols]
val_targets = val_df[target_col]

test_inputs = test_df[input_cols]


from sklearn.metrics import mean_squared_error

def evaluate(model):
    train_preds = model.predict(train_inputs)
    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_preds = model.predict(val_inputs)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    return train_rmse, val_rmse, train_preds, val_preds

from sklearn.linear_model import Ridge

model1 = Ridge(random_state=42, alpha=0.9)
model1.fit(train_inputs, train_targets)

print(evaluate(model1))

# from sklearn.ensemble import RandomForestRegressor

# model2 = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth = 10, n_estimators=100)
# model2.fit(train_inputs, train_targets)

# print(evaluate(model2))


from xgboost import XGBRegressor

model3 = XGBRegressor(max_depth = 5, objective ='reg:squarederror', n_estimators = 200, random_state = 42, n_jobs = -1)

model3.fit(train_inputs, train_targets)
print(evaluate(model3))

def test_params(ModelClass, ** params):
    """Trains a model with the given parameters and returns training & validation RMSE"""
    model=ModelClass( ** params).fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(val_inputs), val_targets, squared=False)
    return train_rmse, val_rmse

def test_param_and_plot(ModelClass, param_name, param_values, ** other_params):
    """Trains multiple models by varying the value of param_name according to param_values"""
    train_errors, val_errors = [], []
    for value in param_values:
        params = dict(other_params)
        params[param_name] = value
        train_rmse, val_rmse = test_params(ModelClass, ** params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)


    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])

best_params = {
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'reg:squarederror',
    'learning_rate': 0.08
}


test_param_and_plot(XGBRegressor, 'max_depth', [3, 5, 7], ** best_params)

plt.show()