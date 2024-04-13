import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

store_df = pd.read_csv('store_rossmann.csv')
ross_df = pd.read_csv('train_rossmann.csv', low_memory=False)

merged_df = ross_df.merge(store_df, how='left', on='Store')

test_df = pd.read_csv('test_rossmann.csv')
merged_test_df = test_df.merge(store_df,how='left', on='Store')
print(merged_df.info())
print(merged_df.describe())
print(merged_df.duplicated().sum())

merged_df['Date'] = pd.to_datetime(merged_df.Date)
merged_test_df['Date'] = pd.to_datetime(merged_test_df.Date)

sns.histplot(data = merged_df, x='Sales')
print(merged_df.Open.value_counts())
merged_df = merged_df[merged_df.Open == 0].copy()

merged_df['Day'] = merged_df.Date.dt.day
merged_df['Month'] = merged_df.Date.dt.month
merged_df['Year'] = merged_df.Date.dt.year

merged_test_df['Day'] = merged_test_df.Date.dt.day
merged_test_df['Month'] = merged_test_df.Date.dt.month
merged_test_df['Year'] = merged_test_df.Date.dt.year

train_size = int(0.75*len(merged_df))
sorted_df = merged_df.sort_values('Date')

train_df, vali_df = sorted_df[:train_size], sorted_df[train_size:]

input_cols = ['Store','DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment', 'Day', 'Month', 'Year']
target_cols = 'Sales'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()

vali_inputs = vali_df[input_cols].copy()
vali_targets = vali_df[target_cols].copy()

test_inputs = merged_test_df[input_cols].copy()

numeric_cols = ['Store', 'Day', 'Month', 'Year']
cat_cols = ['DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
vali_inputs[numeric_cols] = imputer.transform(vali_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
vali_inputs[numeric_cols] = scaler.transform(vali_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown= 'ignore').fit(train_inputs[cat_cols])
encoded_cols = list(encoder.get_feature_names_out(cat_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[cat_cols])
vali_inputs[encoded_cols] = encoder.transform(vali_inputs[cat_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[cat_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_vali = vali_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

from sklearn.metrics import mean_squared_error

def try_model(model, X_train,y_train,X_vali,y_vali):
    model.fit(X_train,y_train)

    train_preds = model.predict(X_train)
    vali_preds = model.predcit(X_vali)

    train_rmse = mean_squared_error(y_train, train_preds, squared=False)
    vali_rmse = mean_squared_error(y_vali, vali_preds, squared=False)

    return train_rmse, vali_rmse


