import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

store_df = pd.read_csv('Rossmann/store_rossmann.csv')
ross_df = pd.read_csv('Rossmann/train_rossmann.csv', low_memory=False)
test_df = pd.read_csv('Rossmann/test_rossmann.csv')
submission_df = pd.read_csv('Rossmann/sample_submission_rossmann.csv')

merged_df = ross_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df,how='left', on='Store')

def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week

split_date(merged_df)
split_date(merged_test_df)

merged_df = merged_df[merged_df.Open == 1].copy()

def comp_month(df):
    df['CompetitionOpen'] = 12* (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df. CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda x: 0 if x<0 else x).fillna(0)

comp_month(merged_df)
comp_month(merged_test_df)



def check_promo_month(row):
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    try:
        months = (row['PrdmoInterval'] or '').split(',')
        if row['Promo20pen'] and month2str[row['Month']] in months:
            return 1
        else:
            return 0
    except Exception:
        return 0

def promo_cols(df):
# Months since Promo2 was open
    df['Promo20pen']=12 * (df.Year -df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek)*4
    df['Promo20pen']=df['Promo20pen'].fillna(0).map(lambda x: 0 if x <0 else x).fillna(0)*df['Promo2']
# Whether a new round of promotions was started in the current month
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1) * df['Promo2']



# print(merged_df.head(5).WeekOfYear - merged_df.head(5).Promo2SinceWeek)
# print(merged_df.head(5)[['Promo20pen','WeekOfYear','Promo2SinceWeek', 'Promo2']])
# check_promo_month(merged_df)
# check_promo_month(merged_test_df)


promo_cols(merged_df)
promo_cols(merged_test_df)


input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen',
    'Day', 'Month', 'Year', 'WeekOfYear', 'Promo2',
    'Promo20pen', 'IsPromo2Month']
target_col = 'Sales'

inputs = merged_df[input_cols].copy()
targets = merged_df[target_col].copy()

test_inputs = merged_test_df[input_cols].copy()

numeric_cols = ['Store', 'Promo', 'SchoolHoliday',
    'CompetitionDistance', 'CompetitionOpen', 'Promo2', 'Promo20pen', 'IsPromo2Month',
    'Day', 'Month', 'Year', 'WeekOfYear', ]
cat_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']

max_distance = inputs.CompetitionDistance.max()

inputs['CompetitionDistance'].fillna(max_distance*2, inplace=True)
test_inputs['CompetitionDistance'].fillna(max_distance*2, inplace=True)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
scaler = MinMaxScaler().fit(inputs[numeric_cols])
inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(inputs[cat_cols])
encoded_cols = list(encoder.get_feature_names_out(cat_cols))

inputs[encoded_cols] = encoder.transform(inputs[cat_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[cat_cols])

X = inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

print(X.head(5))
print(targets.sum())


from xgboost import XGBRegressor

model = XGBRegressor(random_state=42, n_jobs = -1, n_estimators = 20, max_depth = 4)

model.fit(X, targets)

preds = model.predict(X)

print(preds.sum())

from sklearn.metrics import mean_squared_error

def rmse(a,b):
    return mean_squared_error(a,b, squared=False)

print(rmse(preds, targets)) 


from xgboost import plot_tree
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt

#plot_tree(model, rankdir='LR', num_trees=1)
trees = model.get_booster().get_dump()

importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print(importance_df.head(10))

import seaborn as sns
plt.figure()
sns.barplot(data = importance_df.head(10), x='importance', y='feature')

from sklearn.model_selection import KFold

def train_and_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = XGBRegressor(random_state = 42, n_jobs = -1, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse

kfold = KFold(n_splits=5, shuffle=True)

models = []

for train_idx, val_idx in kfold.split(X):
    X_train, train_targets = X.iloc[train_idx], targets.iloc[train_idx]
    X_val, val_targets = X.iloc[val_idx], targets.iloc[val_idx]
    model, train_rmse, val_rmse = train_and_evaluate(X_train, train_targets, X_val, val_targets, max_depth = 4, n_estimators = 20)
    models.append(model)
    print('TrainRMSE:{}, ValiRMSE{}'.format(train_rmse, val_rmse))


def predict_avg(models,inputs):
    return np.mean([model.predict(inputs) for model in models], axis = 0)

preds = predict_avg(models, X)


    
