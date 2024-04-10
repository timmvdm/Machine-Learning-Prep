import opendatasets as od
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

#Download Dataset
raw_df = pd.read_csv('weatherAUS.csv')
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)


# if dataset too big use only a sample
use_sample = False
sample_fraction = 0.1
if use_sample:
    raw_df  = raw_df.sample(frac=sample_fraction).copy()


#Sanity Check
print(raw_df.info())
print(raw_df.describe())

#Create train, vali, test datasets
year = pd.to_datetime(raw_df.Date).dt.year
train_df, vali_df, test_df  = raw_df[year<2015],raw_df[year==2015], raw_df[year>2015]
#or with test_train_split randomly
# train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
# train_df, vali_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


#Create inputs + targets
input_cols = list(train_df.columns)[1:-1] #ignore date and RainTomorrow
target_cols = 'RainTomorrow'

train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_cols].copy()
vali_inputs, vali_targets = vali_df[input_cols].copy(), vali_df[target_cols].copy()
test_inputs, test_targets = test_df[input_cols].copy(), test_df[target_cols].copy()


#Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
cat_cols = train_inputs.select_dtypes('object').columns.tolist()


#Impute missing numeric features
imputer = SimpleImputer(strategy='mean').fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
vali_inputs[numeric_cols] = imputer.transform(vali_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


#Sanity Check
print(train_inputs[numeric_cols].isna().sum())



#Scaler numeric features
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
vali_inputs[numeric_cols] = scaler.transform(vali_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


#Sanity Check
print(train_inputs.info())
print(train_inputs.describe())


#One-Hot-Encode categroical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown= 'ignore').fit(raw_df[cat_cols])
encoded_cols = list(encoder.get_feature_names_out(cat_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[cat_cols])
vali_inputs[encoded_cols] = encoder.transform(vali_inputs[cat_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[cat_cols])

 

#Sanity Check
print(train_inputs.info())
print(train_inputs.describe())


#Save preporcessed data to disk
train_inputs.to_parquet('train_inputs.parquet')
vali_inputs.to_parquet('vali_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')
pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(vali_targets).to_parquet('vali_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')
