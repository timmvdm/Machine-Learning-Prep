import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

raw_df = pd.read_csv('C:/Users/timmv/OneDrive/Dokumente/GitHub/ML-Course/weatherAUS.csv')

raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

use_sample = False
sample_fraction = 0.1
if use_sample:
    raw_df  = raw_df.sample(frac=sample_fraction).copy()

### train, test, vali split

from sklearn.model_selection import train_test_split
# random splitting
# train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
# train_df, vali_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
# time split is better


year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year<2015]
vali_df = raw_df[year==2015]
test_df = raw_df[year>2015]


input_cols = list(train_df.columns)[1:-1] #ignore date and RainTomorrow
target_cols = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_cols].copy()

vali_inputs = vali_df[input_cols].copy()
vali_targets = vali_df[target_cols].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_cols].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
cat_cols = train_inputs.select_dtypes('object').columns.tolist()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
imputer.fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
vali_inputs[numeric_cols] = imputer.transform(vali_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
vali_inputs[numeric_cols] = scaler.transform(vali_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown= 'ignore')
encoder.fit(raw_df[cat_cols])

encoded_cols = list(encoder.get_feature_names_out(cat_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[cat_cols])
vali_inputs[encoded_cols] = encoder.transform(vali_inputs[cat_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[cat_cols])

pd.set_option('display.max_columns', None)

print(train_inputs.shape)
print(train_targets.shape)
print(vali_inputs.shape)
print(vali_targets.shape)
print(test_inputs.shape)
print(test_targets.shape)

train_inputs.to_parquet('train_inputs.parquet')
vali_inputs.to_parquet('vali_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')

pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(vali_targets).to_parquet('vali_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'liblinear')

model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)
# print(model.coef_.tolist())
weight_df = pd.DataFrame({'feature': (numeric_cols + encoded_cols),
              'weight': model.coef_.tolist()[0]
              })

import seaborn as sns
# plt.figure(figsize=(5,50))
# sns.barplot(data= weight_df, x='weight', y='feature')
# plt.show()

X_train = train_inputs[numeric_cols + encoded_cols]
X_vali = vali_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# train_pred = model.predict(X_train)
# vali_pred = model.predict(X_vali)
# test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

# print(accuracy_score(train_targets, train_pred))
# train_probs = model.predict_proba(X_train)
# classes = model.classes_
# print('probs for', classes, ':', train_probs)

from sklearn.metrics import confusion_matrix
# confusion_matrix(train_targets,train_pred, normalize=True)

def predict_and_plot(inputs, targets,name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets,preds)
    print('Accurracy: {:.2f}%'.format(accuracy*100))
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    return preds

train_pred = predict_and_plot(X_train,train_targets,'Training')
vali_pred = predict_and_plot(X_vali,vali_targets,'Validation')
test_pred = predict_and_plot(X_test,test_targets,'Test')

# plt.show()

def random_choice(data):
    return np.random.choice(['No', 'Yes'], len(data))

def all_no(data):
    return np.full(len(data), 'No')

def predict_input(single_data):
    input_df = pd.DataFrame([single_data])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[cat_cols])
    X_input = input_df[numeric_cols+ encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

new_input = {'Date': '2021-06-19',
            'Location': 'Launceston',
            'MinTemp': 23.2,
            'MaxTemp': 33.2,
            'Rainfall': 10.2,
            'Evaporation': 4.2,
            'Sunshine': np.nan,
            'WindGustDir': 'NNW',
            'WindGustSpeed': 52.0,
            'WindDir9am': 'NW',
            'WindDir3pm': 'NNE',
            'WindSpeed9am': 13.0,
            'WindSpeed3pm': 20.0,
            'Humidity9am': 89.0,
            'Humidity3pm': 58.0,
            'Pressure9am': 1004.8,
            'Pressure3pm': 1001.5,
            'Cloud9am': 8.0,
            'Cloud3pm': 5.0,
            'Temp9am': 25.7,
            'Temp3pm': 33.0,
            'RainToday': 'Yes'}

print(predict_input(new_input))