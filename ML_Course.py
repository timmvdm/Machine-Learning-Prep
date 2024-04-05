import pandas as pd
medical_df = pd.read_csv('C:/Users/timmv/OneDrive/Dokumente/GitHub/ML-Course/Medical_Cost.csv')
# print(medical_df.info())
# print(medical_df.describe())

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# %matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# print(medical_df.age.describe())

# fig = px.histogram(medical_df,x='age', marginal='box',nbins = 47, title='Age distribution')
# fig.update_layout(bargap=0.1)
# fig.show()

# fig = px.histogram(medical_df,x='bmi', marginal='box',color_discrete_sequence=['red'], title='BMI distribution')
# fig.update_layout(bargap=0.1)
# fig.show()

# fig = px.histogram(medical_df,x='charges', color='smoker',color_discrete_sequence=['green','grey'], marginal='box', title='Annual medical charges (smoker)')
# fig.update_layout(bargap=0.1)
# fig.show()


# fig = px.histogram(medical_df,x='charges', color='sex',color_discrete_sequence=['green','grey'], marginal='box', title='Annual medical charges (sex)')
# fig.update_layout(bargap=0.1)
# fig.show()

# fig = px.histogram(medical_df,x='charges', color='region',color_discrete_sequence=['green','grey'], marginal='box', title='Annual medical charges (region)')
# fig.update_layout(bargap=0.1)
# fig.show()

# px.histogram(medical_df,x='smoker', color = 'sex', color_discrete_sequence=['red', 'blue'])
# px.histogram(medical_df,x='smoker', color = 'region', color_discrete_sequence=['red', 'blue'])
# px.histogram(medical_df,x='smoker', color = 'children', color_discrete_sequence=['red', 'blue'])
# px.histogram(medical_df,x='smoker', color = 'sex', color_discrete_sequence=['red', 'blue'])

# fig = px.scatter(medical_df, x='age', y='charges', color='smoker', opacity = 0.8, hover_data=['sex'], title = 'Age vs Charges')
# fig.update_traces(marker_size=5)
# fig.show()

# fig = px.scatter(medical_df, x='bmi', y='charges', color='smoker', opacity = 0.8, hover_data=['sex'], title = 'BMI vs Charges')
# fig.update_traces(marker_size=5)
# fig.show()

# medical_df.corr()
# smoker_values ={'no': 0, 'yes':1}
# smoker_numeric = medical_df.smoker.map(smoker_values)

# sns.heatmap(medical_df.corr(), cmap='Reds', annot = 'True')

non_smoker_df = medical_df[medical_df.smoker == 'no']
plt.title('Age vs. Charges')
sns.scatterplot(data = non_smoker_df, x = 'age', y= 'charges', alpha = 0.7 ,s=15)

def estimate_charges(age, w,b):
    return age*w+b

from sklearn.linear_model import LinearRegression
import numpy as np

# model = LinearRegression()
# inputs = non_smoker_df[['age']]
# targets = non_smoker_df.charges
# # print(non_smoker_df.age[:5],non_smoker_df.charges[:5])
# # print(non_smoker_df[['age']][:5])
# model.fit(inputs, targets)
# print(model.predict(np.array([[23],[37],[61]])))
# predictions = model.predict(inputs)
# print(predictions)

smoker_codes = {'yes':1,'no':0}
sex_codes = {'female':0, 'male':1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)
medical_df['sex_code'] = medical_df.sex.map(sex_codes)


from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
one_hot = enc.transform(medical_df[['region']]).toarray()

def rmse(t, p):
    return np.sqrt(np.mean(np.square(t-p)))

medical_df[['northeast','northwest', 'southeast', 'southwest']] = one_hot
print(medical_df.head())

input_cols = ['age', 'bmi','children', 'smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast','southwest']
inputs, targets = medical_df[input_cols],medical_df['charges']
model_ges = LinearRegression().fit(inputs,targets)
predictions = model_ges.predict(inputs)
loss = rmse(targets, predictions)
print('loss_ges = ', loss)

non_smoker_df = medical_df[medical_df.smoker == 'no']
smoker_df = medical_df[medical_df.smoker == 'yes']

inputs_non_smoker, targets_non_smoker = non_smoker_df[input_cols], non_smoker_df['charges']
inputs_smoker, targets_smoker = smoker_df[input_cols], smoker_df['charges']
model_non_smoker = LinearRegression().fit(inputs_non_smoker,targets_non_smoker)
model_smoker = LinearRegression().fit(inputs_smoker, targets_smoker)
preds_non_smoker = model_non_smoker.predict(inputs_non_smoker)
preds_smoker = model_smoker.predict(inputs_smoker)
loss_non_smoker = rmse(targets_non_smoker, preds_non_smoker)
loss_smoker = rmse(targets_smoker,preds_smoker)
print('loss_non_smoker = ', loss_non_smoker)
print('loss_smoker = ', loss_smoker)

from sklearn.preprocessing import StandardScaler

numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])
scaled_inputs = scaler.transform(medical_df[numeric_cols])
cat_cols = ['smoker_code', 'sex_code', 'northeast','northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values
# print(categorical_data)

inputs = np.concatenate((scaled_inputs,categorical_data), axis=1)
targets = medical_df.charges
model = LinearRegression().fit(inputs, targets)
pred = model.predict(inputs)
loss = rmse(targets, pred)

print(loss)


from sklearn.model_selection import train_test_split
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size =0.1)
model = LinearRegression().fit(inputs_train, targets_train)
pred_test = model.predict(inputs_test)
pred_train = model.predict(inputs_train)
loss_test = rmse(targets_test, pred_test)
loss_train = rmse(targets_train,pred_train)


print(loss_test)
print(loss_train)


