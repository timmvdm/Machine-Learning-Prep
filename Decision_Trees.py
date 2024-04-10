import opendatasets as od
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

numeric_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 
                'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
encoded_cols = ['Location_Adelaide', 'Location_Albany', 'Location_Albury', 'Location_AliceSprings',
                 'Location_BadgerysCreek', 'Location_Ballarat', 'Location_Bendigo', 'Location_Brisbane',
                  'Location_Cairns', 'Location_Canberra', 'Location_Cobar', 'Location_CoffsHarbour',
                    'Location_Dartmoor', 'Location_Darwin', 'Location_GoldCoast', 'Location_Hobart', 
                    'Location_Katherine', 'Location_Launceston', 'Location_Melbourne', 'Location_MelbourneAirport',
                    'Location_Mildura', 'Location_Moree', 'Location_MountGambier', 'Location_MountGinini',
                    'Location_Newcastle', 'Location_Nhil', 'Location_NorahHead', 'Location_NorfolkIsland',
                    'Location_Nuriootpa', 'Location_PearceRAAF', 'Location_Penrith', 'Location_Perth', 
                    'Location_PerthAirport', 'Location_Portland', 'Location_Richmond', 'Location_Sale', 
                    'Location_SalmonGums', 'Location_Sydney', 'Location_SydneyAirport',
                    'Location_Townsville', 'Location_Tuggeranong', 'Location_Uluru', 'Location_WaggaWagga',
                    'Location_Walpole', 'Location_Watsonia', 'Location_Williamtown', 
                    'Location_Witchcliffe', 'Location_Wollongong', 'Location_Woomera', 'WindGustDir_E',
                    'WindGustDir_ENE', 'WindGustDir_ESE', 'WindGustDir_N', 'WindGustDir_NE', 
                    'WindGustDir_NNE', 'WindGustDir_NNW', 'WindGustDir_NW', 'WindGustDir_S', 
                    'WindGustDir_SE', 'WindGustDir_SSE', 'WindGustDir_SSW', 'WindGustDir_SW',
                    'WindGustDir_W', 'WindGustDir_WNW', 'WindGustDir_WSW', 'WindGustDir_nan',
                    'WindDir9am_E', 'WindDir9am_ENE', 'WindDir9am_ESE', 'WindDir9am_N',
                    'WindDir9am_NE', 'WindDir9am_NNE', 'WindDir9am_NNW', 'WindDir9am_NW', 
                    'WindDir9am_S', 'WindDir9am_SE', 'WindDir9am_SSE', 'WindDir9am_SSW',
                    'WindDir9am_SW', 'WindDir9am_W', 'WindDir9am_WNW', 'WindDir9am_WSW',
                    'WindDir9am_nan', 'WindDir3pm_E', 'WindDir3pm_ENE', 'WindDir3pm_ESE',
                    'WindDir3pm_N', 'WindDir3pm_NE', 'WindDir3pm_NNE', 'WindDir3pm_NNW',
                    'WindDir3pm_NW', 'WindDir3pm_S', 'WindDir3pm_SE', 'WindDir3pm_SSE',
                    'WindDir3pm_SSW', 'WindDir3pm_SW', 'WindDir3pm_W', 'WindDir3pm_WNW',
                    'WindDir3pm_WSW', 'WindDir3pm_nan', 'RainToday_No', 'RainToday_Yes']

X_train = pd.read_parquet('train_inputs.parquet')[numeric_cols + encoded_cols]
X_vali = pd.read_parquet('vali_inputs.parquet')[numeric_cols + encoded_cols]
X_test = pd.read_parquet('test_inputs.parquet')[numeric_cols + encoded_cols]

train_targets = pd.read_parquet('train_targets.parquet')
vali_targets = pd.read_parquet('vali_targets.parquet')
test_targets = pd.read_parquet('test_targets.parquet')


# print(X_train, train_target)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, train_targets)
'''this leads to overfiting. either set max leave nodes (how many max results) or max depth (how many levels). 
uses Gini-index (the  node with lowestgini coefficient gets split) to build tree and consider all split for all features
max_nodes and max depth is not the same because decision trees build are based gini coefficient and do not split not equally for all levels'''

def predict_and_plot(inputs, targets,name=''):
    '''model has to be defined beforehand'''
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets,preds)
    print('Accurracy: {:.6f}%'.format(accuracy*100))
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    plt.show()
    return preds

# def rmse(targets, predictions):
#     return np.sqrt(np.mean(np.square(targets-predictions)))


train_preds = predict_and_plot(X_train, train_targets)
# print(rmse(train_targets,train_preds))

from sklearn.tree import plot_tree, export_text

plt.figure(figsize = (80,20))
plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True)
plt.show()

tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
print(tree_text[:5000])
print(model.feature_importances_)
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
sns.barplot(data= importance_df.head(10), x='importance', y='feature')
