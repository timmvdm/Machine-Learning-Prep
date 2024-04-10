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

train_targets = pd.read_parquet('train_targets.parquet').squeeze()
vali_targets = pd.read_parquet('vali_targets.parquet').squeeze()
test_targets = pd.read_parquet('test_targets.parquet').squeeze()


# print(train_targets.squeeze())

from sklearn.ensemble import RandomForestClassifier

base_model = RandomForestClassifier(random_state=42).fit(X_train, train_targets)
base_train_acc = base_model.score(X_train,train_targets)
base_vali_acc = base_model.score(X_vali,vali_targets)
base_accs = base_train_acc, base_vali_acc

# enter first individual decision trees with model.estimators_[0] within forest
print(base_model.predict_proba(X_train))

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# print(importance_df.head(10))
sns.barplot(data= importance_df.head(10), x='importance', y='feature')
print('base:', base_accs)

def test_params(**params):
    model = RandomForestClassifier(random_state = 42, **params).fit(X_train,train_targets)
    return model.score(X_train, train_targets), model.score(X_vali, vali_targets)

'''params like: max_depth, max_leaf_nodes, max_features (default = sqrt(number of features))
max_depth: how many levels
max_leaf_node: how many final leave nodes because the tree gets split based on (gini) coef and not level by level
max_features: tree gets trained not on all features but on some of them and only considers the best splits
if always all features trees would not fluctuate and no ensemble advantages
min_samples_split: node(knoten) will split by default until there is only on leaf in the end; min_sample_split says that the 
node is split only if it has more than min_sample_split rows
min_samples_leaf: a leaf only gets created if it is bigger than the min_samples_leaf size
min_impurity_decrease: splits should lower the gini index by at least this number because otherwise overfitting
the gini score decreases down the tree and is between 0 and 1
--> the node should be split only if the decrease in the gini coefficient is bigger that min_impurity_decrease
bootstrap: picking rows with replacement --> same number of rows but maybe some double/tripel= training on fraction of data
--> different bootstraps for different trees --> creates randomness
max_samples: None= all samples, int= number of samples, float= percentage of all samples
class_weight: 'balanced' or with certain targets {'No':1,'Yes:2} assigned weights to the different target classes in gini index and splitting
  '''

