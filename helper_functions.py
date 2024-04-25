import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

'''next two lines just for no error warning'''
model, imputer, scaler, encoder = 0,0,0,0
numeric_cols, cat_cols, encoded_cols = [],[],[]


def plot_df(df, target_column):
    # print(df.columns)
    # cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    for i in cols:
        fig, ax = plt.subplots()
        sns.histplot(df[i])
        fig1, ax1 = plt.subplots()
        sns.scatterplot(df, x=df[i], y=df[target_column])
        plt.show()

# plot_df(df, 'charges')

def plot_raw_distributions(raw_df, X='', Y='', Color='', hover='',scatter = True, Title=''):
    sns.set_style('darkgrid')
    matplotlib.rcParams['font.size']=14
    matplotlib.rcParams['figure.figsize'] = (10,6)
    matplotlib.rcParams['figure.facecolor'] = '#00000000'
    fig1 = px.histogram(raw_df,x=X, color=Color, color_discrete_sequence=['green','grey'], marginal='box', title=Title)
    fig1.update_layout(bargap=0.1)
    fig1.show()
    if scatter:
        fig2 = px.scatter(raw_df, x=X, y=Y, color=Color, opacity = 0.8, hover_data=[hover], title = Title)
        fig2.update_traces(marker_size=5)
        fig2.show()

def heatmap(df):
    sns.heatmap(df.corr(), cmap='Reds', annot = 'True')
    

def feature_weights(weight_df):
    '''weight_df = pd.DataFrame({'feature': (numeric_cols + encoded_cols),
              'weight': model.coef_.tolist()[0]
              })'''
    plt.figure(figsize=(5,50))
    sns.barplot(data= weight_df, x='weight', y='feature')
    plt.show()

def predict_and_plot(inputs, targets,name=''):
    '''model has to be defined beforehand'''
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

def random_choice(data):
    return np.random.choice(['No', 'Yes'], len(data))

def all_no(data):
    return np.full(len(data), 'No')

def predict_input(single_data):
    ''' numeric, cat and encoded cols have to be defined;
      imputer, scaler, encoder and model have to be fitted before'''
    input_df = pd.DataFrame([single_data])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[cat_cols])
    X_input = input_df[numeric_cols+ encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets-predictions)))