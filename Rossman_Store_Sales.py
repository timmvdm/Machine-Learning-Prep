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

