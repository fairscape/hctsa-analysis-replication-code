import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import l1_min_c
import time
import json

hr = pd.read_csv('/data/HR_daily_samples.csv')
hr = hr.add_prefix('HR_')

print('Read in HR')

sp = pd.read_csv('/data/SPO2_daily_samples.csv')
sp = sp.add_prefix('SP_')

print('Read in SP')


sp['day'] = np.ceil(sp['SP_time'] / 60 / 60 / 24)
hr['day'] = np.ceil(hr['HR_time'] / 60 / 60 / 24)
sp = sp.rename(columns={'SP_id': 'id'})
hr = hr.rename(columns={'HR_id': 'id'})

df = pd.merge(hr, sp, on=['id','day'])
mat_hr = pd.read_csv('./randomDailySample_matlab_HR.csv')
mat_hr = mat_hr.add_prefix('HR_')
mat_hr = mat_hr.rename(columns={'HR_id': 'id', 'HR_time': 'time'})
mat_sp = pd.read_csv('./randomDailySample_matlab_SP.csv')
mat_sp = mat_sp.add_prefix('SP_')
mat_sp = mat_sp.rename(columns={'SP_id': 'id', 'SP_time': 'time'})
df = pd.merge(df, mat_hr, on=['id','time'])
df = pd.merge(df, mat_sp, on=['id','time'])

#Optional: Include if interested in performance on just older babies
# grouped = df.groupby("id")
# too_short = list(grouped['day'].max().index[grouped['day'].max() < 7])
# df = df[~df.id.isin(too_short)]


df = df.replace([np.inf, -np.inf], np.nan)
df = df.select_dtypes(exclude=['object'])
df = df.loc[:,df.isnull().mean() < .035]

print('Building Cleaned Dataframe')

X = df.dropna()
ids = X['id']
time = X['time']
X = X.drop(['id','time'],axis = 1)
X = X.loc[:,X.std() != 0]

def normalize(x):
    #Normalizes column onto scale of 0-1
    return 1 / ( 1 + np.exp( - ( x - x.median() ) / ( 1.35 * (x.quantile(.75) - x.quantile(.25)) ) ) )

X = X.transform(normalize)
X = X.loc[:, X.isnull().sum() == 0]

#Drop some specific columns unfit for modelling
bad_index = []
bad_column = []
for i in range(len(list(distance.columns))):
    column = list(distance.columns)[i]
    if 'MD.pNN.pnn' in column:
        bad_index.append(i)
        bad_column.append(column)
    if 'PH.' in column and 'res.acl' in column:
        bad_index.append(i)
        bad_column.append(column)
    if 'EN.PermEn.3' in column:
        bad_index.append(i)
        bad_column.append(column)
    if 'SY.LocalGlobal' in column and 'skew' in column:
        bad_index.append(i)
        bad_column.append(column)
    if 'SY.LocalGlobal' in column and 'ac1' in column:
        bad_index.append(i)
        bad_column.append(column)
    if 'SY.LocalGlobal' in column and 'kurtosis' in column:
        bad_index.append(i)
        bad_column.append(column)
    if 'Time' in column or 'day' in column:
        bad_index.append(i)
        bad_column.append(column)

X['time'] = time
X['id'] = ids
X.to_csv('/outputs/normalized_data.csv',index = False)
