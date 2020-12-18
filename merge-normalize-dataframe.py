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

#Optional: Include if interested in performance on just older babies
# grouped = df.groupby("id")
# too_short = list(grouped['day'].max().index[grouped['day'].max() < 7])
# df = df[~df.id.isin(too_short)]

print('Reading in Patient Outcome Data')

patientDF = pd.read_excel('/data/UVA NICU Infants.xlsx')
patientDF['id'] = patientDF['PatientID']
patientDF = patientDF[['NICU Outcome','Outcome PMA','GestAge','GestAgeDays','id']].fillna(0)

df = pd.merge(df, patientDF, on='id')
df['PMA'] = df['GestAge'] + df['GestAgeDays'] / 7 + df['day'] / 7
df = df.replace([np.inf, -np.inf], np.nan)
df = df.select_dtypes(exclude=['object'])
df['y'] = 0
df['y'][np.logical_and(df['NICU Outcome'],(df['Outcome PMA'] - df['PMA']) <= 1)] = 1
df = df.loc[:,df.isnull().mean() < .1]

print('Building Cleaned Dataframe')

X = df.dropna()
ids = X['id']
X = X.drop(['id'],axis = 1)
Y = X['y']
X = X.loc[:,X.std() != 0]
def normalize(x):
    #Normalizes column onto scale of 0-1
    return 1 / ( 1 + np.exp( - ( x - x.median() ) / ( 1.35 * (x.quantile(.75) - x.quantile(.25)) ) ) )

X = X.transform(normalize)
X = X.loc[:, X.isnull().sum() == 0]
#Final number here will change move to however many columns you want to keep
X = X.iloc[:,:3410]
X['y'] = Y
X['id'] = ids
X.to_csv('/outputs/normalized_data.csv',index = False)
