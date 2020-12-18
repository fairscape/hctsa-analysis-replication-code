from minio import Minio
import pickle
from io import BytesIO


minioClient = Minio('minio:9000',
                    access_key='access_key',
                    secret_key='secret_key',
                    secure=False)

with open("all_ids.pkl", "rb") as fp:
    both = pickle.load(fp)

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
for patient in both[:1]:
    print(patient)
    try:
        file = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_HR.csv'
        file2 = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_HR3.csv'
        df = pd.read_csv(minioClient.get_object('prevent', file))
        df3 = pd.read_csv(minioClient.get_object('prevent', file2)).drop(['time'],axis = 1)
        hr = pd.concat([df.reset_index(drop=True), df3], axis=1)
        hr = hr.add_prefix('HR_')
        hr = hr.rename(columns={'HR_time': 'time'})
    except:
        print('Patient ' + str(patient) + ' failed to load HR')
        continue

    try:
        file = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_SPO2-%.csv'
        file2 = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_SPO2-%3.csv'
        df = pd.read_csv(minioClient.get_object('prevent', file))
        df3 = pd.read_csv(minioClient.get_object('prevent', file2)).drop(['time'],axis = 1)
        sp = pd.concat([df.reset_index(drop=True), df3], axis=1)
        sp = sp.add_prefix('SP_')
        sp = sp.rename(columns={'SP_time': 'time'})
    except:
        print('Patient ' + str(patient) + ' failed to load SP')
        continue
    full = pd.merge(hr, sp, on=['time'])
    full = full[(full['HR_Observations'] == 300) & (full['SP_Observations'] == 300)]
    full['day'] = np.ceil(full['time'] / 60 / 60 / 24)
    full = full.groupby('day').apply(lambda df: df.sample(1))
    full['id'] = patient

total = full.copy()

import time
for patient in both[1:]:
    start = time.time()
    if str(patient) == '7191':
        continue
    print(patient)
    try:
        file = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_HR.csv'
        file2 = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_HR3.csv'
        df = pd.read_csv(minioClient.get_object('prevent', file))
        df3 = pd.read_csv(minioClient.get_object('prevent', file2)).drop(['time'],axis = 1)
        hr = pd.concat([df.reset_index(drop=True), df3], axis=1)
        hr = hr.add_prefix('HR_')
        hr = hr.rename(columns={'HR_time': 'time'})
    except:
        print('Patient ' + str(patient) + ' failed to load HR')
        continue

    try:
        file = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_SPO2-%.csv'
        file2 = 'Non-PreVent-hctsa/UVA_' + str(patient) + '/UVA_' + str(patient) + '_SPO2-%3.csv'
        df = pd.read_csv(minioClient.get_object('prevent', file))
        df3 = pd.read_csv(minioClient.get_object('prevent', file2)).drop(['time'],axis = 1)
        sp = pd.concat([df.reset_index(drop=True), df3], axis=1)
        sp = sp.add_prefix('SP_')
        sp = sp.rename(columns={'SP_time': 'time'})
    except:
        print('Patient ' + str(patient) + ' failed to load SP')
        continue
    full = pd.merge(hr, sp, on=['time'])
    full = full[(full['HR_Observations'] == 300) & (full['SP_Observations'] == 300)]
    full['day'] = np.ceil(full['time'] / 60 / 60 / 24)
    full = full.groupby('day').apply(lambda df: df.sample(1))
    full['id'] = patient
    total = pd.concat([total,full])
    print('Took: ' + str(time.time() - start))

csv_bytes = total.to_csv(index = False).encode('utf-8')
csv_buffer = BytesIO(csv_bytes)


minioClient.put_object('breakfast',
                       'randomDailySample.csv',
                        data=csv_buffer,
                        length=len(csv_bytes),
                        content_type='application/csv')
