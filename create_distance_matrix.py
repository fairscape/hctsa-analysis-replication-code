from scipy.stats import chi2_contingency
import math
import numpy as np
import pandas as pd
import os

def norm_MI(x,y,xbins = [],ybins=[],bins=10):

    if xbins == []:

        c_xy = np.histogram2d(x, y, bins=10)[0]

    else:

        c_xy = np.histogram2d(x, y, bins=(xbins, ybins))[0]

    count = len(x)

    H_XY  = 0
    H_X   = 0
    H_Y   = 0
    H_XgY = 0

    for i in range(bins):

        px = sum(c_xy[i] / count)

        if px == 0:

            continue

        H_X += - px * math.log2(px)

        py = sum(c_xy[:,i] / count)

        H_Y += - py * math.log2(py)

        for j in range(bins):

            py = sum(c_xy[:,j] / count)

            if py == 0:

                continue

            if c_xy[i][j] == 0:

                continue

            H_XY  += - (c_xy[i][j] / count) * math.log2(   c_xy[i][j] / count)
            H_XgY += - (c_xy[i][j] / count) * math.log2( ( c_xy[i][j] / count) / py)

    I_XY = H_X - H_XgY

    return 1 - I_XY / H_XY

def calc_ent(c_xy,count,bins):
    H_X_Y = 0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i][j] == 0:
                continue
            H_X_Y += - (c_xy[i][j] / count) * math.log2(c_xy[i][j] / count)

    return H_X_Y

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

file = '/data/normalized_data.csv'

X =  pd.read_csv(file_name)

#Remove columns not fit for clustering
X = X.drop(['HR_Observations','SP_Observations','day','HR_time','SP_time'],axis = 1,errors='ignore')
X = X.drop(['y','id'],axis = 1,errors='ignore')

numCols = X.shape[1]

bins = []
for i in range(numCols):
    bins.append(histedges_equalN(X.iloc[:,i].values,10))


pairs = []
distance = np.zeros((numCols,numCols))
for i in range(numCols):
    print(i)
    for j in range(i,numCols):
        try:
            distance[i,j] = norm_MI(X.iloc[:,i].values,X.iloc[:,j].values,bins[i],bins[j])
        except:
            pairs.append(str(i) + ':' + str(j))
            distance[i,j] = 1

for pair in pairs:
    i = int(pair.split(':')[0])
    j = int(pair.split(':')[0])
    distance[i,j] = norm_MI(X.iloc[:,i].values,X.iloc[:,j].values,bins[i],bins[j])

for i in range(numCols):
    for j in range(i,numCols):
        distance[j][i] = distance[i][j]

col_names = X.columns
del X

distance = pd.DataFrame(distance)
distance.columns = col_names[:numCols]

distance.to_csv('/outputs/distance_between_operations_HR_SP.csv',index = False)
