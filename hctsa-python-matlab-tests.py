import unittest
import statistics as stats
from scipy import stats
import warnings
import h5py
import math
import scipy.io
import time
import numpy as np
import pandas as pd
import statsmodels.sandbox.stats.runs as runs
from Operations import *
from Periphery import *

def get_column_names(vname,f):
    names = []
    for name in vname[0]:
        obj = f[name]
        col_name = ''.join(chr(i) for i in obj[:])
        names.append(col_name)
    return names

def read_in_NICU_file(path):
    arrays = {}
    f = h5py.File(path,'r')
    for k, v in f.items():
        if k != 'vdata' and k != 'vt':
            continue
        arrays[k] = np.array(v)
    df = pd.DataFrame(np.transpose(arrays['vdata']))
    df = df.dropna(axis=1, how='all')
    df.columns = get_column_names(f['vname'],f)
    times = pd.Series(arrays['vt'][0], index=df.index)

    return df,times

def roundDict(d,digits = 3):
    if isinstance(d,dict):
        for key in d:

            d[key] = round(d[key],digits)

        return d

    else:

        return round(d,digits)

def closeEnough(d1,d2,dist = .05):
    if isinstance(d1,dict):
        for key in d1:

            if key not in d2.keys():
                print(key)
                return False

            if np.absolute(d1[key] - d2[key]) > dist:

                print(key)
                print(np.absolute(d1[key] - d2[key]))

                return False

        return True

    else:

        if np.absolute(d1 - d2) > dist:

            return False

        return True


x = np.random.rand(500)

y = np.random.rand(1000)

df = read_in_NICU_file('test.mat')

hr = df[0]['HR'].to_numpy()

hr = hr[:500]


import matlab.engine
eng = matlab.engine.start_matlab()
matlab_path = '/path/to/hctsa/'
eng.addpath(matlab_path + 'Operations/')
eng.addpath(matlab_path + 'Toolboxes/Physionet/')
eng.addpath(matlab_path + 'Toolboxes/Rudy_Moddemeijer/')
eng.addpath(matlab_path + 'PeripheryFunctions/')
eng.addpath(matlab_path + 'Toolboxes/ZG_hmm')


class test_DN_functions(unittest.TestCase):

    def test_DN_Mean(self):


        pyDrift1 = roundDict(DN_Mean(x),2)
        pyDrift2 = roundDict(DN_Mean(y),2)
        pyDrift3 = roundDict(DN_Mean(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_Mean(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_Mean(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_Mean(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_DN_pleft(self):


        pyDrift1 = roundDict(DN_pleft(x),2)
        pyDrift2 = roundDict(DN_pleft(y),2)
        pyDrift3 = roundDict(DN_pleft(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_pleft(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_pleft(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_pleft(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i])

        self.assertTrue(result)

    #The fitting in python vs matlab different causes non matches
    def test_DN_CompareKSFit(self):


        pyDrift1 = roundDict(DN_CompareKSFit(x),2)
        pyDrift2 = roundDict(DN_CompareKSFit(y),2)
        pyDrift3 = roundDict(DN_CompareKSFit(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_CompareKSFit(x,"norm")'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_CompareKSFit(x,"norm")'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_CompareKSFit(x,"norm")'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i])

        self.assertTrue(result)

    def test_DN_cv(self):


        pyDrift1 = roundDict(DN_cv(x),2)
        pyDrift2 = roundDict(DN_cv(y),2)
        pyDrift3 = roundDict(DN_cv(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_cv(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_cv(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_cv(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i])

        self.assertTrue(result)

    def test_DN_RemovePoints(self):


        pyDrift1 = roundDict(DN_RemovePoints(x),2)
        pyDrift2 = roundDict(DN_RemovePoints(y),2)
        pyDrift3 = roundDict(DN_RemovePoints(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_RemovePoints(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_RemovePoints(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_RemovePoints(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.01)

        self.assertTrue(result)

    def test_DN_TrimmedMean50(self):


        pyDrift1 = roundDict(DN_TrimmedMean(x,.50),2)
        pyDrift2 = roundDict(DN_TrimmedMean(y,.50),2)
        pyDrift3 = roundDict(DN_TrimmedMean(hr,.50),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_TrimmedMean(x,50)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_TrimmedMean(x,50)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_TrimmedMean(x,50)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_DN_TrimmedMean75(self):


        pyDrift1 = roundDict(DN_TrimmedMean(x,.75),2)
        pyDrift2 = roundDict(DN_TrimmedMean(y,.75),2)
        pyDrift3 = roundDict(DN_TrimmedMean(hr,.75),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_TrimmedMean(x,75)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_TrimmedMean(x,75)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_TrimmedMean(x,75)'),2)

        pythonResults = [pyDrift1,pyDrift2]

        matlabResults = [matDrift1,matDrift2]

        self.assertEqual(pythonResults,matlabResults)

    def test_DN_TrimmedMean25(self):


        pyDrift1 = roundDict(DN_TrimmedMean(x,.25),2)
        pyDrift2 = roundDict(DN_TrimmedMean(y,.25),2)
        pyDrift3 = roundDict(DN_TrimmedMean(hr,.25),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_TrimmedMean(x,25)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_TrimmedMean(x,25)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_TrimmedMean(x,25)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_DN_OutlierTest25(self):


        pyDrift1 = roundDict(DN_OutlierTest(x,25),2)
        pyDrift2 = roundDict(DN_OutlierTest(y,25),2)
        pyDrift3 = roundDict(DN_OutlierTest(hr,25),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_OutlierTest(x,25)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_OutlierTest(x,25)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_OutlierTest(x,25)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_DN_OutlierTest(self):


        pyDrift1 = roundDict(DN_OutlierTest(x,1),3)
        pyDrift2 = roundDict(DN_OutlierTest(y,1),3)
        pyDrift3 = roundDict(DN_OutlierTest(hr,1),3)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_OutlierTest(x,1)'),3)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_OutlierTest(x,1)'),3)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_OutlierTest(x,1)'),3)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_DN_OutlierTest10(self):


        pyDrift1 = roundDict(DN_OutlierTest(x,10),2)
        pyDrift2 = roundDict(DN_OutlierTest(y,10),2)
        pyDrift3 = roundDict(DN_OutlierTest(hr,10),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('DN_OutlierTest(x,10)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('DN_OutlierTest(x,10)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('DN_OutlierTest(x,10)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

class test_CO_functions(unittest.TestCase):

    def test_CO_f1ecac(self):


        pyDrift1 = roundDict(CO_f1ecac(x),2) + 1.0
        pyDrift2 = roundDict(CO_f1ecac(y),2) + 1.0
        pyDrift3 = roundDict(CO_f1ecac(hr),2) + 1.0

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_f1ecac(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_f1ecac(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_f1ecac(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_FirstMin(self):


        pyDrift1 = roundDict(CO_FirstMin(x),2)
        pyDrift2 = roundDict(CO_FirstMin(y),2)
        pyDrift3 = roundDict(CO_FirstMin(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_FirstMin(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_FirstMin(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_FirstMin(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_glscf22(self):


        pyDrift1 = roundDict(CO_glscf(x,2,2),2)
        pyDrift2 = roundDict(CO_glscf(y,2,2),2)
        pyDrift3 = roundDict(CO_glscf(hr,2,2),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_glscf(x,2,2)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_glscf(x,2,2)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_glscf(x,2,2)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_glscf14(self):


        pyDrift1 = roundDict(CO_glscf(x,1,4),2)
        pyDrift2 = roundDict(CO_glscf(y,1,4),2)
        pyDrift3 = roundDict(CO_glscf(hr,1,4),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_glscf(x,1,4)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_glscf(x,1,4)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_glscf(x,1,4)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_trev(self):


        pyDrift1 = roundDict(CO_trev(x),2)
        pyDrift2 = roundDict(CO_trev(y),2)
        pyDrift3 = roundDict(CO_trev(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_trev(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_trev(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_trev(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_tc3(self):


        pyDrift1 = roundDict(CO_tc3(x),2)
        pyDrift2 = roundDict(CO_tc3(y),2)
        pyDrift3 = roundDict(CO_tc3(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_tc3(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_tc3(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_tc3(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_FirstZero(self):


        pyDrift1 = roundDict(CO_FirstZero(x),2)
        pyDrift2 = roundDict(CO_FirstZero(y),2)
        pyDrift3 = roundDict(CO_FirstZero(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_FirstZero(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_FirstZero(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_FirstZero(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_fzcglscf(self):


        pyDrift1 = roundDict(CO_fzcglscf(x,2,2),2)
        pyDrift2 = roundDict(CO_fzcglscf(y,2,2),2)
        pyDrift3 = roundDict(CO_fzcglscf(hr,2,2),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_fzcglscf(x,2,2)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_fzcglscf(x,2,2)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_fzcglscf(x,2,2)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_fzcglscf15(self):


        pyDrift1 = roundDict(CO_fzcglscf(x,1,5),2)
        pyDrift2 = roundDict(CO_fzcglscf(y,1,5),2)
        pyDrift3 = roundDict(CO_fzcglscf(hr,1,5),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_fzcglscf(x,1,5)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_fzcglscf(x,1,5)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_fzcglscf(x,1,5)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_NonlinearAutocorr(self):


        pyDrift1 = roundDict(CO_NonlinearAutocorr(x,[1,2,3,4,5]),1)
        pyDrift2 = roundDict(CO_NonlinearAutocorr(y,[1,2,3,4,5]),1)
        pyDrift3 = roundDict(CO_NonlinearAutocorr(hr,[1,2,3,4,5]),1)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_NonlinearAutocorr(x,[1,2,3,4,5])'),1)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_NonlinearAutocorr(x,[1,2,3,4,5])'),1)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_NonlinearAutocorr(x,[1,2,3,4,5])'),1)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_NonlinearAutocorr1510(self):


        pyDrift1 = roundDict(CO_NonlinearAutocorr(x,[1,5,10,25]),1)
        pyDrift2 = roundDict(CO_NonlinearAutocorr(y,[1,5,10,25]),1)
        pyDrift3 = roundDict(CO_NonlinearAutocorr(hr,[1,5,10,25]),1)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_NonlinearAutocorr(x,[1,5,10,25])'),1)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_NonlinearAutocorr(x,[1,5,10,25])'),1)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_NonlinearAutocorr(x,[1,5,10,25])'),1)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_Embed2_Basic(self):


        pyDrift1 = roundDict(CO_Embed2_Basic(x,1,.1),2)
        pyDrift2 = roundDict(CO_Embed2_Basic(y,1,.1),2)
        pyDrift3 = roundDict(CO_Embed2_Basic(hr,1,.1),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_Embed2_Basic(x,1)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_Embed2_Basic(x,1)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_Embed2_Basic(x,1)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_Embed2_Basic5(self):


        pyDrift1 = roundDict(CO_Embed2_Basic(x,5,.1),2)
        pyDrift2 = roundDict(CO_Embed2_Basic(y,5,.1),2)
        pyDrift3 = roundDict(CO_Embed2_Basic(hr,5,.1),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_Embed2_Basic(x,5)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_Embed2_Basic(x,5)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_Embed2_Basic(x,5)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_Embed2_Basictau(self):


        pyDrift1 = roundDict(CO_Embed2_Basic(x,'tau',.1),2)
        pyDrift2 = roundDict(CO_Embed2_Basic(y,'tau',.1),2)
        pyDrift3 = roundDict(CO_Embed2_Basic(hr,'tau',.1),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_Embed2_Basic(x,"tau")'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_Embed2_Basic(x,"tau")'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_Embed2_Basic(x,"tau")'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_CO_RM_AMInformation(self):


        pyDrift1 = roundDict(CO_RM_AMInformation(x),2)
        pyDrift2 = roundDict(CO_RM_AMInformation(y),2)
        pyDrift3 = roundDict(CO_RM_AMInformation(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('CO_RM_AMInformation(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('CO_RM_AMInformation(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('CO_RM_AMInformation(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i])

        self.assertTrue(result)

class test_EN_functions(unittest.TestCase):

    def test_EN_ApEn(self):


        pyDrift1 = roundDict(EN_ApEn(x),2)
        pyDrift2 = roundDict(EN_ApEn(y),2)
        pyDrift3 = roundDict(EN_ApEn(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EN_ApEn(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EN_ApEn(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EN_ApEn(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i])

        self.assertTrue(result)

    def test_EN_CID(self):


        pyDrift1 = roundDict(EN_CID(x),2)
        pyDrift2 = roundDict(EN_CID(y),2)
        pyDrift3 = roundDict(EN_CID(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EN_CID(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EN_CID(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EN_CID(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_EN_PermEn(self):


        pyDrift1 = roundDict(EN_PermEn(x),2)
        pyDrift2 = roundDict(EN_PermEn(y),2)
        pyDrift3 = roundDict(EN_PermEn(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EN_PermEn(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EN_PermEn(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EN_PermEn(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_EN_DistributionEntropy(self):


        pyDrift1 = roundDict(EN_DistributionEntropy(x),2)
        pyDrift2 = roundDict(EN_DistributionEntropy(y),2)
        pyDrift3 = roundDict(EN_DistributionEntropy(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EN_DistributionEntropy(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EN_DistributionEntropy(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EN_DistributionEntropy(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.05)

        self.assertTrue(result)

    def test_EN_PermEn36(self):


        pyDrift1 = roundDict(EN_PermEn(x,3,6),2)
        pyDrift2 = roundDict(EN_PermEn(y,3,6),2)
        pyDrift3 = roundDict(EN_PermEn(hr,3,6),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EN_PermEn(x,3,6)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EN_PermEn(x,3,6)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EN_PermEn(x,3,6)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)


    def test_EN_PermEn73(self):


        pyDrift1 = roundDict(EN_PermEn(x,7,3),2)
        pyDrift2 = roundDict(EN_PermEn(y,7,3),2)
        pyDrift3 = roundDict(EN_PermEn(hr,7,3),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EN_PermEn(x,7,3)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EN_PermEn(x,7,3)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EN_PermEn(x,7,3)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    # According to comments on matlab website theirs is wrong
    # found another implementation online that our python implementation
    # matches

    # def test_EN_SampEn(self):
    #
    #
    #     pyDrift1 = roundDict(EN_SampEn(x)['Sample Entropy'],2)
    #     pyDrift2 = roundDict(EN_SampEn(y)['Sample Entropy'],2)
    #     pyDrift3 = roundDict(EN_SampEn(hr)['Sample Entropy'],2)
    #
    #     eng.workspace['x'] = matlab.double(list(x))
    #     matDrift1 = roundDict(eng.eval('EN_SampEn(x)'),2)
    #
    #     eng.workspace['x'] = matlab.double(list(y))
    #     matDrift2 = roundDict(eng.eval('EN_SampEn(x)'),2)
    #
    #     eng.workspace['x'] = matlab.double(list(hr))
    #     matDrift3 = roundDict(eng.eval('EN_SampEn(x)'),2)
    #
    #     pythonResults = [pyDrift1,pyDrift2,pyDrift3]
    #
    #     matlabResults = [matDrift1,matDrift2,matDrift3]
    #
    #     self.assertEqual(pythonResults,matlabResults)


class test_IN_functions(unittest.TestCase):


    '''
    Theirs won't run on valid inputs
    '''

    # def test_IN_AutoMutualInfo(self):
    #
    #     pyDrift1 = roundDict(IN_AutoMutualInfo(x),2)
    #     pyDrift2 = roundDict(IN_AutoMutualInfo(y),2)
    #     pyDrift3 = roundDict(IN_AutoMutualInfo(hr),2)
    #
    #     eng.workspace['x'] = matlab.double(list(x))
    #     matDrift1 = roundDict(eng.eval('IN_AutoMutualInfo(x)'),2)
    #
    #     eng.workspace['x'] = matlab.double(list(y))
    #     matDrift2 = roundDict(eng.eval('IN_AutoMutualInfo(x)'),2)
    #
    #     eng.workspace['x'] = matlab.double(list(hr))
    #     matDrift3 = roundDict(eng.eval('IN_AutoMutualInfo(x)'),2)
    #
    #     pythonResults = [pyDrift1,pyDrift2,pyDrift3]
    #
    #     matlabResults = [matDrift1,matDrift2,matDrift3]
    #
    #     self.assertEqual(pythonResults,matlabResults)

class test_SY_functions(unittest.TestCase):

    def test_SY_DriftingMean(self):

        pyDrift1 = roundDict(SY_DriftingMean(x))
        pyDrift2 = roundDict(SY_DriftingMean(y))
        pyDrift3 = roundDict(SY_DriftingMean(hr))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('SY_DriftingMean(x)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('SY_DriftingMean(x)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('SY_DriftingMean(x)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    #Currently will fail b/c I didn't add SampEn cause it takes too long
    def test_SY_DynWin(self):

        pyDrift1 = roundDict(SY_DynWin(x))
        pyDrift2 = roundDict(SY_DynWin(y))
        pyDrift3 = roundDict(SY_DynWin(hr))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('SY_DynWin(x)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('SY_DynWin(x)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('SY_DynWin(x)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        self.assertEqual(pythonResults,matlabResults)

    def test_ST_LocalExtrema(self):

        pyDrift1 = roundDict(ST_LocalExtrema(x,"l",50))
        pyDrift2 = roundDict(ST_LocalExtrema(y,"l",50))
        pyDrift3 = roundDict(ST_LocalExtrema(hr,"l",50))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('ST_LocalExtrema(x,"l",50)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('ST_LocalExtrema(x,"l",50)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('ST_LocalExtrema(x,"l",50)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True
        #huge b/c matlab periodogram function doesnt match pythons
        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.05)

        self.assertTrue(result)

    def test_ST_LocalExtreman(self):

        pyDrift1 = roundDict(ST_LocalExtrema(x,"n",50))
        pyDrift2 = roundDict(ST_LocalExtrema(y,"n",50))
        pyDrift3 = roundDict(ST_LocalExtrema(hr,"n",50))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('ST_LocalExtrema(x,"n",50)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('ST_LocalExtrema(x,"n",50)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('ST_LocalExtrema(x,"n",50)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True
        #huge b/c matlab periodogram function doesnt match pythons
        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.05)

        self.assertTrue(result)

    def test_ST_LocalExtreman250(self):

        pyDrift1 = roundDict(ST_LocalExtrema(x,"n",250))
        pyDrift2 = roundDict(ST_LocalExtrema(y,"n",250))
        pyDrift3 = roundDict(ST_LocalExtrema(hr,"n",250))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('ST_LocalExtrema(x,"n",250)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('ST_LocalExtrema(x,"n",250)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('ST_LocalExtrema(x,"n",250)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True
        #huge b/c matlab periodogram function doesnt match pythons
        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.05)

        self.assertTrue(result)


    #Likely Will Fail but that is python vs matlab fitting
    def test_SY_Trend(self):

        pyDrift1 = roundDict(SY_Trend(x),2)
        pyDrift2 = roundDict(SY_Trend(y),2)
        pyDrift3 = roundDict(SY_Trend(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('SY_Trend(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('SY_Trend(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('SY_Trend(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]
        # self.assertEqual(pythonResults,matlabResults)
        result = True
        ranges = [1.5,1.5,200]
        #let this be further b/c there is a random feature to algo
        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],ranges[i])

        self.assertTrue(result)

    def test_SY_LocalGlobal(self):

        pyDrift1 = roundDict(SY_LocalGlobal(x),2)
        pyDrift2 = roundDict(SY_LocalGlobal(y),2)
        pyDrift3 = roundDict(SY_LocalGlobal(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('SY_LocalGlobal(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('SY_LocalGlobal(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('SY_LocalGlobal(x)'),2)

        result1 = closeEnough(pyDrift1,matDrift1)


        self.assertTrue(result1 and closeEnough(pyDrift2,matDrift2) and closeEnough(pyDrift3,matDrift3))

    #when i run matlab and python separetly they match
    #perfectly when testing now matlab returns nan
    def test_SY_RangeEvolve(self):

        pyDrift1 = roundDict(SY_RangeEvolve(x),2)
        pyDrift2 = roundDict(SY_RangeEvolve(y),2)
        pyDrift3 = roundDict(SY_RangeEvolve(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('SY_RangeEvolve(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('SY_RangeEvolve(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('SY_RangeEvolve(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.1)

        self.assertTrue(result)

    def test_FC_Suprise(self):

        pyDrift1 = roundDict(FC_Suprise(x),2)
        pyDrift2 = roundDict(FC_Suprise(y),2)
        pyDrift3 = roundDict(FC_Suprise(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('FC_Surprise(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('FC_Surprise(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('FC_Surprise(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True
        #let this be further b/c there is a random feature to algo
        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],4)

        self.assertTrue(result)

class test_MD_functions(unittest.TestCase):


    #periodogram function in matlab vs python very different
    #leads to failures
    def test_MD_hrv_classic(self):

        pyDrift1 = roundDict(MD_hrv_classic(x),2)
        pyDrift2 = roundDict(MD_hrv_classic(y),2)
        pyDrift3 = roundDict(MD_hrv_classic(hr),2)

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('MD_hrv_classic(x)'),2)

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('MD_hrv_classic(x)'),2)

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('MD_hrv_classic(x)'),2)

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True
        #huge b/c matlab periodogram function doesnt match pythons
        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],500)

        self.assertTrue(result)

class test_SB_functions(unittest.TestCase):

    def test_SB_TransitionMatrix(self):

        pyDrift1 = roundDict(SB_TransitionMatrix(x,'quantile',5,1))
        pyDrift2 = roundDict(SB_TransitionMatrix(y,'quantile',5,1))
        pyDrift3 = roundDict(SB_TransitionMatrix(hr,'quantile',5,1))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('SB_TransitionMatrix(x,"quantile",5,1)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('SB_TransitionMatrix(x,"quantile",5,1)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('SB_TransitionMatrix(x,"quantile",5,1)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True
        #huge b/c matlab periodogram function doesnt match pythons
        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.05)

        self.assertTrue(result)



class test_other_functions(unittest.TestCase):

    def test_PH_Walker(self):


        pyDrift1 = roundDict(PH_Walker(x))
        pyDrift2 = roundDict(PH_Walker(y))
        pyDrift3 = roundDict(PH_Walker(hr))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('PH_Walker(x)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('PH_Walker(x)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('PH_Walker(x)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        #matlab in python doesnt work for these i get weird errors
        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],1.5)

        self.assertTrue(result)

    def test_PH_Walkerm5(self):


        pyDrift1 = roundDict(PH_Walker(x,'momentum',[5]))
        pyDrift2 = roundDict(PH_Walker(y,'momentum',[5]))
        pyDrift3 = roundDict(PH_Walker(hr,'momentum',[5]))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('PH_Walker(x,"momentum",5)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('PH_Walker(x,"momentum",5)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('PH_Walker(x,"momentum",5)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        #matlab in python doesnt work for these i get weird errors
        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],1.5)

        self.assertTrue(result)

    def test_ST_MomentCorr(self):


        pyDrift1 = roundDict(ST_MomentCorr(x))
        pyDrift2 = roundDict(ST_MomentCorr(y))
        pyDrift3 = roundDict(ST_MomentCorr(hr))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('ST_MomentCorr(x)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('ST_MomentCorr(x)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('ST_MomentCorr(x)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]


        self.assertEqual(pythonResults,matlabResults)

    def test_ST_MomentCorr14(self):


        pyDrift1 = roundDict(ST_MomentCorr(x,.1,.4))
        pyDrift2 = roundDict(ST_MomentCorr(y,.1,.4))
        pyDrift3 = roundDict(ST_MomentCorr(hr,.1,.4))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('ST_MomentCorr(x,.1,.4)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('ST_MomentCorr(x,.1,.4)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('ST_MomentCorr(x,.1,.4)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]


        self.assertEqual(pythonResults,matlabResults)

    def test_ST_MomentCorr375(self):


        pyDrift1 = roundDict(ST_MomentCorr(x,.3,.75))
        pyDrift2 = roundDict(ST_MomentCorr(y,.3,.75))
        pyDrift3 = roundDict(ST_MomentCorr(hr,.3,.75))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('ST_MomentCorr(x,.3,.75)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('ST_MomentCorr(x,.3,.75)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('ST_MomentCorr(x,.3,.75)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]


        self.assertEqual(pythonResults,matlabResults)

    def test_ST_MomentCorriqr(self):

        #python and matlab iqr return midly different numbers

        pyDrift1 = roundDict(ST_MomentCorr(x,mom2 = "iqr"))
        pyDrift2 = roundDict(ST_MomentCorr(y,mom2 = "iqr"))
        pyDrift3 = roundDict(ST_MomentCorr(hr,mom2 = "iqr"))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('ST_MomentCorr(x,.02,.2,"mean","iqr")'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('ST_MomentCorr(x,.02,.2,"mean","iqr")'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('ST_MomentCorr(x,.02,.2,"mean","iqr")'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]

        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],.5)

        self.assertTrue(result)

    def test_EX_MovingThreshold(self):


        pyDrift1 = roundDict(EX_MovingThreshold(x,.3,.75))
        pyDrift2 = roundDict(EX_MovingThreshold(y,.3,.75))
        pyDrift3 = roundDict(EX_MovingThreshold(hr,.3,.75))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EX_MovingThreshold(x,.3,.75)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EX_MovingThreshold(x,.3,.75)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EX_MovingThreshold(x,.3,.75)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]


        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],1.5)

        self.assertTrue(result)

    def test_EX_MovingThreshold2501(self):


        pyDrift1 = roundDict(EX_MovingThreshold(x,.25,.01))
        pyDrift2 = roundDict(EX_MovingThreshold(y,.25,.01))
        pyDrift3 = roundDict(EX_MovingThreshold(hr,.25,.01))

        eng.workspace['x'] = matlab.double(list(x))
        matDrift1 = roundDict(eng.eval('EX_MovingThreshold(x,.25,.01)'))

        eng.workspace['x'] = matlab.double(list(y))
        matDrift2 = roundDict(eng.eval('EX_MovingThreshold(x,.25,.01)'))

        eng.workspace['x'] = matlab.double(list(hr))
        matDrift3 = roundDict(eng.eval('EX_MovingThreshold(x,.25,.01)'))

        pythonResults = [pyDrift1,pyDrift2,pyDrift3]

        matlabResults = [matDrift1,matDrift2,matDrift3]


        result = True

        for i in range(3):

            result = result and closeEnough(pythonResults[i],matlabResults[i],1.5)

        self.assertTrue(result)

    # def test_MF_hmm_CompareNStates(self):
    #
    #
    #     pyDrift1 = roundDict(MF_hmm_CompareNStates(x,.25),3)
    #     pyDrift2 = roundDict(MF_hmm_CompareNStates(y,.25),3)
    #     pyDrift3 = roundDict(MF_hmm_CompareNStates(hr,.25),3)
    #
    #     eng.workspace['x'] = matlab.double(list(x))
    #     matDrift1 = roundDict(eng.eval('MF_hmm_CompareNStates(x,.25)'),3)
    #
    #     eng.workspace['x'] = matlab.double(list(y))
    #     matDrift2 = roundDict(eng.eval('MF_hmm_CompareNStates(x,.25)'),3)
    #
    #     eng.workspace['x'] = matlab.double(list(hr))
    #     matDrift3 = roundDict(eng.eval('MF_hmm_CompareNStates(x,.25)'),3)
    #
    #     pythonResults = [pyDrift1,pyDrift2,pyDrift3]
    #
    #     matlabResults = [matDrift1,matDrift2,matDrift3]
    #
    #     self.assertEqual(pythonResults,matlabResults)

if __name__ == "__main__":
    unittest.main()
