import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import l1_min_c
import time

#######
#
# Build main df for analysis
#
########
patientDF = pd.read_csv('/Users/j/Documents/Work/Coulter Analysis/Daily Average/UVA NICU Infants.csv')
patientDF['id'] = patientDF['PatientID']

mat_hr = pd.read_csv('./randomDailySample_matlab_HR.csv')
mat_hr = mat_hr.add_prefix('HR_')
mat_hr = mat_hr.rename(columns={'HR_id': 'id', 'HR_time': 'time'})

mat_sp = pd.read_csv('./randomDailySample_matlab_SP.csv')
mat_sp = mat_sp.add_prefix('SP_')
mat_sp = mat_sp.rename(columns={'SP_id': 'id', 'SP_time': 'time'})

df = pd.read_csv('../dailySample/randomDailySample.csv')
df = pd.merge(df, mat_hr, on=['id','time'])
df = pd.merge(df, mat_sp, on=['id','time'])

clean = list(pd.read_csv('normalized_data.csv',nrows=2).columns)

df = df[clean]
# Or can make outcomes from patientDFß
outcomes = pd.read_csv('../HCTSA Outcomes.csv')
outcomes = outcomes.rename(columns = {'ID':'id','Time':'time'})
df = pd.merge(df, outcomes, on=['id','time'])

#Rename columns all ' ' becomes '.'
new_names = {}
new_columns = []
columns = list(df.columns)
for col in columns:
    new = col.replace(" ",'.',10)
    new = new.replace('[','.')
    new = new.replace("'",'.')
    new = new.replace(',','.')
    new = new.replace(']','.')
    new = new.replace('_','.')
    new_names[col] = new
    new_columns.append(new)
df = df.rename(columns=new_names)

########################
# Make df for each Model
########################
clusters_20 = ["HR.CO.tc3.1..denom","HR.DN.cv.3","HR.Quantile.99" ,"HR.ST.LocalExtrema.n100.minabsmin","HR.skew2", "HR.SB.TransitionMatrix23.sumdiagcov" ,"HR.SB.MotifThree.quantile.hhhh" ,"SP.PH.Walkerprop.0.9..sw.stdrat","SP.CO.tc3.1..denom","HR.MF.arfit.sbc.7","HR.SB.MotifThree.diffquant.hhhh", "HR.SY.StdNthDer.17","SP.ST.LocalExtrema.n100.minabsmin","SP.DN.RemovePointsmin.0.2.mean", "SP.SB.TransitionMatrix22.mineig","SP.MF.arfit.sbc.7" ,"SP.AutoCorr.lag.4" ,"SP.SB.MotifThree.diffquant.hhh","SP.SB.TransitionMatrix21.T10", "SP.SB.BinaryMethod.iqr.pstretch1"]
clusters_20.extend(['id','time','BW','GA','PMA','DIED.WEEK','DIED'])
c20 = df[clusters_20]
patientDF = patientDF[['id','Outcome PMA']]
c20 = pd.merge(c20, patientDF, on='id')

hr_only = ["HR.CO.tc3.1..denom","HR.DN.cv.3","HR.Quantile.99" ,"HR.ST.LocalExtrema.n100.minabsmin","HR.skew2", "HR.SB.TransitionMatrix23.sumdiagcov" ,"HR.SB.MotifThree.quantile.hhhh" ,"HR.MF.arfit.sbc.7","HR.SB.MotifThree.diffquant.hhhh", "HR.SY.StdNthDer.17"]
hr_only.extend(['id','time','BW','GA','PMA','DIED.WEEK','DIED'])
hr_df = df[hr_only]
patientDF = patientDF[['id','Outcome PMA']]
hr_df = pd.merge(hr_df, patientDF, on='id')

sp_only = ["SP.PH.Walkerprop.0.9..sw.stdrat","SP.CO.tc3.1..denom","SP.ST.LocalExtrema.n100.minabsmin","SP.DN.RemovePointsmin.0.2.mean", "SP.SB.TransitionMatrix22.mineig","SP.MF.arfit.sbc.7" ,"SP.AutoCorr.lag.4" ,"SP.SB.MotifThree.diffquant.hhh","SP.SB.TransitionMatrix21.T10", "SP.SB.BinaryMethod.iqr.pstretch1"]
sp_only.extend(['id','time','BW','GA','PMA','DIED.WEEK','DIED'])
sp_df = df[sp_only]
patientDF = patientDF[['id','Outcome PMA']]
sp_df = pd.merge(sp_df, patientDF, on='id')

diff_uu = ['HR.SB.MotifTwo.diff.uu']
diff_uu.extend(['id','time','BW','GA','PMA','DIED.WEEK','DIED'])
diff_df = df[diff_uu]
patientDF = patientDF[['id','Outcome PMA']]
diff_df = pd.merge(diff_df, patientDF, on='id')

mean = ['HR.mean','HR.std','SP.mean','SP.std']
mean.extend(['id','time','BW','GA','PMA','DIED.WEEK','DIED'])
mean_df = df[mean]
patientDF = patientDF[['id','Outcome PMA']]
mean_df = pd.merge(mean_df, patientDF, on='id')

#############################
# Modelling Fucntions skip to fitting below
#############################
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.model_selection._split import _BaseKFold
from collections import Counter, defaultdict
from sklearn.utils import check_random_state

def normalize(x):

    return 1 / ( 1 + np.exp( - ( x - x.median() ) / ( 1.35 * (x.quantile(.75) - x.quantile(.25)) ) ) )

class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.

    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.

    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).

    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 6 6 7]
           [1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 8 8]
           [0 0 1 1 1 0 0]
    TRAIN: [1 1 3 3 3 4 5 5 5 5 8 8]
           [0 0 1 1 1 1 0 0 0 0 0 0]
     TEST: [2 2 6 6 7]
           [1 1 0 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]

    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).

    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    # Implementation based on this kaggle kernel:
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def _iter_test_indices(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts,
                                      key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(np.std(
                        [y_counts_per_fold[j][label] / y_distr[label]
                         for j in range(self.n_splits)]))
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group in enumerate(groups)
                            if group in groups_per_fold[i]]
            yield test_indices


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
def fitLogistic(x,y):
    x = x.to_numpy().reshape(-1, 1)
    y = y.values
    results = {}
    logmodel = LogisticRegression()
    logmodel.fit(x,y)
    probs = logmodel.predict_proba(x)[:,1]
    results = {}
    return results,logmodel

def test_f1(y,preds):
    TP = sum(y[y == 1] == preds[y == 1])
    FP = len(y[y==1]) - TP
    TN = sum(y[y == 0] == preds[y == 0])
    FN = len(y[y==0]) - TN
    prec = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2 * (prec * recall) / ( prec + recall )

def fitLogistic2(x,y):
    x = x.to_numpy()
    y = y.values
    results = {}
    logmodel = LogisticRegression()
    logmodel.fit(x,y)
    return {},logmodel

def calc_test_stats(x_test,y_test,model):
        probs = model.predict_proba(x_test)[:,1]
        tpr, fpr, thresholds = metrics.roc_curve(y_test, probs, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        y = y_test.copy()
        y = y.reset_index(drop=True)
        ordering = probs.argsort()
        y = y[ordering]
        preds = np.concatenate((np.repeat(0,len(y) - sum(y)),np.repeat(1,sum(y))))
        f1 = metrics.f1_score(y,preds)
        results = {}
        results['f1'] = f1
        if auc < .5:
            auc = 1 - auc
        results['auc'] = auc
        return results

def fitLasso(X,y,c):
    clf.set_params(C=c)
    clf.fit(X, y)
    return clf, clf.coef_.ravel().copy()
def make_X_Y(df,outcome = 1,demo = True):
    df['y'] = 0
    if not isinstance(outcome,list):
        df['y'][(df['DIED'] == 1) & (df['Outcome PMA'] - df['PMA'] <= outcome)] = 1
    else:
        df['y'][(df['DIED'] == 1) & (df['Outcome PMA'] - df['PMA'] <= outcome[0]) & (df['Outcome PMA'] - df['PMA'] > outcome[1])] = 1
    if demo:
        X = df.iloc[:,:-4]
        X['y'] = df['y']
        X['WEEK'] = X['PMA'] - X['GA']
        X = X.dropna()
        Y = X['y']
        ids = X['id']
        print('Patients with data: ')
        print(len(set(X[X['y'] == 1]['id'])))
        X = X.drop(['id', 'time','y','PMA'], axis=1)
        X.iloc[:,:-3] = X.iloc[:,:-3].transform(normalize)
        #X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
        return X,Y,ids
    X = df.iloc[:,:20]
    X['y'] = df['y']
    X['id'] = df['id']
    X = X.dropna()
    Y = X['y']
    ids = X['id']
    print('Patients with data: ')
    print(len(set(X[X['y'] == 1]['id'])))
    X = X.drop(['id','y'], axis=1)
    X.iloc[:,:-3] = X.iloc[:,:-3].transform(normalize)
    #X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
    return X,Y,ids

def make_models(df,outcome = 1,demo = True,cs = [5]):
    X,Y,ids = make_X_Y(df,outcome,demo)
    clf = linear_model.LogisticRegression(penalty='l1', solver='liblinear',
                                          tol=1e-6, max_iter=int(1e6),
                                          warm_start=True,
                                          intercept_scaling=10000.)
    group_kfold = StratifiedGroupKFold(n_splits=3)
    group_kfold.get_n_splits(X, Y, ids)
    Y = Y.reset_index(drop = True)
    X = X.reset_index(drop = True)
    ids = ids.reset_index(drop = True)
    splits = 3
    group_kfold = StratifiedGroupKFold(n_splits=splits)
    group_kfold.get_n_splits(X, Y, ids)
    full = {}
    import warnings
    warnings.filterwarnings("ignore")
    j = 1
    for train_index, test_index in group_kfold.split(X, Y.astype(int), ids):
        print('Iteration is: ' + str(j))
        j = j + 1
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y[train_index], Y[test_index]

        for i in range(len(cs)):

            print('Fitting for c:' + str(cs[i]))
            c = cs[i]
            model, columns = fitLasso(X_train,y_train,c)

            if c in full.keys():
                full[c] += calc_test_stats(X_test,y_test,model)['auc'] / splits
            else:
                full[c] = calc_test_stats(X_test,y_test,model)['auc'] / splits
    return full

########################################
#
# Fit lasso models for each
#
#########################################

print('20 Medoids modelling')
print(make_models(c20,1))
print(make_models(c20,1/7))
print(make_models(c20,[2/7,3/7]))
print(make_models(c20,[6/7,7/7]))

print('HR only modelling')
print(make_models(hr,1))
print(make_models(hr,1/7))
print(make_models(hr,[2/7,3/7]))
print(make_models(hr,[6/7,7/7]))

print('SPo2 only modelling')
print(make_models(sp,1))
print(make_models(sp,1/7))
print(make_models(sp,[2/7,3/7]))
print(make_models(sp,[6/7,7/7]))


################################
# Clustering graphic
################################
import scipy
import seaborn as sns
r = scipy.stats.spearmanr(c20, axis=0)
sns_plot = sns.clustermap(
    pd.DataFrame(
        #squareform(dist.pdist(reshaped.T, 'correlation')),
        abs(r.correlation),
        columns = c20.columns,
        index = c20.columns
    ),
    cmap="PuBu"
    #cmap=sns.color_palette(uva_colormap)
)
ax = sns_plot.ax_heatmap
ax.set_xticks([])
ax.set_yticks([])
sns_plot.savefig("corr_20_features.png",dpi = 300)

import scipy
import seaborn as sns
r = scipy.stats.spearmanr(df, axis=0)
sns_plot = sns.clustermap(
    pd.DataFrame(
        #squareform(dist.pdist(reshaped.T, 'correlation')),
        abs(r.correlation),
        columns = df.columns,
        index = df.columns
    ),
    cmap="PuBu"
    #cmap=sns.color_palette(uva_colormap)
)
ax = sns_plot.ax_heatmap
ax.set_xticks([])
ax.set_yticks([])
sns_plot.savefig("corr_all_features.png",dpi = 300)
