import timeit
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from itertools import combinations 
from tqdm import tqdm

def absolute_maximum_scale(series):
    return series / series.abs().max()

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

# Prepare the data

print("processing data...")

h5file = h5py.File("../../cat10_rc/multi.h5", "r")

df = pd.DataFrame()
iterations = []
segments = []
ids = []
weights = []
mindists = []
successes = []
charges = []
distmats = []

for i in tqdm(range(50,501), ncols=80):
    fstring = 'iterations/iter_'+str(i).zfill(8)
    pcoord = h5file[fstring]['pcoord'][:,1,-1]
    success = h5file[fstring]['auxdata/success'][:,-1]
    weight = h5file[fstring]['seg_index']['weight'] 
    shape = h5file[fstring]['seg_index']['weight'].shape[0]
    charge = h5file[fstring]['auxdata/charges'][:,-1]
    distmat = h5file[fstring]['auxdata/distmat'][:,-1]

    for j in range(1,shape+1):
        identity = str(i).zfill(3)+str(j).zfill(3)

        iterations.append(i)
        segments.append(j)
        ids.append(identity)
        weights.append(weight[j-1])
        mindists.append(pcoord[j-1])
        successes.append(int(success[j-1]))
        charges.append(charge[j-1])
        distmats.append(distmat[j-1])

df['ID'] = ids
df['Iteration'] = iterations
df['Segment'] = segments
df['Weight'] = weights
df['Mindist'] = mindists
df['Success'] = successes

charges = np.array(charges)
distmats = np.array(distmats)

atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 
         'H2', 'C10', 'C11', 'C12', 'C13', 'C14', 'H3', 'H4', 'H5', 
         'H6', 'C15', 'C16', 'C17', 'C18', 'C19', 'H7', 'H8', 'H9', 
         'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'N1', 'N2', 'N3']

for idx, atom in enumerate(atoms):
    fname = "q"+atom
    df[fname] = charges[:,idx]

perm = combinations(atoms,2) 
a = np.array(list(perm))

for i in range(0,a.shape[0]):
    if 'H' in a[i,0]:
        continue
    elif 'H' in a[i,1]:
        continue
    else:
        fname = str(a[i,0])+"-"+str(a[i,1])
        df[fname] = distmats[:,i]

h5file.close()

#ionpair = df.loc[(df['Mindist'] < 5) & (df['Mindist'] > 2.25)]

# Split/scale the data and train the model with weights from WE

print("training the model...")

X = df.iloc[:,6:]
y = df.iloc[:,5]
W = df.iloc[:,3]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, W, test_size=0.5, random_state=42)

W_train = min_max_scaling(W_train)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Train the model
regr = AdaBoostClassifier()
regr.fit(X_train, y_train, W_train)

# Predict and evaluate

print("evaluating predictions...")

y_pred = regr.predict(X_test)

auc = roc_auc_score(y_test, y_pred)
print("AUC score:",auc)

target_names = ['no reaction', 'reaction']

print(confusion_matrix(y_test, y_pred))
