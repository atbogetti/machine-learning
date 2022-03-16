import numpy as np
import pandas as pd
import h5py
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from itertools import combinations 

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
distmats = []

for i in range(1,501):
    fstring = 'iterations/iter_'+str(i).zfill(8)
    pcoord = h5file[fstring]['pcoord'][:,1,-1]
    success = h5file[fstring]['auxdata/success'][:,-1]
    weight = h5file[fstring]['seg_index']['weight']
    shape = h5file[fstring]['seg_index']['weight'].shape[0]
    distmat = h5file[fstring]['auxdata/distmat'][:,-1]

    for j in range(1,shape+1):
        identity = str(i).zfill(3)+str(j).zfill(3)

        iterations.append(i)
        segments.append(j)
        ids.append(identity)
        weights.append(weight[j-1])
        mindists.append(pcoord[j-1])
        successes.append(int(success[j-1]))
        distmats.append(distmat[j-1])

df['ID'] = ids
df['Iteration'] = iterations
df['Segment'] = segments
df['Weight'] = weights
df['Mindist'] = mindists
df['Success'] = successes

distmats = np.array(distmats)
atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 
         'H2', 'C10', 'C11', 'C12', 'C13', 'C14', 'H3', 'H4', 'H5', 
         'H6', 'C15', 'C16', 'C17', 'C18', 'C19', 'H7', 'H8', 'H9', 
         'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'N1', 'N2', 'N3']
perm = combinations(atoms,2) 
a = np.array(list(perm))

for idx in range(0,666):
    if 'H' in a[idx,0]:
        continue
    elif 'H' in a[idx,1]:
        continue
    else:
        fname = str(a[idx,0])+"-"+str(a[idx,1])
        df[fname] = distmats[:,idx]

ionpair = df.loc[(df['Mindist'] < 5) & (df['Mindist'] > 2.5)]

h5file.close()

# Split the data and train the model

print("training the model...")

X = df.iloc[:,6:672]
y = df.iloc[:,5]
W = df.iloc[:,3]
Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest = train_test_split(X, y, W, test_size=0.25, random_state=None)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Predict and evaluate

print("evaluating predictions...")

predictions = regr.predict(Xtest)
auc = roc_auc_score(ytest, predictions)
print("AUC score:",auc)
