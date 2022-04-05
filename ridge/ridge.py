import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from itertools import combinations 
from tqdm import tqdm

def absolute_maximum_scale(series):
    return series / series.abs().max()

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

# Prepare the data

print("processing data...")

h5file = h5py.File("../../cat10_rc/01/west.h5", "r")

df = pd.DataFrame()
iterations = []
segments = []
ids = []
weights = []
mindists = []
successes = []
distmats = []
angles = []
aoas = []
charges = []

for i in tqdm(range(50,501), ncols=80):
    fstring = 'iterations/iter_'+str(i).zfill(8)
    pcoord = h5file[fstring]['pcoord'][:,1,-1]
    success = h5file[fstring]['auxdata/success'][:,-1]
    weight = h5file[fstring]['seg_index']['weight'] 
    shape = h5file[fstring]['seg_index']['weight'].shape[0]
    distmat = h5file[fstring]['auxdata/distmat'][:,-1]
    angle = h5file[fstring]['auxdata/angles'][:,-1]
    aoa = h5file[fstring]['auxdata/aoa'][:,-1]
    charge = h5file[fstring]['auxdata/charges'][:,-1]

    for j in range(1,shape+1):
        identity = str(i).zfill(3)+str(j).zfill(3)

        iterations.append(i)
        segments.append(j)
        ids.append(identity)
        weights.append(weight[j-1])
        mindists.append(pcoord[j-1])
        successes.append(int(success[j-1]))
        distmats.append(distmat[j-1])
        angles.append(angle[j-1])
        aoas.append(angle[j-1])
        charges.append(charge[j-1])

df['ID'] = ids
df['Iteration'] = iterations
df['Segment'] = segments
df['Weight'] = weights
df['Mindist'] = mindists
df['Success'] = successes

distmats = np.array(distmats)
angles = np.array(angles)
aoas = np.array(aoas)
charges = np.array(charges)

atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 
         'H2', 'C10', 'C11', 'C12', 'C13', 'C14', 'H3', 'H4', 'H5', 
         'H6', 'C15', 'C16', 'C17', 'C18', 'C19', 'H7', 'H8', 'H9', 
         'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'N1', 'N2', 'N3']

perm = combinations(atoms,2) 
a = np.array(list(perm))

perm2 = combinations(atoms,3)
a2 = np.array(list(perm2))

a3 = np.array(['P1-N1', 'P1-N2', 'P1-N3', 'P1-Or', 'P2-N1', 'P2-N2', 'P2-N3', 'P2-Or'])

for i in range(0,a.shape[0]):
    if 'H' in a[i,0]:
        continue
    elif 'H' in a[i,1]:
        continue
    else:
        fname = str(a[i,0])+"-"+str(a[i,1])
        df[fname] = distmats[:,i]

for i in range(0,a2.shape[0]):
    if 'H' in a2[i,0]:
        continue
    elif 'H' in a2[i,1]:
        continue
    elif 'H' in a2[i,2]:
        continue
    else:
        fname = 'a' + str(a2[i,0]) + '-' + str(a2[i,1]) + '-' + str(a2[i,2])
        df[fname] = angles[:,i]

for i in range(0,a3.shape[0]):
    fname = a3[i]
    df[fname] = aoas[:,i]

for idx, atom in enumerate(atoms):
    if 'H' in atom:
        continue
    else:
        fname = "q"+atom
        df[fname] = charges[:,idx]

ionpair = df.loc[(df['Mindist'] < 4) & (df['Mindist'] > 2.25)]

#Normalize weights
df['Weight'] = min_max_scaling(df['Weight'])

#plt.hist(df['Weight'], bins=100)
#plt.savefig("weights_hist.pdf")

print(ionpair)

h5file.close()

# Split/scale the data and train the model with weights from WE

print("training the model...")

#ionpair['qC1'].plot(kind='hist')
#plt.savefig("qC1_hist.pdf")

X = ionpair.iloc[:,6:]
#X = X.values.reshape(-1,1)
y = ionpair.iloc[:,5]
W = ionpair.iloc[:,3]
Xtrain, Xtest, ytrain, ytest, Wtrain, Wtest = train_test_split(X, y, W, test_size=0.25, random_state=None)

#Scale feature data
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain_scaled = scaler.transform(Xtrain)

scaler = preprocessing.StandardScaler().fit(Xtest)
Xtest_scaled = scaler.transform(Xtest)

## Do a quick PCA
#pca = PCA(1000)
#pca_data = pd.DataFrame(pca.fit_transform(Xtrain_scaled))
##pca_data = pd.DataFrame(pca.fit_transform(Xtrain_scaled), columns=['Pc1', 'Pc2', 'Pc3', 'Pc4', 'Pc5', 'Pc6', 'Pc7', 'Pc8', 'Pc9', 'Pc10'])
#pca_data_test = pd.DataFrame(pca.fit_transform(Xtest_scaled))
#pca_data_test = pd.DataFrame(pca.fit_transform(Xtest_scaled), columns=['Pc1', 'Pc2', 'Pc3', 'Pc4', 'Pc5', 'Pc6', 'Pc7', 'Pc8', 'Pc9', 'Pc10'])
#pca_data['Success'] = ionpair['Success']

# Plot the first two PCs colored by the cluster labels
#pca_data['Pc1'].plot(kind='hist', bins=50)
#pca_data.plot(kind='scatter', x='Pc1', y='Pc2', c='Success', cmap='jet')
#plt.savefig("pca_2d.pdf")

# Train the model
regr = linear_model.RidgeClassifier()
regr.fit(Xtrain_scaled, ytrain, Wtrain)
#regr.fit(pca_data, ytrain, Wtrain)

# Predict and evaluate

print("evaluating predictions...")

predictions = regr.predict(Xtest_scaled)
#predictions = regr.predict(pca_data_test)
auc = roc_auc_score(ytest, predictions)
print("AUC score:",auc)
