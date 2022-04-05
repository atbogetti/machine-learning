import timeit
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn import svm
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
#angles = []
aoas = []

for i in tqdm(range(50,501), ncols=80):
    fstring = 'iterations/iter_'+str(i).zfill(8)
    pcoord = h5file[fstring]['pcoord'][:,1,-1]
    success = h5file[fstring]['auxdata/success'][:,-1]
    weight = h5file[fstring]['seg_index']['weight'] 
    shape = h5file[fstring]['seg_index']['weight'].shape[0]
    charge = h5file[fstring]['auxdata/charges'][:,-1]
    distmat = h5file[fstring]['auxdata/distmat'][:,-1]
#    angle = h5file[fstring]['auxdata/angles'][:,-1]
    aoa = h5file[fstring]['auxdata/aoa'][:,-1]

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
#        angles.append(angle[j-1])
        aoas.append(aoa[j-1])

df['ID'] = ids
df['Iteration'] = iterations
df['Segment'] = segments
df['Weight'] = weights
df['Mindist'] = mindists
df['Success'] = successes

charges = np.array(charges)
distmats = np.array(distmats)
#angles = np.array(angles)
aoas = np.array(aoas)

atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 
         'H2', 'C10', 'C11', 'C12', 'C13', 'C14', 'H3', 'H4', 'H5', 
         'H6', 'C15', 'C16', 'C17', 'C18', 'C19', 'H7', 'H8', 'H9', 
         'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'N1', 'N2', 'N3']

for idx, atom in enumerate(atoms):
    if 'H' in atom:
        continue
    else:
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

#perm2 = combinations(atoms,3)
#a2 = np.array(list(perm2))
#
#for i in range(0,a2.shape[0]):
#    if 'H' in a2[i,0]:
#        continue
#    elif 'H' in a2[i,1]:
#        continue
#    elif 'H' in a2[i,2]:
#        continue
#    else:
#        fname = 'a' + str(a2[i,0]) + '-' + str(a2[i,1]) + '-' + str(a2[i,2])
#        df[fname] = angles[:,i]

a3 = np.array(['P1-N1', 'P1-N2', 'P1-N3', 'P1-Or', 'P2-N1', 'P2-N2', 'P2-N3', 'P2-Or'])

for i in range(0,a3.shape[0]):
    fname = a3[i]
    df[fname] = aoas[:,i]

h5file.close()

print('pruning samples based on weight...')

ionpair = df.loc[(df['Mindist'] < 5) & (df['Mindist'] > 2.25)]

unsuccessful_count = np.where(ionpair['Success'] == 0)[0].shape[0]
successful_count = np.where(ionpair['Success'] == 1)[0].shape[0]
difference = unsuccessful_count - successful_count
print('successful:', successful_count, 'unsuccessful:', unsuccessful_count)

unsuccessful = ionpair.loc[(ionpair['Success'] == 0)]
sorted_weights = unsuccessful.sort_values(by=['Weight'])
to_remove = sorted_weights[:difference]
to_remove_IDs = to_remove['ID'].values

ionpair = ionpair[ionpair.ID.isin(to_remove_IDs) == False]

unsuccessful_count = np.where(ionpair['Success'] == 0)[0].shape[0]
successful_count = np.where(ionpair['Success'] == 1)[0].shape[0]
print('successful:', successful_count, 'unsuccessful:', unsuccessful_count)

# Split/scale the data and train the model with weights from WE

print("training the model...")

names = ionpair.columns.values.tolist()
X_names = names[6:]
X = ionpair.iloc[:,6:]
y = ionpair.iloc[:,5]
W = ionpair.iloc[:,3]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, W, test_size=0.25, random_state=42)

W_train = min_max_scaling(W_train)

##pca = PCA(10)
##pca_train = pd.DataFrame(pca.fit_transform(X_train))
##pca_test = pd.DataFrame(pca.fit_transform(X_test))

from sklearn import tree

# Train the model
clf = svm.SVC()
clf.fit(X_train, y_train, W_train)
#clf.fit(pca_train, y_train, W_train)

# Predict and evaluate

print("evaluating predictions...")

y_pred = clf.predict(X_test)
#y_pred = clf.predict(pca_test)

y_probs = clf.decision_function(X_test)
#y_probs = y_probs[1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_probs)
roc_auc = metrics.auc(fpr, tpr)
print("AUC score:", roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("roc_curve.pdf")

target_names = ['no reaction', 'reaction']

print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
