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

h5file = h5py.File("../../../cat15_rc/multi.h5", "r")

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
         'H2', 'C10', 'C11', 'C12', 'C13', 'C14', 'H3', 'H4', 'C15',
         'H5', 'C15', 'C16', 'C17', 'C18', 'C19', 'H6', 'H7', 'H8', 
         'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'F1', 'F2', 'F3',
         'N1', 'N2', 'N3']

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

ionpair = df.loc[(df['Mindist'] < 3.25) & (df['Mindist'] > 2.25)]
unsuccessful_weight = ionpair['Weight'].loc[(ionpair['Success'] == 0)].sum()
successful_weight = ionpair['Weight'].loc[(ionpair['Success'] == 1)].sum()

unsuccessful = ionpair.loc[(ionpair['Success'] == 0)]
sorted_weights = unsuccessful.sort_values(by=['Weight'], ascending=False)

n = 1
check_sum = sorted_weights['Weight'][:n].sum()
while check_sum <= successful_weight:
    n += 1
    check_sum = sorted_weights['Weight'][:n].sum()

print(n, 'unsuccessful samples needed to equal successful sample weights')

to_remove = sorted_weights[n:]
to_remove_IDs = to_remove['ID'].values

ionpair = ionpair[ionpair.ID.isin(to_remove_IDs) == False]

unsuccessful_count = np.where(ionpair['Success'] == 0)[0].shape[0]
successful_count = np.where(ionpair['Success'] == 1)[0].shape[0]
print('successful:', successful_count, 'unsuccessful:', unsuccessful_count)

unsuccessful_weight_sum = ionpair['Weight'].loc[(ionpair['Success'] == 0)].sum()
successful_weight_sum = ionpair['Weight'].loc[(ionpair['Success'] == 1)].sum()
print('total unsuccessful weight:', unsuccessful_weight_sum, 'total successful weight:', successful_weight_sum)

# Split/scale the data and train the model with weights from WE

print("training the model...")

names = ionpair.columns.values.tolist()
X_names = names[6:]
#X = ionpair.iloc[:,6:]
for feature_id in range(6,ionpair.shape[1]):
    X = ionpair.iloc[:,feature_id]
    print("Feature:", names[feature_id])
    y = ionpair.iloc[:,5]
    W = ionpair.iloc[:,3]
    
    X = StandardScaler().fit_transform(X.values.reshape(-1,1))
    
    X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, y, W, test_size=0.25, random_state=42)
    
    W_train = min_max_scaling(W_train)
    
    
    # Train the model
    regr = linear_model.LogisticRegression(solver='liblinear', penalty='l1', max_iter=1e6, C=1)
    regr.fit(X_train, y_train, W_train)
    
    # Predict and evaluate
    
    print("evaluating predictions...")
    
    y_pred = regr.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred)
    print("AUC score:",auc)
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_curve, auc
    
    acc = regr.score(X_test, y_test)
    print("Overall accuracy:", acc)
    yscore = regr.predict_proba(X_test)[:,1]
    fpr = dict()
    tpr = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
    #roc_auc = dict()
    #roc_auc = auc(fpr, tpr)
    #plt.figure(figsize=(10,10))
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([-0.05, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.grid(True)
    #plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
    #plt.legend(loc="lower right", shadow=True, fancybox =True)
    #plt.savefig("auc_curve.pdf")
    
    target_names = ['no reaction', 'reaction']
    
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
