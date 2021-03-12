import numpy as np
import pandas as pd
import itertools
import pickle
import time

from sys import getsizeof
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

import tools


# Importing the original data
file_dir = 'C:/Users/yle4/OneDrive - CDC/Documents/projects/az covid/'
records = pd.read_csv(file_dir + 'records.csv')

# List of symptom names and case definitions
symptom_list = [
    'fever', 'chills', 'shiver', 'ma', 'congestion',
    'sorethroat', 'cough', 'sob', 'difficultbreath', 'nauseavom',
    'headache', 'abpain', 'diarrhea', 'losstastesmell', 'fatigue'
]

# Loading the arrays and making the targets
X = pickle.load(open(file_dir + 'X.pkl', 'rb'))
pcr = np.array(records.poscovid == 1, dtype=np.uint8)
ant = np.array(records.ANTIGEN_RESULT_ == 'Positive', dtype=np.uint8)
ml = np.concatenate((pcr.reshape(-1, 1), 
                     ant.reshape(-1, 1)), axis=1)
mc = np.zeros((X.shape[0]), dtype=np.uint8)
for i in range(X.shape[0]):
    if pcr[i] == 0 and ant[i] == 0:
        mc[i] = 0
    elif pcr[i] == 1 and ant[i] == 1:
        mc[i] = 3
    elif pcr[i] == 1 and ant[i] == 0:
        mc[i] = 1
    if pcr[i] == 0 and ant[i] == 1:
        mc[i] = 2

'''Results for PCR'''
# Splitting into traininig and test sets
train, test = train_test_split(range(X.shape[0]),
                               test_size=0.4,
                               stratify=pcr,
                               random_state=2021)
val, test = train_test_split(test,
                             test_size=0.5,
                             stratify=pcr[test],
                             random_state=2021)

# Running a few baseline models
lgr = LogisticRegression(penalty='none')
lgr.fit(X, pcr)
coefs = pd.DataFrame(np.exp(lgr.coef_)[0],
                     columns=['aOR'])
coefs['symptom'] = symptom_list
coefs.to_csv(file_dir + 'pcr_lgr_coefs.csv', index=False)

pcr_rf = RandomForestClassifier(n_estimators=10000, 
                            n_jobs=-1,
                            oob_score=True)
pcr_rf.fit(X, pcr)
pcr_rf_stats = tools.clf_metrics(pcr,
                                 pcr_rf.oob_decision_function_[:, 1],
                                 preds_are_probs=True,
                                 cutpoint=f1_cut,
                                 mod_name='pcr_rf')
pcr_rf_roc = 

'''Mixing in the symptom-based predictors'''
# Recreating some of the other case defs
taste = records.losstastesmell.values
fever = records.fever.values
sob = records.sob.values
chills = records.chills.values
ma = records.ma.values

fc = np.array(fever + chills > 0, dtype=np.uint8)
sfc = np.array(fc + sob == 2, dtype=np.uint8)
smfc = np.array(sob + ma + fc > 0, dtype=np.uint8)

mc1 = np.array(taste + smfc > 0, dtype=np.uint8)
mc4 = np.array(taste + sfc > 0, dtype=np.uint8)

