import numpy as np
import pandas as pd
import itertools
import pickle
import time
import os

from multiprocessing import Pool
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_curve, roc_auc_score

import tools
import multi


# Globals
UNIX = False
DROP_DISC = False
USE_TODAY = False
COMBINED = True
FIRST_ONLY = False
NO_PREV = False
MAX_M = 5

# Using multiprocessing on Mac/Linux
if UNIX:
    base_dir = '/Users/scottlee/'
    from multi import boot_cis
else:
    base_dir = 'C:/Users/yle4/'
    from tools import boot_cis

# Importing the original data
file_dir = base_dir + 'OneDrive - CDC/Documents/projects/az covid/'
dir_files = os.listdir(file_dir)

if TRAIN:
    records = pd.read_csv(file_dir + 'records.csv')
else:
    records = pd.read_csv(file_dir + 'combo_records.csv')

# List of symptom names and case definitions
symptom_list = [
    'fever', 'chills', 'shiver', 'ma', 'congestion',
    'sorethroat', 'cough', 'sob', 'difficultbreath', 'nauseavom',
    'headache', 'abpain', 'diarrhea', 'losstastesmell', 'fatigue'
]

today_list = [
      'fevertoday', 'chillstoday', 'shivertoday', 'muscletoday', 
      'congestiontoday', 'sorethroattoday', 'coughtoday', 'sobtoday', 
      'difficultbreathtoday', 'nauseavomtoday', 'headachetoday', 
      'abpaintoday', 'diarrheatoday', 'losstastesmelltoday', 
      'fatiguetoday'
]

# Deciding what variables to include
var_list = symptom_list

if COMBINED:
    var_list = [s + '_comb' for s in var_list]
else:
    if USE_TODAY:
        var_list += today_list

# Making them combined
X = records[var_list].values
pcr = records.pcr.values
ant = records.ant.values
X_ant = np.concatenate((X, ant.reshape(-1, 1)), axis=1)

# Training RFs with and without antigen as a predictor
for m in range(2, MAX_M):
    rf = RandomForestClassifier(n_estimators=10000,
                                max_depth=m, 
                                n_jobs=-1, 
                                oob_score=True)
    rf.fit(X, pcr)
    probs = rf.oob_decision_function_[:, 1]
    records['rf_symp_' + str(m) + '_probs'] = probs
    
    rf_ant = RandomForestClassifier(n_estimators=10000,
                                    max_depth=m,
                                    n_jobs=-1,
                                    oob_score=True)
    rf_ant.fit(X_ant, pcr)
    ant_probs = rf_ant.oob_decision_function_[:, 1]
    records['rf_sympant_' + str(m) + '_probs'] = probs

# And then saving the dataset to disk
records.to_csv(file_dir + 'rf_records.csv', index=False)

