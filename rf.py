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
USE_REV = True
COMBINED = True
FIRST_ONLY = False
NO_PREV = False
N_BOOT = 100
ROUND = 2

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
records = pd.read_csv(file_dir + 'records.csv')

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
rf = RandomForestClassifier(n_estimators=10000, 
                            n_jobs=-1, 
                            oob_score=True)
rf.fit(X, pcr)
rf_probs = rf.oob_decision_function_[:, 1]

rf_ant = RandomForestClassifier(n_estimators=10000,
                                n_jobs=-1,
                                oob_score=True)
rf_ant.fit(X_ant, pcr)
rf_ant_probs = rf_ant.oob_decision_function_[:, 1]

# Using leave-one-out to select thresholds for max J
loo = LeaveOneOut()
rf_preds = np.zeros(X.shape[0])
rf_ant_preds = np.zeros(X.shape[0])

for train, test in loo.split(X, pcr):
    rf_gm = tools.grid_metrics(pcr[train], 
                               rf_probs[train])
    rf_cut = rf_gm.cutoff.values[rf_gm.j.argmax()]
    rf_preds[test] = tools.threshold(rf_probs[test], 
                                    rf_cut)
    
    rf_ant_gm = tools.grid_metrics(pcr[train], 
                                   rf_ant_probs[train])
    rf_ant_cut = rf_ant_gm.cutoff.values[rf_ant_gm.j.argmax()]
    rf_ant_preds[test] = tools.threshold(rf_ant_probs[test], 
                                         rf_ant_cut)

