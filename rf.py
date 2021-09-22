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
TRAIN = False

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
if TRAIN:
    rf = RandomForestClassifier(n_estimators=10000, 
                                n_jobs=-1, 
                                oob_score=True)
    rf.fit(X, pcr)
    probs = rf.oob_decision_function_[:, 1]

    rf_ant = RandomForestClassifier(n_estimators=10000,
                                    n_jobs=-1,
                                    oob_score=True)
    rf_ant.fit(X_ant, pcr)
    ant_probs = rf_ant.oob_decision_function_[:, 1]
else:
    probs = records.rf_symp_prob
    ant_probs = records.rf_sympant_prob

# Using leave-one-out to select thresholds for max J
loo = LeaveOneOut()
j_preds = np.zeros(X.shape[0])
j_ant_preds = np.zeros(X.shape[0])
f1_preds = np.zeros(X.shape[0])
f1_ant_preds = np.zeros(X.shape[0])

# Running the loop; this takes a while
for train, test in loo.split(X, pcr):
    gm = tools.grid_metrics(pcr[train], 
                            probs[train])
    j_cut = gm.cutoff.values[gm.j.argmax()]
    f1_cut = gm.cutoff.values[gm.f1.argmax()]
    j_preds[test] = tools.threshold(probs[test],
                                    j_cut)
    f1_preds[test] = tools.threshold(probs[test],
                                     f1_cut)
    
    ant_gm = tools.grid_metrics(pcr[train], 
                                ant_probs[train])
    j_ant_cut = ant_gm.cutoff.values[ant_gm.j.argmax()]
    f1_ant_cut = ant_gm.cutoff.values[ant_gm.f1.argmax()]
    j_ant_preds[test] = tools.threshold(ant_probs[test],
                                        j_ant_cut)
    f1_ant_preds[test] = tools.threshold(ant_probs[test],
                                         f1_ant_cut)

# Adding the predictions onto the original dataset
records.rf_j_pred = j_preds.astype(np.uint8)
records.rf_f1_pred = f1_preds.astype(np.uint8)
records.rf_ant_j_pred = j_ant_preds.astype(np.uint8)
records.rf_ant_f1_pred = f1_ant_preds.astype(np.uint8)

# And then saving the dataset to disk
records.to_csv(file_dir + 'rf_records.csv', index=False)

