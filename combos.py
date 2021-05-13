import numpy as np
import pandas as pd
import itertools
import pickle
import time
import os

from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score

import tools
import multi

# Globals
UNIX = True
DROP_DISC = True
USE_TODAY = False
COMBINED = False
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

# Making them combined
var_list = [s + '_comb' for s in symptom_list]
X = records[var_list].values
pcr = records.pcr.values
ant = records.ant.values

# Setting up the combinations
m = 5
n = 6
m_combos = [list(combinations(range(n_symp), i)) for i in range(1, m)]

out = []

for n in range(0, n - 1):
    m_out = []
    for combos in m_combos:
        with Pool() as p:
            X_combos = [X[:, c] for c in combos]
            inputs = [(pcr, np.array(np.sum(m, axis=1) + ant > n, 
                                     dtype=np.uint8))
                       for m in X_combos]
            res = pd.concat(p.starmap(tools.clf_metrics, inputs),
                       axis=0)
            m_out.append(res)
    out.append(m_out)

    
    
 

