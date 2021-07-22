import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import itertools
import pickle
import time
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score

import tools


# Globals
UNIX = True
DROP_DISC = False
USE_TODAY = False
COMBINED = True
CASE_DEF = True
FIRST_ONLY = False
NO_PREV = False
N_BOOT = 500
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

# Optionally droping PCR-/ANT+
if DROP_DISC:
    disc = np.where((records.Test_Result == 'Negative') & 
                    (records.ANTIGEN_RESULT_ == 'Positive'))[0]
    records.drop(disc, axis=0, inplace=True)

# Optionally keeping only the first test results
if FIRST_ONLY:
    g = records.groupby('PatientID')
    date = 'Sample_Collection_Date'
    records = g.apply(lambda x: x.sort_values(ascending=True,
                                              by=date).head(1))
    file_dir += 'first_only/'
    df_name += '_first'

# Optionally leaving out folks who were previously positive
if NO_PREV:
    prev = np.where(records.poscovid != 1)[0]
    records = records.iloc[prev, :]
    df_name += '_noprev'

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

# Optionally adding the complex case definitions
if CASE_DEF:
    var_list += ['CSTE', 'cc1', 'cc4']

X = records[var_list].values

# Making the columns for the table
totals = X.sum(axis=0)
percents = totals / X.shape[0] * 100

symp_ids = [np.where(records[s] == 1)[0] for s in var_list]
multi_ids = [np.where(records.multi == m)[0] 
             for m in records.multi.unique()]

ant_totals = [records.ant.values[s].sum() for s in symp_ids]
ant_percents = ant_totals / totals * 100
pcr_totals = [records.pcr.values[s].sum() for s in symp_ids]
pcr_percents = pcr_totals / totals * 100

pp_totals = [np.sum(records.multi.values[s] == 2)
                  for s in symp_ids]
pp_percents = pp_totals / totals * 100
nn_totals = [np.sum(records.multi.values[s] == 0)
                  for s in symp_ids]
nn_percents = nn_totals / totals * 100
np_totals = [np.sum(records.multi.values[s] == 1)
                  for s in symp_ids]
np_percents = np_totals / totals * 100
pn_totals = [np.sum(records.multi.values[s] == 3)
                  for s in symp_ids]
pn_percents = pn_totals / totals * 100

# Writing to disk
out = pd.DataFrame([
    totals, percents, ant_totals,
    ant_percents, pcr_totals, pcr_percents,
    pp_totals, pp_percents, nn_totals,
    nn_percents, np_totals, np_percents,
    pn_totals, pn_percents
]).T
out.index = var_list
out.to_csv(file_dir + 'table1.csv')
