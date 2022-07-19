import numpy as np
import pandas as pd
import pickle
import time
import os

from multiprocessing import Pool

import tools
import multi


# Globals
UNIX = True
USE_TODAY = False
FIRST_ONLY = True
COMBINED = True

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

if FIRST_ONLY:
    date = pd.to_datetime(records.Sample_Collection_Date)
    records['date'] = date
    date_grouped = records.sort_values('date').groupby('PatientID')
    records = date_grouped.head(1)

# Making them combined
pcr = records.pcr.values
ant = records.ant.values

# Running the symptom CIs
var_list += ['ant']
var_cis = [multi.boot_cis(pcr, records[s]) for s in var_list]

# Running the candidate cis
cand_names = ['s95', 'sa95', 's90', 'sa90', 's80', 'sa80']
cand_cis = []
for c in cand_names:
    cand_cis.append(multi.boot_cis(pcr, records[c]))

# Running CIs for the existing definitions
cc1_cis = multi.boot_cis(pcr, records.cc1)
cc1a_cis = multi.boot_cis(pcr, (records.cc1) | (ant == 1))

cc4_cis = multi.boot_cis(pcr, records.cc4)
cc4a_cis = multi.boot_cis(pcr, (records.cc4) | (ant == 1))

cste_cis = multi.boot_cis(pcr, records.cste_new)
cstea_cis = multi.boot_cis(pcr, (records.cste_new) | (ant == 1))

def_names = ['cc1', 'cc1a', 'cc4', 'cc4a', 'cste', 'cstea']
def_cis = [cc1_cis, cc1a_cis, cc4_cis, cc4a_cis, cste_cis, cstea_cis]

# Bundling everything and pickling for later
var_names = [s.replace('_comb', '') for s in var_list]
all_cis = var_cis + cand_cis + def_cis
all_names = var_names + cand_names + def_names
ci_dict = dict(zip(all_names, all_cis))

with open(file_dir + 'pkl/cis.pkl', 'wb') as f:
    pickle.dump(ci_dict, f)

# Writing the CIs to a table
ci_tab = tools.merge_ci_list(all_cis, round=2)
ci_tab['rule'] = all_names
ci_tab.to_csv(file_dir + 'all_cis.csv', index=False)
