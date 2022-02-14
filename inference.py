import numpy as np
import pandas as pd
import time
import os

from multiprocessing import Pool

import tools
import multi


# Globals
UNIX = True
USE_TODAY = False
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

# Making them combined
pcr = records.pcr.values
ant = records.ant.values

# Pulling out some single symptoms
taste = records.losstastesmell_comb.values
fever = records.fever_comb.values
head = records.headache_comb.values
chills = records.chills_comb.values
ma = records.ma_comb.values
shiver = records.shiver_comb.values
breath = records.difficultbreath_comb.values
cough = records.cough_comb.values
fatigue = records.fatigue_comb.values

# Reconstructing some candidate definitions
s95 = np.array(head + fever + taste >= 2, dtype=np.uint8)
sa95 = np.array((shiver + head + taste >= 2) | (ant == 1),
                dtype=np.uint8)

s90 = np.array(fever + chills + taste >= 1, dtype=np.uint8)
sa90 = np.array((breath + chills + taste >=1) | (ant == 1),
                dtype=np.uint8)

s80 = np.array(fever + chills + cough + ma + taste >= 1, dtype=np.uint8)
sa80 = np.array((fever + chills + fatigue + ma + taste >= 1) | (ant == 1),
                dtype=np.uint8)

# Running the candidate cis
candidates = [s95, sa95, s90, sa90, s80, sa80]
cand_names = ['s95', 'sa95', 's90', 'sa90', 's80', 'sa80']
cand_cis = []
for c in candidates:
    cand_cis.append(multi.boot_cis(pcr, c))

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
all_cis = cand_cis + def_cis
all_names = cand_names + def_names
ci_dict = dict(zip(all_names, all_cis))

with open(file_dir + 'pkl/cis.pkl', 'wb') as f:
    pickle.dump(ci_dict, f)

# Writing the CIs to a table
ci_tab = tools.merge_ci_list(all_cis, round=2)
ci_tab['rule'] = all_names
ci_tab.to_csv(file_dir + 'compound_cis.csv', index=False)


