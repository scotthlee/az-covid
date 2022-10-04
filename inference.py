import numpy as np
import pandas as pd
import pickle
import os

import tools
import multi


# Globals
UNIX = True
USE_TODAY = False
FIRST_ONLY = False
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
records = pd.read_csv(file_dir + 'rf_records.csv')

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
ids = records.PatientID.values

# Making a few constructed variables
N = records.shape[0]
records['any_ever'] = pd.Series(records[var_list].sum(1) > 0).astype(np.uint8)
records['any_now'] = pd.Series(records[today_list].sum(1) > 0).astype(np.uint8)
records['tp'] = pd.Series(pcr + ant == 2).astype(np.uint8)
records['tn'] = pd.Series(pcr + ant == 0).astype(np.uint8)
records['fp'] = pd.Series((ant == 1) & (pcr == 0)).astype(np.uint8)
records['fn'] = pd.Series((ant == 0) & (pcr == 1)).astype(np.uint8)

# Making a version with only the first visits for doing the demog tables
date = pd.to_datetime(records.Sample_Collection_Date)
records['date'] = date
date_grouped = records.sort_values('date').groupby('PatientID')
first_records = date_grouped.head(1)
if FIRST_ONLY:
    records = first_records

# Making a supplemental table 1
dem_cols = ['Patient_Sex','AGEGRP', 'Patient_Race',
              'Patient_Ethnicity']
dem_tabs = [pd.crosstab(first_records[c], 'n') for c in dem_cols]
dem_tabs = pd.concat(dem_tabs, axis=0)
dem_tabs['%'] = dem_tabs.n / first_records.shape[0]
dem_tabs.to_csv(file_dir + 'dem_tabs.csv')

test_cols = ['any_now', 'any_ever', 'poscovid', 
             'closecontact', 'ant', 'pcr']
test_tabs = [pd.crosstab(records[c], 'n') for c in test_cols]
test_tabs = pd.concat(test_tabs, axis=0)
test_tabs['%'] = test_tabs.n / N
test_tabs.to_csv(file_dir + 'test_tabs.csv')

# Running the symptom CIs
var_list += ['ant']
var_cis = [multi.boot_cis(targets=pcr, 
                          guesses=records[s], 
                          sample_by=ids) for s in var_list]

# Running the candidate cis
cand_names = ['s95', 'sa95', 's90', 'sa90', 's80', 'sa80']
cand_cis = []
for c in cand_names:
    cand_cis.append(multi.boot_cis(targets=pcr, 
                                   guesses=records[c],
                                   sample_by=ids))

# Running CIs for the existing definitions
cc1_cis = multi.boot_cis(pcr, records.cc1, 
                         sample_by=ids)
cc1a_cis = multi.boot_cis(pcr, (records.cc1) | (ant == 1),
                          sample_by=ids)

cc4_cis = multi.boot_cis(pcr, records.cc4,
                          sample_by=ids)
cc4a_cis = multi.boot_cis(pcr, (records.cc4) | (ant == 1),
                          sample_by=ids)

cste_cis = multi.boot_cis(pcr, records.cste_clin_exp,
                          sample_by=ids)
cstea_cis = multi.boot_cis(pcr, (records.cste_new) | (ant == 1),
                          sample_by=ids)

def_names = ['cc1', 'cc4', 'cste', 'cc1a', 'cc4a', 'cstea']
def_cis = [cc1_cis, cc4_cis, cste_cis, cc1a_cis, cc4a_cis, cstea_cis]

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
