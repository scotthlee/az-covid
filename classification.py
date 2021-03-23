import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score

import tools


# Globals
N_BOOT = 100
ROUND = 2

# Importing the original data
file_dir = 'C:/Users/yle4/OneDrive - CDC/Documents/projects/az covid/'
dir_files = os.listdir(file_dir)
records = pd.read_csv(file_dir + 'records.csv')

# List of symptom names and case definitions
symptoms = [
    'fever', 'chills', 'shiver', 'ma', 'congestion',
    'sorethroat', 'cough', 'sob', 'difficultbreath', 'nauseavom',
    'headache', 'abpain', 'diarrhea', 'losstastesmell', 'fatigue'
]
case_defs = ['CSTE', 'cc1', 'cc4']
var_list = symptoms + case_defs

# Getting CIs for the symptoms
if 'pcr_cis.pkl' not in dir_files:
    pcr_cis = [tools.boot_cis(records.pcr, 
                              records[var],
                              n=N_BOOT,
                              group=records.PatientID)
               for var in var_list]
    ant_cis = [tools.boot_cis(records.ant, 
                              records[var], 
                              n=N_BOOT,
                              group=records.PatientID)
               for var in var_list]
    
    pickle.dump(pcr_cis, open(file_dir + 'pcr_cis.pkl', 'wb'))
    pickle.dump(ant_cis, open(file_dir + 'ant_cis.pkl', 'wb'))
else:
    pcr_cis = pickle.load(open(file_dir + 'pcr_cis.pkl', 'rb'))
    ant_cis = pickle.load(open(file_dir + 'ant_cis.pkl', 'rb'))

# Making a table wth all the CIs
str_pcr_cis = [c.cis.round(ROUND).astype(str) for c in pcr_cis]
str_pcr_paste = [pd.DataFrame(c.stat + ' (' + c.lower + ', ' + c.upper + ')',
                              columns=[var_list[i]])
                 for i, c in enumerate(str_pcr_cis)]
pcr_ci_out = pd.concat([c.transpose() for c in str_pcr_paste],
                       axis=0)

str_ant_cis = [c.cis.round(ROUND).astype(str) for c in ant_cis]
str_ant_paste = [pd.DataFrame(c.stat + ' (' + c.lower + ', ' + c.upper + ')',
                              columns=[var_list[i]])
                 for i, c in enumerate(str_ant_cis)]
ant_ci_out = pd.concat([c.transpose() for c in str_ant_paste],
                       axis=0)

# Writing the confidence intervals to disk
if 'clf_cis.xlsx' not in dir_files:
    writer = pd.ExcelWriter(file_dir + 'clf_cis.xlsx')
    for i, cis in enumerate([pcr_ci_out, ant_ci_out]):
        cis.to_excel(writer, sheet_name=['pcr', 'ant'][i])
    writer.save()

