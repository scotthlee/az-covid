import numpy as np
import pandas as pd
import pickle
import os

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score

from tools import boot_diff_cis


# Globals
N_BOOT = 100
ROUND = 2
UNIX = True

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
symptoms = [
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

case_defs = ['CSTE', 'cc1', 'cc4']
var_list = symptoms + case_defs

# Getting CIs for the symptoms
if 'pcr_cis.pkl' not in dir_files:
    pcr_cis = [boot_cis(records.pcr,
                        records[var],
                        n=N_BOOT,
                        group=records.PatientID)
               for var in var_list]
    ant_cis = [boot_cis(records.ant,
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
pcr_ci_out = merge_ci_list(pcr_cis, 
                           mod_names=var_list, 
                           round=2)
ant_ci_out = merge_ci_list(ant_cis, 
                           mod_names=var_list, 
                           round=2)

# Writing the confidence intervals to disk
if 'clf_cis.xlsx' not in dir_files:
    writer = pd.ExcelWriter(file_dir + 'clf_cis.xlsx')
    for i, cis in enumerate([pcr_ci_out, ant_ci_out]):
        cis.to_excel(writer, sheet_name=['pcr', 'ant'][i])
    writer.save()

