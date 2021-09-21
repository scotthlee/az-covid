import numpy as np
import pandas as pd
import pickle
import os

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score

from tools import diff_boot_cis, merge_ci_list


# Globals
UNIX = True
DROP_DISC = False
USE_TODAY = False
FIRST_ONLY = False
NO_PREV = False
COMBINED = True
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

# Optionally leaving out folks who were previously positive
if NO_PREV:
    prev = np.where(records.poscovid != 1)[0]
    records = records.iloc[prev, :]

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

if COMBINED:
    combined = np.add(records[symptom_list].values,
                      records[today_list].values)
    records[symptom_list] = np.greater(combined, 0).astype(np.uint8)

case_defs = ['CSTE', 'cc1', 'cc4']
var_list = symptom_list + case_defs

# Getting CIs for the symptoms
dir_files = os.listdir(file_dir)

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
else:
    pcr_ci_out.to_csv(file_dir + 'pcr_cis.csv')
    ant_ci_out.to_csv(file_dir + 'ant_cis.csv')
