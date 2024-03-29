import numpy as np
import pandas as pd
import itertools
import pickle
import time
import os

import tools


UNIX = False

# Using multiprocessing on Mac/Linux
if UNIX:
    base_dir = '/Users/scottlee/'
else:
    base_dir = 'C:/Users/yle4/'

# Importing the original data
file_dir = base_dir + 'OneDrive - CDC/Documents/projects/az covid/'
dir_files = os.listdir(file_dir)
records = pd.read_csv(file_dir + 'original_records.csv')

# Cleaning up column names
records.columns = records.columns.str.replace('ever', '')
records.columns.values[7] = 'fever'
records.columns.values[26] = 'fevertoday'

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

# Replacing NaNs in the symptoms today
for i, s in enumerate(today_list):
    records[s].replace(np.nan, 0, inplace=True)

# Making and saving a composite symptom array
s = records[symptom_list].values
t = records[today_list].values
comp = pd.DataFrame(np.array(s + t > 0, dtype=np.uint8), 
             columns=symptom_list)
comp.to_csv(file_dir + 'comp.csv', index=False)

# Copying past symptoms for people with sametoday
same = np.where(records.sametoday == 1)[0]
for i, s in enumerate(today_list):
    records[s][same] = records[symptom_list[i]][same]

# Loading the arrays and making the targets
n_tests = records.shape[0]
pcr = np.array(records.Test_Result == 'Positive', dtype=np.uint8)
ant = np.array(records.ANTIGEN_RESULT_ == 'Positive', dtype=np.uint8)
ml = np.concatenate((pcr.reshape(-1, 1), 
                     ant.reshape(-1, 1)), axis=1)
mc = np.zeros((n_tests), dtype=np.uint8)
mc_names = ['-/-', '+/-', '+/+']
for i in range(n_tests):
    if pcr[i] == 0 and ant[i] == 0:
        mc[i] = 0
    elif pcr[i] == 1 and ant[i] == 1:
        mc[i] = 2
    elif pcr[i] == 1 and ant[i] == 0:
        mc[i] = 1

records['multi'] = pd.Series(mc)
records['pcr'] = pd.Series(pcr)
records['ant'] = pd.Series(ant)

# Combining past and present symptoms
combined = np.add(records[symptom_list].values,
                  records[today_list].values)
X = np.greater(combined, 0).astype(np.uint8)
records[[s + '_comb' for s in symptom_list]] = X

# Making a binary variable for any symptoms
records['any_symp'] = np.array(X.sum(axis=1) > 0,
                               dtype=np.uint8)


# Recreating some of the other case defs
taste = records.losstastesmell.values
fever = records.fever.values
sob = records.sob.values
chills = records.chills.values
ma = records.ma.values

fc = np.array(fever + chills > 0, dtype=np.uint8)
sfc = np.array(fc + sob == 2, dtype=np.uint8)
smfc = np.array(sob + ma + fc > 0, dtype=np.uint8)

mc1 = np.array(taste + smfc > 0, dtype=np.uint8)
mc4 = np.array(taste + sfc > 0, dtype=np.uint8)

records['cc1'] = pd.Series(mc1)
records['cc4'] = pd.Series(mc4)

# The same for today
taste = records.losstastesmelltoday.values
fever = records.fevertoday.values
sob = records.sobtoday.values
chills = records.chillstoday.values
ma = records.muscletoday.values

fc = np.array(fever + chills > 0, dtype=np.uint8)
sfc = np.array(fc + sob == 2, dtype=np.uint8)
smfc = np.array(sob + ma + fc > 0, dtype=np.uint8)

mc1 = np.array(taste + smfc > 0, dtype=np.uint8)
mc4 = np.array(taste + sfc > 0, dtype=np.uint8)

records['cc1_today'] = pd.Series(mc1)
records['cc4_today'] = pd.Series(mc4)

# And the same for combined
taste = records.losstastesmell_comb.values
breath = records.difficultbreath_comb.values
fever = records.fever_comb.values
sob = records.sob_comb.values
chills = records.chills_comb.values
ma = records.ma_comb.values

fc = np.array(fever + chills > 0, dtype=np.uint8)
sfc = np.array(fc + sob + breath + ma >= 3, dtype=np.uint8)
smfc = np.array(sob + ma + fc > 0, dtype=np.uint8)

mc1 = np.array(taste + smfc > 0, dtype=np.uint8)
mc4 = np.array(taste + sfc > 0, dtype=np.uint8)

records['cc1_comb'] = pd.Series(mc1)
records['cc4_comb'] = pd.Series(mc4)

# Redoing the CSTE definition with the latest symptoms
contact = np.array(records.closecontact.values == 1,
                   dtype=np.uint8)

shiver = records.shiver_comb.values
headache = records.headache_comb.values
throat = records.sorethroat_comb.values
naus = records.nauseavom_comb.values
diarrhea = records.diarrhea_comb.values
fatigue = records.fatigue_comb.values
congestion = records.congestion_comb.values
cough = records.cough_comb.values

part1 = np.array(fever + chills + shiver + ma + headache + throat + 
                 naus + diarrhea + fatigue + congestion > 1,
                 dtype=np.uint8)
part2 = np.array(cough + sob + breath + taste > 0, dtype=np.uint8)
clinical = np.array(part1 + part2 > 0, dtype=np.uint8)
crit1 = np.array(clinical + contact == 2, dtype=np.uint8)

records['cste_clin_exp'] = crit1
records['cste_new'] = np.array(crit1 + ant > 0, dtype=np.uint8)


records.to_csv(file_dir + 'records.csv', index=False)
