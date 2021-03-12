import numpy as np
import pandas as pd

import tools

# Setting the working directory
file_dir = 'C:/Users/yle4/OneDrive - CDC/Documents/projects/hh-transmission/'

# Brining in the two datasets
master = pd.read_csv(file_dir + 'source csv/master.csv')
records = pd.read_csv(file_dir + 'source csv/unf records.csv')

# Making a list of columns to keep
base_list = [
    'study_id_merge', 'hh_id', 'age', 'age_adult', 'ethnicity',
    'sex', 'race', 'test1_pcr', 'test1'
]
symptom_list = [
    'wheeze', 'throat', 'sob', 'nausea', 'myalgia', 'headache',
    'fatigue', 'discomf', 'diarrhea', 'cough', 'chestpain', 
    'abdpain', 'fever_chills', 'nasal_combo', 'tastesmell_combo'
]
case_list = [
    'ili', 'cdc', 'ari', 'cste', 'cli', 'vaccine_a1', 'vaccine_a2', 
    'vaccine_a3', 'vaccine_a_all'
]
all_cols = tools.flatten([base_list, symptom_list, case_list])

# Renaming some of the original columns
records.columns = records.columns.str.replace('_P14', '')
records.columns = records.columns.str.lower()
records.rename(columns={'cdc2':'cdc',
                        'cdc_testing':'cdc_old',
                        'classiccli':'cli',
                        'ari_who_rsv':'ari'}, inplace=True)

# Removing columns we don't need for the analysis
records = records[all_cols]

# Checking out the sero pos data
sero_cols = np.where(['sero' in doc for doc in master.columns.values])[0]
sero_conv = pd.Series(np.array(master.sero_pair == 2, dtype=np.uint8))
sero = master[['study_id_merge', 'sero_pos']]
sero['sero_conv'] = pd.Series(sero_conv)

# Merging the sero data with the study records
records = records.merge(sero, on='study_id_merge')

# Replacing NaNs in the sero_pos column with 2
sero_nan = np.where([np.isnan(res) for res in records.sero_pos])[0]
records.sero_pos[sero_nan] = 2
records.sero_pos = records.sero_pos.astype(np.uint8)

# Casting the original testing columns as numbers
pcr_pos = np.array(records.test1_pcr == 'Yes', dtype=np.uint8)
any_pos = np.array(records.test1 == 'Yes', dtype=np.uint8)
records['pcr_pos'] = pd.Series(pcr_pos)
records['any_pos'] = pd.Series(any_pos)
records = records.drop(['test1_pcr', 'test1'], axis=1)

# Saving the merged records to disk
records.to_csv(file_dir + 'source csv/records.csv', index=False)

# Making a separate dataset for public release
records = records.drop(['age', 'ethnicity', 'race', 'sex'], 
                              axis=1)
records.study_id_merge = records.study_id_merge.str.replace('UT', 'A')
records.study_id_merge = records.study_id_merge.str.replace('WI', 'B')
records.hh_id = records.hh_id.str.replace('UT', 'A')
records.hh_id = records.hh_id.str.replace('WI', 'B')

# Writing the public release version to disk
records.to_csv(file_dir + 'source csv/public records.csv', index=False)
