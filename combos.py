import numpy as np
import pandas as pd
import itertools
import pickle
import time
import os

from multiprocessing import Pool
from itertools import combinations

import tools
import multi


# Globals
UNIX = False
DROP_DISC = False
USE_TODAY = False
USE_REV = True
COMBINED = True
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

# Making a reversed X
if USE_REV:
    X_rev = -X + 1
    X = np.concatenate((X, X_rev), axis=1)

# Setting up the combinations
n = 5
n_symp = X.shape[1]
m_combos = [list(combinations(range(n_symp), i)) for i in range(1, n+1)]

# Dropping impossible symptom pairings
if USE_REV:
    clashes = [[i, i + 15] for i in range(n_symp)]
    keepers = [[np.sum([c[0] in l and c[1] in l
                        for c in clashes]) == 0
                for l in combos]
             for combos in m_combos]
    m_combos = [[c for j, c in enumerate(combos) if keepers[i][j]]
                for i, combos in enumerate(m_combos)]

out = []

# Running the loop
for i, combos in enumerate(m_combos):
    c_out = []
    X_combos = [X[:, c] for c in combos]
    for m in range(len(combos[0])):
        inputs = [(pcr, np.array(np.array(np.sum(x, axis=1) > m,
                                          dtype=np.uint8) + ant > 0, 
                                 dtype=np.uint8))
                   for x in X_combos]
        with Pool() as p:
            res = pd.concat(p.starmap(tools.clf_metrics, inputs),
                       axis=0)
        c_out.append(res)
    out.append(c_out)

# Getting the combo names
combo_names = [[' '.join([symptom_list[i] for i in c])
                for c in combos]
               for combos in m_combos]

# Filling in the symptom names
for i, dfs in enumerate(out):
    for j in range(len(dfs)):
        dfs[j]['rule'] = [str(j + 1) + ' of ' + s
                          for s in combo_names[i]]

# Smooshing the result dfs together
out_df = pd.concat([pd.concat(df, axis=0) for df in out], axis=0)
out_df.to_csv(file_dir + 'combo_stats.csv', index=False)

# Getting stats for the other combos
taste = tools.clf_metrics(pcr, records.taste_ant) 
cc1 = tools.clf_metrics(pcr, records.cc1_ant)
cc4 = tools.clf_metrics(pcr, records.cc4_ant)
cste = tools.clf_metrics(pcr, records.cste_ant)
def_stats = [taste, cc1, cc4, cste]
def_names = ['taste', 'cc1', 'cc4', 'cste']
   
prev = np.array(out_df.rel_prev_diff - taste.rel_prev_diff.abs())
prev = np.array(prev < 0)
j = np.array((taste.j - out_df.j) < 0)
out_df['better on prev'] = prev
out_df['better on j'] = j

# Plotting TPR and FPR with color by prevalence accuracy
cb = sns.color_palette('colorblind')
sns.set_style('dark')
sns.set_palette('colorblind')
ax = sns.scatterplot(x=(1 - out_df.spec),
                y=out_df.sens,
                size=out_df['better on prev'],
                hue=out_df['better on j'],
                palette='colorblind',
                s=10,
                alpha=.4) 
ax.set(xlabel='false positive rate',
       ylabel='true positive rate')

# Adding points for the other combinations
for i, df in enumerate(def_stats):
    fpr = 1 - df.spec
    tpr = df.sens
    plt.scatter(x=fpr, y=tpr, color=cb[2])
    plt.text(x=fpr, y=tpr, s=def_names[i])

plt.show()
