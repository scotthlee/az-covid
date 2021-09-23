import numpy as np
import pandas as pd
import seaborn as sns
import os

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

import tools


# Globals
UNIX = False

# Using multiprocessing on Mac/Linux
if UNIX:
    base_dir = '/Users/scottlee/'
    from multi import boot_cis
else:
    base_dir = 'C:/Users/yle4/'
    from tools import boot_cis

# Importing the original data
file_dir = base_dir + 'OneDrive - CDC/Documents/projects/az covid/'
rf_df = pd.read_csv(file_dir + 'rf_records.csv')
combo_df = pd.read_csv(file_dir + 'combo_stats.csv')

# Getting stats for the other combos
pcr = rf_df.pcr
ant = rf_df.ant
def_cols = [rf_df.losstastesmell, rf_df.cc1, rf_df.cc4, 
            rf_df.CSTE, rf_df.taste_ant, rf_df.cc1_ant, 
            rf_df.cc4_ant, rf_df.cste_ant]
def_stats = [tools.clf_metrics(pcr, d) for d in def_cols]
def_stats.append(tools.clf_metrics(pcr, ant))
def_names = ['taste', 'cc1', 'cc4',
             'cste', 'taste+ant', 'cc1+ant', 
             'cc4+ant', 'cste+ant', 'ant_alone']

prev = np.array(combo_df.rel_prev_diff.values - taste.rel_prev_diff.abs().values)
prev = np.array(prev < 0)
j = np.array((taste.j.values - combo_df.j.values) < 0)
combo_df['better on prev'] = prev
combo_df['better on j'] = j

# Doing the ROC curves
symp_roc = roc_curve(pcr, rf_df.rf_symp_prob)
ant_roc = roc_curve(pcr, rf_df.rf_sympant_prob)

# Plotting TPR and FPR with color by prevalence accuracy
cb = sns.color_palette('colorblind')
sns.set_style('dark')
sns.set_palette('colorblind')
ax = sns.scatterplot(x=(1 - combo_df.spec),
                     y=combo_df.sens,
                     hue=combo_df.type,
                     palette='colorblind',
                     s=10,
                     alpha=.4) 
ax.set(xlabel='false positive rate',
       ylabel='true positive rate')

# Adding the ROC curves
sns.lineplot(x=symp_roc[0], 
             y=symp_roc[1], 
             palette='colorblind')
sns.lineplot(x=ant_roc[0], 
             y=ant_roc[1], 
             palette='colorblind')

# Adding points for the other combinations
for i, df in enumerate(def_stats):
    fpr = 1 - df.spec
    tpr = df.sens
    plt.scatter(x=fpr, y=tpr, color=cb[2])
    plt.text(x=fpr, y=tpr, s=def_names[i])

plt.title('combos and the RF in ROC space')
plt.show()
