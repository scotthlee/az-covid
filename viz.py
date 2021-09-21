import numpy as np
import pandas as pd
import seaborn as sns
import os

from matplotlib import pyplot as plt

import tools


# Globals
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
rf_df = pd.read_csv(file_dir + 'rf_records.csv')
symp_df = pd.read_csv(file_dir + 'symp_combo_stats.csv')
ant_df = pd.read_csv(file_dir + 'combo_stats.csv')

# Getting stats for the other combos
taste = tools.clf_metrics(pcr, records.taste_ant) 
cc1 = tools.clf_metrics(pcr, records.cc1_ant)
cc4 = tools.clf_metrics(pcr, records.cc4_ant)
cste = tools.clf_metrics(pcr, records.cste_ant)
def_stats = [taste, cc1, cc4, cste]
def_names = ['taste', 'cc1', 'cc4', 'cste']

prev = np.array(out_df.rel_prev_diff.values - taste.rel_prev_diff.abs().values)
prev = np.array(prev < 0)
j = np.array((taste.j.values - out_df.j.values) < 0)
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
