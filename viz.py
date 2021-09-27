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

# Converting specificity to FPR to make plotting easier
combo_df['fpr'] = 1 - combo_df.spec

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

# Doing the ROC curves
symp_rocs = [roc_curve(pcr, rf_df['symp_' + str(i) + '_prob'])
             for i in range(1, 6)]
ant_rocs = [roc_curve(pcr, rf_df['ant_' + str(i) + '_prob'])
             for i in range(1, 6)]

# Plotting combo and RF ROCs as a function of n and m
sns.set_style('darkgrid')
cr = sns.color_palette('crest')
cb = sns.color_palette('colorblind')

rp = sns.relplot(x='fpr', 
                 y='sens', 
                 hue='m', 
                 col='n', 
                 data=combo_df,
                 kind='scatter',
                 palette='crest')
rp.fig.set_tight_layout(True)

for n, ax in enumerate(rp.axes[0]):
    ax.plot(symp_rocs[n][0], 
            symp_rocs[n][1],
            alpha=0.5,
            color=cr[4])
    ax.plot(ant_rocs[n][0], 
            ant_rocs[n][1],
            alpha=0.5,
            color=cr[4])
    ax.set_xlim((0, 0.4))

plt.show()

# Plotting case definitions against combos
sns.scatterplot(x='fpr', 
                y='sens', 
                data=combo_df, 
                hue='type',
                alpha=0.4,
                palette='colorblind')

for i, df in enumerate(def_stats):
    if 'ant' in def_names[i]:
        col = cb[2]
    else:
        col = cb[2]
    
    fpr = 1 - df.spec
    tpr = df.sens
    plt.scatter(x=fpr, y=tpr, color=col)
    plt.text(x=fpr, y=tpr, s=def_names[i])

plt.tight_layout()
plt.show()

'''
prev = np.array(combo_df.rel_prev_diff.values - taste.rel_prev_diff.abs().values)
prev = np.array(prev < 0)
j = np.array((taste.j.values - combo_df.j.values) < 0)
combo_df['better on prev'] = prev
combo_df['better on j'] = j
'''

