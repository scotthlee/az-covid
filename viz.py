import numpy as np
import pandas as pd
import seaborn as sns
import os

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from skimage.color import rgb2gray

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

# Converting to FPR/TPR for ROC plots
combo_df['fpr'] = 1 - combo_df.spec
combo_df.n += 1

# Adding the special columns if they're not there
pcr = rf_df.pcr
ant = rf_df.ant
taste = rf_df.losstastesmell_comb.values

cc1 = rf_df.cc1_comb.values
cc4 = rf_df.cc4_comb.values

rf_df['taste_ant'] = np.array(ant + taste > 0, dtype=np.uint8)
rf_df['cc1_ant'] = np.array(ant + cc1 > 0, dtype=np.uint8)
rf_df['cc4_ant'] = np.array(ant + cc4 > 0, dtype=np.uint8)

# Getting stats for the other combos
def_cols = [
    rf_df.losstastesmell_comb, rf_df.cc1_comb, rf_df.cc4_comb, 
    rf_df.cste_new, rf_df.taste_ant, rf_df.cc1_ant, 
    rf_df.cc4_ant, rf_df.s95, rf_df.sa95, 
    rf_df.s90, rf_df.sa90, rf_df.s80, 
    rf_df.sa80
]
def_stats = [tools.clf_metrics(pcr, d) for d in def_cols]
def_stats.append(tools.clf_metrics(pcr, ant))
def_names = [
    'taste', 'Reses1', 'Reses4',
    'CSTE', 'taste+ant', 'Reses1+ant', 
    'Reses4+ant', 's95', 's95+ant',
    's90', 's90+ant', 's80',
    's80+ant', 'ant_alone'
]

# Doing the ROC curves
symp_rocs = [roc_curve(pcr, rf_df['symp_' + str(i) + '_prob'])
             for i in range(1, 6)]
ant_rocs = [roc_curve(pcr, rf_df['ant_' + str(i) + '_prob'])
             for i in range(1, 6)]

# Plotting combo and RF ROCs as a function of n and m
gr = sns.color_palette('gray_r')
cr = sns.color_palette('crest')
cb = sns.color_palette('colorblind')
sns.set_style('ticks', rc={'fontname': 'arial', 'fontsize': 8})

facet_kws = {'sharex': False, 'sharey': False}
rp = sns.relplot(x='fpr', 
                 y='sens', 
                 hue='m', 
                 col='n', 
                 data=combo_df,
                 col_wrap=3,
                 height=3.1,
                 aspect=.74,
                 kind='scatter',
                 palette='crest',
                 legend=False,
                 facet_kws=facet_kws)

titles = ['A)', 'B)', 'C)', 'D)', 'E)']
for n, ax in enumerate(rp.axes):
    ax.set(ylabel='Sensitivity', 
           xlabel='1 - Specificity',
           xlim=(0, 0.4),
           ylim=(0, 1))
    ax.plot(symp_rocs[n][0], 
            symp_rocs[n][1],
            alpha=0.5,
            color=cr[4])
    ax.plot(ant_rocs[n][0], 
            ant_rocs[n][1],
            alpha=0.5,
            color=cr[4])
    ax.set_title('')
    ax.set_title(titles[n], position=(-.2, 1.5))

dpi = 1000
rp.fig.set_dpi(dpi)
rp.fig.canvas.draw()
img = np.frombuffer(rp.fig.canvas.tostring_rgb(),
                    dtype=np.uint8)
width, height = rp.fig.get_size_inches() * rp.fig.get_dpi()
img = img.reshape(int(height), int(width), 3)

plt.imsave(file_dir + '/final/figure_1.pdf',
           dpi=dpi,
           arr=rgb2gray(img),
           cmap=plt.cm.gray)
plt.imsave(file_dir + '/final/figure_1.svg',
           dpi=dpi,
           arr=rgb2gray(img),
           cmap=plt.cm.gray)

# Plotting case definitions against combos
sns.set_style('ticks', rc={'fontname': 'arial', 'fontsize': 8})
rcParams['figure.figsize'] = (7, 6)
plt.figure(dpi=1000, figsize=(7, 6))
sp = sns.scatterplot(x='fpr', 
                     y='sens', 
                     data=combo_df,
                     hue='type',
                     alpha=0.4,
                     palette='gray',
                     legend=False,
                     linewidth=0)
sp.set(xlim=(0, 0.4), ylim=(0,1))

def_abbrs = ['A', 'B', 'C', 'D', 'E',
             'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N']

for i, df in enumerate(def_stats):
    label = def_names[i]
    abbr = def_abbrs[i]
    if ('ant' in label) or (label == 'CSTE'):
        col = gr[5]
    else:
        col = gr[1]
    
    x_off = .004
    y_off = -.008
    
    if label in ['Reses4+ant', 's95']:
        y_off = .005
    
    if label in ['Reses4', 's95+ant']:
        y_off -= .010
        
    fpr = 1 - df.spec
    tpr = df.sens
    plt.scatter(x=fpr, y=tpr,
                s=20, 
                color=gr[5],
                edgecolors='white',
                linewidth=0.9)
    plt.text(x=fpr + x_off, 
             y=tpr + y_off,
             color=gr[5], 
             s=abbr,
             bbox=dict(boxstyle='square,pad=0.05',
                       fc='lightgray',
                       ec='white'),
             fontweight='semibold')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.tight_layout()
plt.savefig(file_dir + '/final/figure_2.pdf',
            dpi=dpi,
            format='pdf',
            bbox_inches='tight')
plt.savefig(file_dir + '/final/figure_2.svg',
            dpi=dpi,
            format='svg',
            bbox_inches='tight')
plt.show()
