import numpy as np
import pandas as pd
import seaborn as sns
import os

from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import roc_curve
from skimage.color import rgb2gray

import tools


def savefig(fg, fpath='', gray=True, dpi=1000):
    fg.set_dpi(dpi)
    fg.canvas.draw()
    img = np.frombuffer(fg.canvas.tostring_rgb(),
                        dtype=np.uint8)
    width, height = fg.get_size_inches() * fg.get_dpi()
    img = img.reshape(int(height), int(width), 3)
    if gray:
        img = rgb2gray(img)
    plt.imsave(fpath,
               dpi=dpi,
               arr=img,
               cmap=plt.cm.gray)
    plt.imsave(fpath,
               dpi=dpi,
               arr=img,
               cmap=plt.cm.gray)
    return

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

# Making the panel plot
subs = False
if subs:
    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(7, 3.2 * 2)
else:
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3.2 * .7, 3.2)

for i, n in enumerate(combo_df.n.unique()):
    samp = combo_df[combo_df.n == n]
    i_ = i
    if subs:
        j = 0
        if i >= 3:
            j = 1
            i_ = i - 3
        ax = axes[j][i_]
        sp = sns.scatterplot(x='fpr', 
                             y='sens',
                             hue='m',
                             hue_norm=(0, 4),
                             data=samp,
                             palette='crest',
                             legend=False,
                             ax=ax)
    else:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(3.2 * .7, 3.2)
        sns.scatterplot(x='fpr',
                        y='sens',
                        hue='m',
                        hue_norm=(0, 4),
                        data=samp,
                        palette='crest',
                        legend=False,
                        ax=ax)
    ax.set(ylabel='Sensitivity', 
           xlabel='1 - Specificity',
           xlim=(0, 0.4),
           ylim=(0, 1))
    ax.plot(symp_rocs[i][0], 
            symp_rocs[i][1],
            alpha=0.5,
            color=cr[4])
    ax.plot(ant_rocs[i][0], 
            ant_rocs[i][1],
            alpha=0.5,
            color=cr[4])
    ax.set_title('')
    ax.text(-.14, 1.1, titles[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not subs:
        plt.tight_layout()
        fn = 'Figure 1' + titles[i][0]
        savefig(fig, file_dir + fn + '.svg')
        savefig(fig, file_dir + fn + '.pdf')

axes[1][2].axis('off')
plt.tight_layout()

# Saving the main plot
savefig(fig, file_dir + 'Figure 1x.pdf')
savefig(fig, file_dir + 'Figure 1.svg')

# Plotting case definitions against combos
sns.set_style('ticks', rc={'fontname': 'arial', 'fontsize': 8})
plt.figure(figsize=(7, 6))
sp = sns.scatterplot(x='fpr', 
                     y='sens', 
                     data=combo_df,
                     hue='type',
                     alpha=0.4,
                     palette='gray',
                     legend=False,
                     linewidth=0)
sp.set(xlim=(0, 0.4), ylim=(0,1))

#def_abbrs = ['A', 'B', 'C', 'D', 'E',
#             'F', 'G', 'H', 'I', 'J',
#             'K', 'L', 'M', 'N']
def_abbrs = ['T', 'R1', 'R4', 'C', 'Ta',
             'R1a', 'R4A', 's95', 's95a', 's90',
             's90a', 's80', 's80a', 'a']

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
