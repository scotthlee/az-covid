import numpy as np
import pandas as pd
import itertools
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score

import tools


# Globals
USE_TODAY = True
N_BOOT = 100
ROUND = 2

# Importing the original data
file_dir = 'C:/Users/yle4/OneDrive - CDC/Documents/projects/az covid/'
records = pd.read_csv(file_dir + 'records.csv')

# Droping PCR-/ANT+
records.drop(records[(records.Test_Result == 'Negative') &
                     (records.ANTIGEN_RESULT_ == 'Positive')].index,
             inplace=True)

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

# Deciding what variables to include
var_list = symptom_list
if USE_TODAY:
    var_list += today_list

# Loading the arrays and making the targets
X = records[var_list].values.astype(np.uint8)
pcr = records.pcr.values
ant = records.ant.values
mc = records.multi.values
mc_names = ['-/-', '+/-', '+/+']

'''Results for PCR'''
pcr_lgr = LogisticRegression(penalty='none')
pcr_lgr.fit(X, pcr)
pcr_coefs = pd.DataFrame(np.exp(pcr_lgr.coef_)[0],
                         columns=['aOR'])
pcr_coefs['symptom'] = symptom_list
pcr_coefs.to_csv(file_dir + 'pcr_lgr_coefs.csv', index=False)

pcr_rf = RandomForestClassifier(n_estimators=10000, 
                                n_jobs=-1,
                                oob_score=True)
pcr_rf.fit(X, pcr)
pcr_probs = pcr_rf.oob_decision_function_[:, 1]
pcr_gm = tools.grid_metrics(pcr, pcr_probs)
pcr_cut = pcr_gm.cutoff.values[pcr_gm.f1.argmax()]
pcr_pred = tools.threshold(pcr_probs, pcr_cut)
pcr_rf_stats = tools.clf_metrics(pcr,
                                 pcr_probs,
                                 preds_are_probs=True,
                                 cutpoint=pcr_cut,
                                 mod_name='pcr_rf')
pcr_rf_auc = roc_auc_score(pcr, pcr_probs)
pcr_rf_roc = roc_curve(pcr, pcr_probs)

'''Results for antigen'''
ant_lgr = LogisticRegression(penalty='none')
ant_lgr.fit(X, ant)
ant_coefs = pd.DataFrame(np.exp(ant_lgr.coef_)[0],
                         columns=['aOR'])
ant_coefs['symptom'] = symptom_list
ant_coefs.to_csv(file_dir + 'ant_lgr_coefs.csv', index=False)

ant_rf = RandomForestClassifier(n_estimators=10000,
                                n_jobs=-1,
                                oob_score=True)
ant_rf.fit(X, ant)
ant_probs = ant_rf.oob_decision_function_[:, 1]
ant_gm = tools.grid_metrics(ant, ant_probs)
ant_cut = ant_gm.cutoff.values[ant_gm.f1.argmax()]
ant_pred = tools.threshold(ant_probs, ant_cut)
ant_rf_stats = tools.clf_metrics(ant,
                                 ant_probs,
                                 preds_are_probs=True,
                                 cutpoint=ant_cut,
                                 mod_name='ant_rf')

'''Results for multiclass'''
# Multinomial logistic regression
mc_lgr = LogisticRegression(penalty='none',
                            max_iter=5000,
                            n_jobs=-1,
                            multi_class='multinomial')
mc_lgr.fit(X, mc)
mc_coefs = pd.DataFrame(np.exp(mc_lgr.coef_),
                        columns=var_list)
mc_coefs['test_result'] = mc_names
mc_coefs.to_csv(file_dir + 'mc_lgr_coefs.csv', index=False)

# And the big random forest
mc_rf = RandomForestClassifier(n_estimators=10000,
                               n_jobs=-1,
                               oob_score=True)
mc_rf.fit(X, mc)
mc_probs = mc_rf.oob_decision_function_
mc_pred = np.argmax(mc_probs, axis=1)
macro_stats = tools.clf_metrics(mc,
                                mc_probs,
                                preds_are_probs=True,
                                average='macro',
                                mod_name='macro_mc')
weighted_stats = tools.clf_metrics(mc,
                                   mc_probs,
                                   preds_are_probs=True,
                                   average='weighted',
                                   mod_name='weighted_mc')

# Getting the feature importances for comparison with the coefficients
imps = mc_rf.feature_importances_
imp_sort = np.argsort(imps)[::-1]
sorted_features = [symptom_list[i] for i in imp_sort]
feature_out = pd.DataFrame([sorted_features,
                            [imps[i] for i in imp_sort]]).transpose()
feature_out.columns = ['symptom', 'importance']
feature_out.to_csv(file_dir + 'rf_features.csv', index=False)

# Saving the RF predictions to the records file
records['multi_pred'] = mc_pred
records['pcr_pred'] = pcr_pred
records['ant_pred'] = ant_pred
records.to_csv(file_dir + 'records.csv', index=False)

# Rolling up the different results
stats = pd.concat([pcr_rf_stats,
                   ant_rf_stats,
                   macro_stats,
                   weighted_stats], axis=0)
stats.to_csv(file_dir + 'model_stats.csv', index=False)

'''Getting bootstrap CIs for the multinomial regression'''
# Jackknife coefficients
j_samps = tools.jackknife_sample(X)
j_mods = [LogisticRegression(penalty='none',
                             n_jobs=-1,
                             max_iter=5000).fit(X[j], mc[j]) 
          for j in j_samps]
j_coefs = [mod.coef_ for mod in j_mods]
j_coefs = np.array(j_coefs)
j_means = j_coefs.mean(axis=0)
jacks = (j_coefs, j_means)

# Running the bootstrap
seeds = np.random.randint(1, 1e6, N_BOOT)
boots = [tools.boot_sample(X, seed=seed) for seed in seeds]
b_mods = [LogisticRegression(penalty='none', 
                             n_jobs=-1,
                             max_iter=5000).fit(X[b], mc[b])
          for b in boots]
b_coefs = np.array([mod.coef_ for mod in b_mods])

# Getting the CIs
cis = tools.boot_stat_cis(stat=mc_lgr.coef_,
                          jacks=jacks,
                          boots=b_coefs,
                          exp=True)
stat = pd.DataFrame(np.exp(mc_lgr.coef_), 
                    columns=symptom_list).round(ROUND).astype(str)
lower = pd.DataFrame(cis[0], 
                     columns=symptom_list).round(ROUND).astype(str)
upper = pd.DataFrame(cis[1], 
                     columns=symptom_list).round(ROUND).astype(str)
out = stat + ' (' + lower + ', ' + upper + ')'
out['test_result'] = mc_names
out.to_csv(file_dir + 'lgr_cis.csv', index=False)

