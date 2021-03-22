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


# Importing the original data
file_dir = 'C:/Users/yle4/OneDrive - CDC/Documents/projects/az covid/'
records = pd.read_csv(file_dir + 'records.csv')

# List of symptom names and case definitions
symptom_list = [
    'fever', 'chills', 'shiver', 'ma', 'congestion',
    'sorethroat', 'cough', 'sob', 'difficultbreath', 'nauseavom',
    'headache', 'abpain', 'diarrhea', 'losstastesmell', 'fatigue'
]

today_list = [
      'ftoday', 'chillstoday', 'shivertoday', 'muscletoday', 
      'congestiontoday', 'sorethroattoday', 'coughtoday', 'sobtoday', 
      'difficultbreathtoday', 'nauseavomtoday', 'headachetoday', 
      'abpaintoday', 'diarrheatoday', 'losstastesmelltoday', 
      'fatiguetoday'
]

# Loading the arrays and making the targets
X = pickle.load(open(file_dir + 'X.pkl', 'rb'))
pcr = np.array(records.poscovid == 1, dtype=np.uint8)
ant = np.array(records.ANTIGEN_RESULT_ == 'Positive', dtype=np.uint8)
ml = np.concatenate((pcr.reshape(-1, 1), 
                     ant.reshape(-1, 1)), axis=1)
mc = np.zeros((X.shape[0]), dtype=np.uint8)
for i in range(X.shape[0]):
    if pcr[i] == 0 and ant[i] == 0:
        mc[i] = 0
    elif pcr[i] == 1 and ant[i] == 1:
        mc[i] = 3
    elif pcr[i] == 1 and ant[i] == 0:
        mc[i] = 1
    if pcr[i] == 0 and ant[i] == 1:
        mc[i] = 2

# Splitting into traininig and test sets
train, test = train_test_split(range(X.shape[0]),
                               test_size=0.4,
                               stratify=pcr,
                               random_state=2021)
val, test = train_test_split(test,
                             test_size=0.5,
                             stratify=pcr[test],
                             random_state=2021)

'''Results for PCR'''
# Running a few baseline models
lgr = LogisticRegression(penalty='none')
lgr.fit(X, pcr)
coefs = pd.DataFrame(np.exp(lgr.coef_)[0],
                     columns=['aOR'])
coefs['symptom'] = symptom_list
coefs.to_csv(file_dir + 'pcr_lgr_coefs.csv', index=False)

pcr_rf = RandomForestClassifier(n_estimators=10000, 
                                n_jobs=-1,
                                oob_score=True)
pcr_rf.fit(X, pcr)
rf_probs = pcr_rf.oob_decision_function_[:, 1]
pcr_gm = tools.grid_metrics(pcr, rf_probs)
f1_cut = pcr_gm.cutoff.values[pcr_gm.f1.argmax()]
pcr_rf_stats = tools.clf_metrics(pcr,
                                 rf_probs,
                                 preds_are_probs=True,
                                 cutpoint=f1_cut,
                                 mod_name='pcr_rf')
pcr_rf_auc = roc_auc_score(pcr, rf_probs)
pcr_rf_roc = roc_curve(pcr, rf_probs)

'''Results for antigen'''
lgr = LogisticRegression(penalty='none')
lgr.fit(X, ant)
coefs = pd.DataFrame(np.exp(lgr.coef_)[0],
                     columns=['aOR'])
coefs['symptom'] = symptom_list
coefs.to_csv(file_dir + 'ant_lgr_coefs.csv', index=False)

ant_rf = RandomForestClassifier(n_estimators=10000,
                                n_jobs=-1,
                                oob_score=True)
ant_rf.fit(X, ant)
ant_probs = ant_rf.oob_decision_function_[:, 1]
ant_gm = tools.grid_metrics(ant, ant_probs)
ant_cut = ant_gm.cutoff.values[ant_gm.f1.argmax()]
ant_rf_stats = tools.clf_metrics(ant,
                                 ant_probs,
                                 preds_are_probs=True,
                                 cutpoint=ant_cut,
                                 mod_name='ant_rf')

'''Results for multiclass'''
# Multinomial logistic regression
lgr = LogisticRegression(penalty='none', 
                         multi_class='multinomial')
lgr.fit(X, mc)
coefs = pd.DataFrame(np.exp(lgr.coef_),
                     columns=symptom_list)
coefs['test_class'] = ['-/-', '+/-', '-/+', '+/+']
coefs.to_csv(file_dir + 'mc_lgr_coefs.csv', index=False)

# And the big random forest
mc_rf = RandomForestClassifier(n_estimators=10000,
                               n_jobs=-1,
                               oob_score=True)
mc_rf.fit(X, mc)
mc_probs = mc_rf.oob_decision_function_
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

# Rolling up the different results
stats = pd.concat([pcr_rf_stats,
                   ant_rf_stats,
                   macro_stats,
                   weighted_stats], axis=0)
stats.to_csv(file_dir + 'model_stats.csv', index=False)

'''Getting bootstrap CIs for the multinomial regression'''
# Number of pseudosamples
n_boot = 100

# Jackknife coefficients
j_samps = tools.jackknife_sample(X)
j_mods = [LogisticRegression(penalty='none').fit(X[j], mc[j]) 
          for j in j_samps]
j_coefs = [mod.coef_ for mod in j_mods]
j_coefs = np.array(j_coefs)
j_means = j_coefs.mean(axis=0)
jacks = (j_coefs, j_means)

# Running the bootstrap
seeds = np.random.randint(1, 1e6, n_boot)
boots = [tools.boot_sample(X, seed=seed) for seed in seeds]
b_mods = [LogisticRegression(penalty='none', n_jobs=-1).fit(X[b], mc[b])
          for b in boots]
b_coefs = np.array([mod.coef_ for mod in b_mods])

# Getting the CIs
round = 2
cis = tools.boot_stat_cis(stat=lgr.coef_,
                          jacks=jacks,
                          boots=b_coefs,
                          exp=True)
stat = pd.DataFrame(np.exp(lgr.coef_), 
                    columns=symptom_list).round(round).astype(str)
lower = pd.DataFrame(cis[0], 
                     columns=symptom_list).round(round).astype(str)
upper = pd.DataFrame(cis[1], 
                     columns=symptom_list).round(round).astype(str)
out = stat + ' (' + lower + ', ' + upper + ')'
out['result'] = ['-/-', '+/-', '-/+', '+/+']
out.to_csv(file_dir + 'cis.csv', index=False)


'''Looking at specific symptoms as predictors'''
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

