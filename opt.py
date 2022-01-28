import numpy as np
import scipy as sp
import pandas as pd
import time
import os

from multiprocessing import Pool
from scipy.special import expit

import tools
import multi


# Globals
UNIX = False
USE_TODAY = False
COMBINED = True

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
records = pd.read_csv(file_dir + 'combo_records.csv')

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

if COMBINED:
    var_list = [s + '_comb' for s in var_list]
else:
    if USE_TODAY:
        var_list += today_list

var_list += ['ant']

# Making them combined
y = records.pcr.values
X = records[var_list].values

pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]

X = np.concatenate([X[pos], X[neg]], axis=0)
y = np.concatenate([y[pos], y[neg]], axis=0)

xp = X[:len(pos), :]
xn = X[len(pos):, :]

# Defining some useful functions
def smash_log(x, B=10, d=0):
    return 1 / (1 + np.exp(-x * B)) - d

def j(z, xp, xn, m):
    z = np.round(z)
    tpr = np.sum(np.dot(xp, z) >= m) / xp.shape[0]
    fpr = np.sum(np.dot(xn, z) >= m) / xn.shape[0]
    print(tpr, fpr)
    return tpr - fpr

def j_exp(z, xp, xn, a=1, b=1):
    m = z[-1]
    z = z[:-1]
    tpr = smash_log(np.dot(xp, z) - m).sum() / xp.shape[0]
    fpr = smash_log(np.dot(xn, z) - m).sum() / xn.shape[0]
    return -1 * (a*tpr - b*fpr)

# Setting up the simple NLP
m = 1
n = 4
Ns = 16
bnds = ((0, 1),) * 16
bnds += ((1, 16),)
init = np.zeros(17)

# Setting up the optional constraints for m and n
nA = np.concatenate([np.ones(Ns),
                     np.zeros(1)])
mA = np.concatenate([np.zeros(Ns),
                     np.ones(1)])
ncon = sp.optimize.LinearConstraint(nA, lb=1, ub=n)
mcon = sp.optimize.LinearConstraint(mA, lb=1, ub=m)
                                    
# Running the program
start = time.time()
opt = sp.optimize.minimize(
    fun=j_exp,
    x0=init,
    args=(xp, xn),
    bounds=bnds
)
end = time.time()
end - start

good = opt.x.round()[:-1]
good_cols = np.where(good == 1)[0]
good_s = [var_list[i] for i in good_cols]
good_s
j(good, xp, xn, opt.x.round()[-1])

# Now trying the compound program
def j_exp_comp(z, xp, xn, Nc=2, a=1, b=1):
    ns = xp.shape[1]
    m = z[-Nc:]
    z = z[:-Nc]
    zmat = z.reshape((ns, Nc), order='F')
    
    p_hits = smash_log(smash_log(np.dot(xp, zmat) - m).sum(1) - .5).sum()
    n_hits = smash_log(smash_log(np.dot(xn, zmat) - m).sum(1) - .5).sum()
    
    tpr = p_hits / xp.shape[0]
    fpr = n_hits / xn.shape[0] 
    
    return -1 * (a*tpr - b*fpr)

Nc = 3
z_bnds = ((0, 1),) * Ns * Nc
m_bnds = ((0, 16),) * Nc
bnds = z_bnds + m_bnds

# Constraint so that no symptom appears in more than one combo
z_con_mat = np.concatenate([np.identity(Ns)] * Nc, axis=1)
m_con_mat = np.zeros((Ns, Nc))
nmax_mat = np.concatenate([z_con_mat, m_con_mat], axis=1)
nmax_cons = sp.optimize.LinearConstraint(nmax_mat, lb=0, ub=1)

# Constraint so that m >- n for any combo
z_c_rows = [np.ones(Ns)] * Nc
z_c_mat = np.zeros((Nc, Ns * Nc))
for i, r in enumerate(z_c_rows):
    start = i * Ns
    end = start + Ns
    z_c_mat[i, start:end] = r

z_c_mat = np.concatenate([z_c_mat, np.identity(Nc) * -1],
                         axis=1)
mn_cons = sp.optimize.LinearConstraint(z_c_mat, lb=0, ub=np.inf)

# Constraint that at least one combo must have m >= 1
m_sum = np.concatenate([np.zeros(Ns * Nc),
                        np.ones(Nc)])
m_sum_cons = sp.optimize.LinearConstraint(m_sum, lb=1, ub=np.inf)

#init = np.zeros(len(bnds))
#init = np.ones(len(bnds))
init = np.random.choice([0, 1], len(bnds))

start = time.time()
opt = sp.optimize.minimize(
    fun=j_exp_comp,
    x0=init,
    args=(xp, xn, Nc),
    bounds=bnds,
    method='trust-constr',
    constraints=[nmax_cons, mn_cons, m_sum_cons]
)
end = time.time()
start - end

solution = opt.x.round()
mvals = solution[-Nc:]
good = solution[:-Nc].reshape((Ns, Nc), order='F')
j_exp_comp(opt.x.round(), xp, xn, Nc)
