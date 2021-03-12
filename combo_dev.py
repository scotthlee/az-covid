import numpy as np
import itertools
import pickle
import time

from multiprocessing import Pool
from sys import getsizeof


def flatten(l):
    return [item for sublist in l for item in sublist]


def unique_combo(c):
    if len(np.intersect1d(c[0], c[1])) == 0:
        return c
    else:
        return None


def rowsums(m, min=1):
    sums = np.sum(m, axis=1)
    return np.array(sums >= min, dtype=np.uint8)


def pairsum(X, c, min=(1, 1)):
    a = rowsums(X[:, c[0]], min=min[0])
    b = rowsums(X[:, c[1]], min=min[1])
    return (a, b)


def f1_score(y, y_):
    n = y.shape[0]
    tp = np.sum([y[i] == 1 and y_[i] == 1 for i in range(n)])
    fp = np.sum([y[i] == 0 and y_[i] == 1 for i in range(n)])
    fn = np.sum([y[i] == 1 and y_[i] == 0 for i in range(n)])
    sens = tp / (tp + fn)
    ppv = tp / (tp + fp)
    return 2 * sens * ppv / (sens + ppv)


file_dir = 'C:/Users/yle4/OneDrive - CDC/Documents/projects/az covid/'
# List of symptom names and case definitions
symptom_list = [
    'fever', 'chills', 'shiver', 'ma', 'congestion',
    'sorethroat', 'cough', 'sob', 'difficultbreath', 'nauseavom',
    'headache', 'abpain', 'diarrhea', 'losstastesmell', 'fatigue'
    ]

# Loading the arrays
X = pickle.load(open(file_dir + 'X.pkl', 'rb'))
y = pickle.load(open(file_dir + 'y.pkl', 'rb'))

# Setting the maximum combination size
c_min = 1
c_max = 5
c_list = list(range(c_min, c_max+1))

# Generating the combos
n_symps = range(len(symptom_list))
combos = [[(list(group), k) 
           for group in itertools.combinations(n_symps, k)]
          for k in list(range(c_min, c_max+1))]
combos = flatten(combos)
nums = [combo[1] for combo in combos]
col_combos = [combo[0] for combo in combos]

# Running the combo comparisons for the different age groups
cstat_list = []
cy_list = []
cnames_list = [[symptom_list[n] for n in combo] for combo in col_combos]

# Calculating performance for the metacombinations
meta_iter = [pair for pair in itertools.combinations(col_combos, 2)]

# Starting the pool
metacombos = p.map(unique_combo, meta_iter)
metacombos = [c for c in metacombos if c is not None]

# Combos of m for m-of-n
mcs = [pair for pair in itertools.permutations(range(1, 6), 2)]
mcs += [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

# Sums for at least 2 
raw_pairsums = [[(X, c, mc) for c in metacombos] for mc in mcs]

# Weeding out combinations that don't make sense
pairsum_input = []
for i, pairs in enumerate(raw_pairsums):
    good_pairs = [pair for pair in pairs if len(pair[1][0]) >= pair[2][0] 
                  and len(pair[1][1]) >= pair[2][1]]
    pairsum_input.append(good_pairs)

# Empty list to hold the best indices from each run
top_performers = []

# Max number of combos to consider from 'and', 'or', and 'any'
best_n = 10

# Starting the main loop
for run_num, input in enumerate(pairsum_input):
    print(run_num)
    print(getsizeof(input))
    # Getting the rowsums for each of the combos
    with Pool() as p:
        psums = p.starmap(pairsum, input)
        csums = p.map(combo_sum, [ps for ps in psums])
        and_f1s = p.starmap(f1_score, [(y, cs[:, 0]) for cs in csums])
        any_f1s = p.starmap(f1_score, [(y, cs[:, 1]) for cs in csums])
    
    # Pulling out the best from each one
    top_and = np.argsort(and_f1s)[::-1][0:best_n]
    top_any = np.argsort(any_f1s)[::-1][0:best_n]
    top_all = [top_and, top_any]
    modes = ['both', 'any']
    prefixes = ['both of ', 'any of ']
