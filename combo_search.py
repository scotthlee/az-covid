import numpy as np
import pandas as pd
import itertools
import time

from multiprocessing import Pool
from sklearn.metrics import f1_score

import tools
import multi


'''
Importing and organizing the data
'''
file_dir = 'C:/Users/yle4/OneDrive - CDC/Documents/projects/az covid/'
records = pd.read_csv(file_dir + 'records.csv')

# List of symptom names and case definitions
symptom_list = [
    'fever', 'chills', 'shiver', 'ma', 'congestion',
    'sorethroat', 'cough', 'sob', 'difficultbreath', 'nauseavom',
    'headache', 'abpain', 'diarrhea', 'losstastesmell', 'fatigue'
    ]

# Pinning down the inputs and targets
X = np.array(records[symptom_list], dtype=np.uint8)
y = np.array(records.poscovid, dtype=np.uint8)

'''
Calculating performance for the any-of-n (single)
and m-of-n [and/or] m-of-n (meta) combinations
'''
# Starting the pool
p = Pool()

# Setting the maximum combination size
c_min = 1
c_max = 5
c_list = list(range(c_min, c_max+1))

# Generating the combos
n_symps = range(len(symptom_list))
combos = [[(list(group), k) 
           for group in itertools.combinations(n_symps, k)]
          for k in list(range(c_min, c_max+1))]
combos = tools.flatten(combos)
nums = [combo[1] for combo in combos]
col_combos = [combo[0] for combo in combos]

# Running the combo comparisons for the different age groups
cstat_list = []
cy_list = []
cnames_list = [[symptom_list[n] for n in combo] for combo in col_combos]

# Calculating performance for the metacombinations
meta_iter = [pair for pair in itertools.combinations(col_combos, 2)]
metacombos = p.map(tools.unique_combo, meta_iter)
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

for run_num, input in enumerate(pairsum_input):
    print(run_num)
    # Getting the rowsums for each of the combos
    psums = p.starmap(tools.pairsum, input)
    
    # Converting the pair of rowsums to a single column based on 
    # the specified logical criterium ('any', 'or', or 'and')
    csums = p.map(tools.combo_sum, [ps for ps in psums])
    
    # Calculating f1 score for each of the combo sums
    and_f1s = p.starmap(f1_score, [(y, cs[:, 0]) for cs in csums])
    any_f1s = p.starmap(f1_score, [(y, cs[:, 1]) for cs in csums])
    
    # Pulling out the best from each one
    top_and = np.argsort(and_f1s)[::-1][0:best_n]
    top_any = np.argsort(any_f1s)[::-1][0:best_n]
    top_all = [top_and, top_any]
    modes = ['both', 'any']
    prefixes = ['both of ', 'any of ']
    
    # Running the full metrics on the best from each group
    for j, top_idx in enumerate(top_all):
        prefx = prefixes[j]
        mode = modes[j]
        group_best = []
        
        for i in top_idx:
            pair = input[i]
            pair_cols = pair[1]
            pair_m = pair[2]
            
            # Making the string specifying the condition
            colnames = []
            for cols in pair_cols:
                colnames.append(' '.join([str(cnames_list[c][0]) 
                                             for c in cols]))
            s1 = prefx + str(pair_m[0]) + ' from [' + str(colnames[0])
            s2 = '] and ' + str(pair_m[1]) + ' from [' + str(colnames[1])
            s = s1 + s2 + ']'
            mtx = tools.combo_metrics(X, y, pair_cols, pair_m, mode)
            mtx['cond'] = s
            group_best.append(mtx)
        
        # Adding the results to the list
        top_performers.append(pd.concat(group_best, axis=0))
    
    # Writing results for the top combos to disk
    top_df = pd.concat(top_performers, axis=0)
    top_df.to_csv(file_dir + 'metacombo_stats.csv', index=False)
