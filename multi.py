import pandas as pd
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import binom, chi2, norm, percentileofscore
from copy import deepcopy
from multiprocessing import Pool
from copy import deepcopy

from tools import (mcnemar_test, brier_score, threshold, 
                   clf_metrics, boot_sample)


def jackknife_metrics(targets, 
                      guesses, 
                      average='weighted'):
    # Replicates of the dataset with one row missing from each
    rows = np.array(list(range(targets.shape[0])))
    j_rows = [np.delete(rows, row) for row in rows]

    # using a pool to get the metrics across each
    score_input = [(targets[idx], guesses[idx]) for idx in j_rows]
    with Pool() as p:
        scores = p.starmap(clf_metrics, score_input)
    
    scores = pd.concat(scores, axis=0)
    means = scores.mean()
    
    return scores, means


class boot_cis:
    def __init__(
        self,
        targets,
        guesses,
        n=100,
        a=0.05,
        group=None,
        method="bca",
        interpolation="nearest",
        average='weighted',
        mcnemar=False,
        seed=10221983):
        # Converting everything to NumPy arrays, just in case
        stype = type(pd.Series([0]))
        if type(targets) == stype:
            targets = targets.values
        if type(guesses) == stype:
            guesses = guesses.values

        # Getting the point estimates
        stat = clf_metrics(targets,
                           guesses,
                           average=average,
                           mcnemar=mcnemar).transpose()

        # Pulling out the column names to pass to the bootstrap dataframes
        colnames = list(stat.index.values)

        # Making an empty holder for the output
        scores = pd.DataFrame(np.zeros(shape=(n, stat.shape[0])),
                              columns=colnames)

        # Setting the seed
        if seed is None:
            seed = np.random.randint(0, 1e6, 1)
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, n)

        # Generating the bootstrap samples and metrics
        boots = [boot_sample(targets, seed=seed) for seed in seeds]
        score_input = [(targets[b], guesses[b]) for b in boots]
        with Pool() as p:
            scores = p.starmap(clf_metrics, score_input)
        
        scores = pd.concat(scores, axis=0)

        # Calculating the confidence intervals
        lower = (a / 2) * 100
        upper = 100 - lower

        # Making sure a valid method was chosen
        methods = ["pct", "diff", "bca"]
        assert method in methods, "Method must be pct, diff, or bca."

        # Calculating the CIs with method #1: the percentiles of the
        # bootstrapped statistics
        if method == "pct":
            cis = np.nanpercentile(scores,
                                   q=(lower, upper),
                                   interpolation=interpolation,
                                   axis=0)
            cis = pd.DataFrame(cis.transpose(),
                               columns=["lower", "upper"],
                               index=colnames)

        # Or with method #2: the percentiles of the difference between the
        # obesrved statistics and the bootstrapped statistics
        elif method == "diff":
            stat_vals = stat.transpose().values.ravel()
            diffs = stat_vals - scores
            percents = np.nanpercentile(diffs,
                                        q=(lower, upper),
                                        interpolation=interpolation,
                                        axis=0)
            lower_bound = pd.Series(stat_vals + percents[0])
            upper_bound = pd.Series(stat_vals + percents[1])
            cis = pd.concat([lower_bound, upper_bound], axis=1)
            cis = cis.set_index(stat.index)

        # Or with method #3: the bias-corrected and accelerated bootstrap
        elif method == "bca":
            # Calculating the bias-correction factor
            stat_vals = stat.transpose().values.ravel()
            n_less = np.sum(scores < stat_vals, axis=0)
            p_less = n_less / n
            z0 = norm.ppf(p_less)

            # Fixing infs in z0
            z0[np.where(np.isinf(z0))[0]] = 0.0

            # Estiamating the acceleration factor
            j = jackknife_metrics(targets, guesses)
            diffs = j[1] - j[0]
            numer = np.sum(np.power(diffs, 3))
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3 / 2)

            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6

            # Finishing up the acceleration parameter
            acc = numer / denom
            self.jack = j

            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a / 2))
            lterm = (z0 + zl) / (1 - acc * (z0 + zl))
            uterm = (z0 + zu) / (1 - acc * (z0 + zu))
            ql = norm.cdf(z0 + lterm) * 100
            qu = norm.cdf(z0 + uterm) * 100
            
            # Passing things back to the class
            self.acc = acc.values
            self.b = z0
            self.ql = ql
            self.qu = qu

            # Returning the CIs based on the adjusted quintiles
            cis = [
                np.nanpercentile(
                    scores.iloc[:, i],
                    q=(ql[i], qu[i]),
                    interpolation=interpolation,
                    axis=0,
                ) for i in range(len(ql))
            ]
            cis = pd.DataFrame(cis, 
                               columns=["lower", "upper"], 
                               index=colnames)

        # Putting the stats with the lower and upper estimates
        cis = pd.concat([stat, cis], axis=1)
        cis.columns = ["stat", "lower", "upper"]

        # Passing the results back up to the class
        self.cis = cis
        self.scores = scores

        return
