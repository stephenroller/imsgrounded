#!/usr/bin/env python

import sys
import os, os.path
import pandas as pd

from pandas.core.reshape import melt
from eval_cleanups import *
from standard_cleanup import *
from noisify import *
from scipy.stats import spearmanr, norm
from progress import ProgressBar

REPEATS = 100


heads, mods, whole, assoc = load_data()
concatted = pd.concat([heads, mods], ignore_index=True)

agg_concat_orig = combine_measures(aggregate_ratings(concatted))['mean']
agg_whole_orig = aggregate_ratings(whole)['mean']

# ------ RANDOMIZE BY ZSCORES TEST --------

def mass_outside(zstar):
    return 2 * norm().cdf(-abs(zstar))

output = []
# first zscores
KS = [None] + range(1, 21)
NOISES = [0.0, 0.01, 0.05] #, 0.1, 0.25, 0.5]

pb = ProgressBar(len(KS) * REPEATS * len(NOISES))
pb.errput()
for k in KS:
    cleaner = k and SvdCleaner(k) or BaselineCleaner()
    for percent_noise in NOISES:
        this_row = {}
        for i in xrange(REPEATS):
            noisy_concat = randomize_offsets(concatted, percent_noise)
            clean_concat = cleaner.scores(noisy_concat)
            noisy_whole = randomize_offsets(whole, percent_noise)
            clean_whole = cleaner.scores(noisy_whole)

            agg_concat = combine_measures(aggregate_ratings(noisy_concat))['mean']
            agg_whole = aggregate_ratings(noisy_whole)['mean']

            agg_cl_concat = combine_measures(aggregate_ratings(clean_concat))['mean']
            agg_cl_whole = aggregate_ratings(clean_whole)['mean']

            pairs = {
                'noisy_noisy': (agg_concat, agg_whole),
                'clean_noisy': (agg_cl_concat, agg_whole),
                'noisy_clean': (agg_concat, agg_cl_whole),
                'clean_clean': (agg_cl_concat, agg_cl_whole),
                'noisy_orig':  (agg_concat, agg_whole_orig),
                'clean_orig':  (agg_cl_concat, agg_whole_orig),
                'orig_noisy':  (agg_concat_orig, agg_whole),
                'orig_clean':  (agg_concat_orig, agg_cl_whole),
            }

            for key, (l, r) in pairs.iteritems():
                if key not in this_row:
                    this_row[key] = []
                rho, p = na_spearmanr(l, r)
                this_row[key].append(rho)

            pb.incr()
            if i % 5 == 0 : pb.errput()

        row = dict(k=str(k), p=percent_noise)
        for k, v in this_row.iteritems():
            v = pd.Series(v)
            mn = v.mean()
            sd = v.std()
            row[k] = mn
            row[k + '_std'] = sd
            row[k + '_low'] = mn - 2 * sd
            row[k + '_high'] = mn + 2 * sd


        output.append(row)

    pb.errput()

to_graph = pd.DataFrame(output)
to_graph.to_csv(sys.stdout, index=False)


print
print
print



