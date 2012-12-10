#!/usr/bin/env python

import sys
import os, os.path
import pandas as pd

from pandas.core.reshape import melt
from eval_cleanups import *
from standard_cleanup import *
from noisify import *
from progress import ProgressBar

REPEATS = 100


heads, mods, whole, assoc = load_data()
concatted = pd.concat([heads, mods], ignore_index=True)

agg_concat_orig = combine_measures(aggregate_ratings(concatted))['mean']
agg_whole_orig = aggregate_ratings(whole)['mean']

output = []

NUM_DROP = range(1, 26)
pb = ProgressBar(len(NUM_DROP) * REPEATS)
pb.errput()
for n in NUM_DROP:
    this_row = {}
    for i in xrange(REPEATS):
        noisy_concat = replace_subjects(concatted, n)
        noisy_whole = replace_subjects(whole, n)
        clean_concat = remove_most_deviant_subjects(noisy_concat, n)
        clean_whole = remove_most_deviant_subjects(noisy_whole, n)

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

        for k, (l, r) in pairs.iteritems():
            if k not in this_row:
                this_row[k] = []
            rho, p = na_spearmanr(l, r)
            this_row[k].append(rho)

        pb.incr()
        if i % 5 == 0 : pb.errput()

    row = dict(n=n)
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


# ----- REPLACE SUBJECTS WITH RANDOMNESS ------


# from standard_cleanup import remove_most_deviant_subjects
# output = []
# NUM_DROP = range(125)
# pb = ProgressBar(len(NUM_DROP) * REPEATS)
# pb.errput()
# for n in NUM_DROP:
#     z, a, b, c = [], [], [], []
#     for i in xrange(REPEATS):
#         noisy_concat = replace_subjects(concatted, n)
#         noisy_whole = replace_subjects(whole, n)
# 
#         agg_concat = combine_measures(aggregate_ratings(noisy_concat))['mean']
#         agg_whole = aggregate_ratings(noisy_whole)['mean']
# 
#         agg_cl_concat = combine_measures(aggregate_ratings(remove_most_deviant_subjects(noisy_concat, n)))['mean']
#         agg_cl_whole = aggregate_ratings(remove_most_deviant_subjects(noisy_whole, n))['mean']
# 
#         rho_concat_whole, p = spearmanr(agg_concat, agg_whole)
#         rho_clconcat_whole, p = spearmanr(agg_cl_concat, agg_whole)
#         rho_concat_clwhole, p = spearmanr(agg_concat, agg_cl_whole)
#         rho_clconcat_clwhole, p = spearmanr(agg_cl_concat, agg_cl_whole)
# 
#         z.append(rho_concat_whole)
#         a.append(rho_clconcat_whole)
#         b.append(rho_concat_clwhole)
#         c.append(rho_clconcat_clwhole)
# 
#         pb.incr()
#         if i % 5 == 0 : pb.errput()
# 
#     z = pd.Series(z)
#     a = pd.Series(a)
#     b = pd.Series(b)
#     c = pd.Series(c)
# 
#     row = dict(n=n)
#     row['concat_whole'] = z.mean()
#     row['concat_whole_low'] = z.mean() - 2 * z.std()
#     row['concat_whole_high'] = z.mean() + 2 * z.std()
#     row['concat2_whole'] = a.mean()
#     row['concat2_whole_low'] = a.mean() - 2 * a.std()
#     row['concat2_whole_high'] = a.mean() + 2 * a.std()
#     row['concat_whole2'] = b.mean()
#     row['concat_whole2_low'] = b.mean() - 2 * b.std()
#     row['concat_whole2_high'] = b.mean() + 2 * b.std()
#     row['concat2_whole2'] = c.mean()
#     row['concat2_whole2_low'] = c.mean() - 2 * c.std()
#     row['concat2_whole2_high'] = c.mean() + 2 * c.std()
# 
#     output.append(row)
# 
#     pb.errput()
# 
# 
# to_graph = pd.DataFrame(output)
# to_graph.to_csv(sys.stdout, index=False)
# 
# 
# 
# 
# 

