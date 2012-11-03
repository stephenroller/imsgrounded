#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import argparse

import pandas as pd
from scipy.stats import spearmanr

from standard_cleanup import aggregate_ratings
from standard_cleanup import remove_deviant_ratings, remove_deviant_subjects


HEAD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_head.csv'
MOD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_const.csv'
WHOLE_FILE = '/Users/stephen/Working/imsgrounded/data/comp/amt_reshaped.csv'
ASSOC_FILE = '/Users/stephen/Working/imsgrounded/results/big_assoc_similarties.csv'

COMBINE_METHODS = ['sum', 'prod', 'mean']


# Things we need to output:
# - Correlations with wholes, from perspective of just heads and just mods
# - Correlation with wholes, from perspective of sum/prod of heads and mods together
# - ditto for Association measures
# - results of varying the filter parameters
# - do the same cleanups for the wholes?


def decrange(start, stop, inc):
    out = []
    x = start
    while x <= stop:
        out.append(x)
        x += inc
    return out


def combine_measures(agg_heads_and_mods, method='sum'):
    # returns a new DataFrame that's the exact same format as the whole file
    # method should be either 'sum' or 'prod'.
    grouped = agg_heads_and_mods.groupby('compound')
    if method == 'sum':
        together = grouped.sum()
    elif method == 'prod':
        together = grouped.prod()
    elif method == 'mean':
        together = grouped.mean()
    else:
        raise ValueError("Invalid method for combining measures.")
    together['compound'] = together.index
    return together

def rho_with_wholes(indiv, whole):
    assert (indiv['compound'] == whole['compound']).all()

    rho, p = spearmanr(indiv['mean'], whole['mean'])
    # hardcode statistical significance
    assert p < .05
    return rho, p

def rho_with_assoc(indiv, assoc, measure):
    assert (indiv['compound'] == assoc['compound']).all()
    assert (indiv['const'] == assoc['const']).all()
    assert measure in ('jaccard', 'cosine')
    return spearmanr(indiv['mean'], assoc[measure])

heads = pd.read_csv(HEAD_FILE)
mods = pd.read_csv(MOD_FILE)
whole = pd.read_csv(WHOLE_FILE)
assoc = pd.read_csv(ASSOC_FILE)

# make sure we drop the association measures between a compound
# and itself. they're always 1.
assoc = assoc[assoc.compound != assoc.const]

# we can only work with the intersection of all 3 files in terms of
# judgements
good_compounds = reduce(
    set.intersection, 
    [set(x.compound) for x in [heads, mods, whole, assoc]]
)
heads, mods, whole, assoc = [
    d[d.compound.map(good_compounds.__contains__)]
    for d in [heads, mods, whole, assoc]
]


# go ahead and sort the whole judgements and assoc measures
whole = whole.sort('compound')
assoc = assoc.sort(['compound', 'const'])


results = []

concatted = pd.concat([heads, mods])
for min_rho in [None] + decrange(0.35, 0.6, 0.05):
    for zscore in [None] + decrange(1.0, 3.0, 0.5):
        data = concatted
        try:
            if min_rho:
                data = remove_deviant_subjects(data, min_rho)

            if zscore:
                data = remove_deviant_ratings(data, zscore)

            agg = aggregate_ratings(data)
            agg = agg.sort(['compound', 'const'])
        except:
            # print "Whoops, everything removed. Skipping.\n"
            continue

        row = {'zscore': zscore, 'dev_subj': min_rho}
        for method in COMBINE_METHODS:
            together = combine_measures(agg, method)
            together = together.sort('compound')
            rho, p = rho_with_wholes(together, aggregate_ratings(whole))
            # print "with wholes rho %s: %f" % (method, rho)
            row['with_wholes_%s' % method] = rho


        for measure in ['cosine', 'jaccard']:
            rho, p = rho_with_assoc(agg, assoc, measure)
            # print "with assoc %s rho: %f" % (measure, rho)
            row['with_assoc_%s' % measure] = rho

        results.append(row)

pd.DataFrame(results).to_csv(sys.stdout, index=False)




