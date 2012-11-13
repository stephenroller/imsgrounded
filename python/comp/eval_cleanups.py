#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import argparse

import pandas as pd
from pandas.core.reshape import melt
from scipy.stats import spearmanr

from standard_cleanup import aggregate_ratings
from standard_cleanup import remove_deviant_ratings, remove_deviant_subjects
from elo import elo
from rebin import rebin


HEAD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_head.csv'
MOD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_const.csv'
WHOLE_FILE = '/Users/stephen/Working/imsgrounded/data/comp/amt_reshaped.csv'
ASSOC_FILE = '/Users/stephen/Working/imsgrounded/results/big_assoc_similarties.csv'

COMBINE_METHODS = ['sum', 'prod']


# Things we need to output:
# - Correlations with wholes, from perspective of just heads and just mods
# - Correlation with wholes, from perspective of sum/prod of heads and mods together
# - ditto for Association measures
# - results of varying the filter parameters
# - do the same cleanups for the wholes?

class Cleaner(object):
    def scores(self, df):
        raise NotImplementedError('Base class.')

    def paramaters(self):
        return {}

class BaselineCleaner(Cleaner):
    def scores(self, df):
        return aggregate_ratings(df)

    def parameters(self):
        return {}


class RemoveDeviantSubjectCleaner(Cleaner):
    def __init__(self, min_rho):
        self.min_rho = min_rho

    def scores(self, df):
        return aggregate_ratings(remove_deviant_subjects(df, self.min_rho))

    def parameters(self):
        return {'min_rho': self.min_rho}

class RemoveDeviantRatings(Cleaner):
    def __init__(self, zscore):
        self.zscore = zscore

    def parameters(self):
        return {'zscore': self.zscore}

    def scores(self, df):
        return aggregate_ratings(remove_deviant_ratings(df, self.zscore))

class EloRatings(Cleaner):
    def __init__(self, k=32, start=1500, spread=400):
        self.k = k
        self.start = start
        self.spread = spread

    def parameters(self):
        return dict(elo_k=self.k, elo_start=self.start, elo_spread=self.spread)

    def scores(self, df):
        return elo(df, self.k, self.start, self.spread)

class RebinCleaner(Cleaner):
    def __init__(self, new_bins):
        if isinstance(new_bins, dict):
            self.bin_names = "".join(str(new_bins[k]) for k in sorted(new_bins.keys()))
            self.bin_mapping = new_bins
        elif isinstance(new_bins, (list, tuple)):
            self.bin_names = "".join(map(str, new_bins))
            self.bin_mapping = dict(zip(range(1, len(new_bins) + 1), new_bins))
        elif isinstance(new_bins, str):
            self.__init__(map(int, new_bins))
        else:
            raise ValueError("Inappropriate bin value.")

    def parameters(self):
        return dict(bins=self.bin_names)

    def scores(self, df):
        return aggregate_ratings(rebin(df, self.bin_mapping))


class SvdCleaner(Cleaner):
    def __init__(self, k):
        self.k = k

    def parameters(self):
        return dict(svd_k=self.k)

    def scores(self, df):
        df = whiten(df, k)
        return aggregate_ratings(df)



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
    else:
        raise ValueError("Invalid method for combining measures.")
    together['compound'] = together.index
    return together

def rho_with_wholes(indiv, whole):
    assert (indiv['compound'] == whole['compound']).all()

    rho, p = spearmanr(indiv['mean'], whole['mean'])
    # hardcode statistical significance
    if not p < .05:
        import pdb
        pdb.set_trace()
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
whole_orig = whole.sort('compound')
whole = aggregate_ratings(whole_orig)
assoc = assoc.sort(['compound', 'const'])
concatted = pd.concat([heads, mods], ignore_index=True)


setups = [BaselineCleaner()]
setups += [RemoveDeviantSubjectCleaner(r) for r in decrange(0.35, 0.6, 0.05)]
#setups += [RemoveDeviantRatings(z) for z in decrange(1.0, 3.0, 0.5)]
#setups += [EloRatings(.05, 1500, 400)]
#setups += [RebinCleaner(b) for b in ["1144477","1444447","1114777","1122233","1222223","1112333"]]
#setups += [SvdCleaner(k) for k in [25, 50, 75, 100, 200, 300, 500]]

results = []
parameters = set()

for cleaner in setups:
    agg = cleaner.scores(concatted).sort(['compound', 'const'])
    whole_clean = cleaner.scores(whole_orig).sort('compound')
    row = cleaner.parameters()
    parameters.update(row.keys())

    for method in COMBINE_METHODS:
        together = combine_measures(agg, method)
        together = together.sort('compound')
        rho, p = rho_with_wholes(together, whole)
        # print "with wholes rho %s: %f" % (method, rho)
        row['Whole judgements (%s)' % method] = rho
        # and now when we clean up whole measures
        rho, p = rho_with_wholes(together, whole_clean)
        row['Whole judgements cleaned (%s)' % method] = rho

    for measure in ['cosine', 'jaccard']:
        rho, p = rho_with_assoc(agg, assoc, measure)
        # print "with assoc %s rho: %f" % (measure, rho)
        row['Association sim (%s)' % measure] = rho

    results.append(row)

results = pd.DataFrame(results)
melt(results, id_vars=parameters).to_csv(sys.stdout, index=False)




