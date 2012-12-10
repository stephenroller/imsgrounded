#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import argparse

import numpy as np
import pandas as pd
from pandas.core.reshape import melt

from standard_cleanup import aggregate_ratings, na_spearmanr
from standard_cleanup import remove_deviant_ratings, remove_deviant_subjects
from rebin import rebin
from whiten import netflix_svd


HEAD_FILE =  'data/comp/comp_ratings_head.csv'
MOD_FILE =   'data/comp/comp_ratings_const.csv'
WHOLE_FILE = 'data/comp/amt_reshaped.csv'
ASSOC_FILE = 'results/big_assoc_similarties.csv'

# COMBINE_METHODS = ['sum', 'prod', 'hmean']
COMBINE_METHODS = ['prod']
# ASSOC_METHODS = ['jaccard', 'cosine']
ASSOC_METHODS = ['jaccard']


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

    def __str__(self):
        return "Base class"


class BaselineCleaner(Cleaner):
    def scores(self, df):
        return df

    def parameters(self):
        return {}

    def __str__(self):
        return "Do nothing"

class RemoveDeviantSubjectCleaner(Cleaner):
    def __init__(self, min_rho):
        self.min_rho = min_rho

    def scores(self, df):
        return remove_deviant_subjects(df, self.min_rho)

    def parameters(self):
        return {'min_rho': self.min_rho}

    def __str__(self):
        return "Remove Subj (rho < %f)" % self.min_rho

class RemoveDeviantRatings(Cleaner):
    def __init__(self, zscore):
        self.zscore = zscore

    def parameters(self):
        return {'zscore': self.zscore}

    def scores(self, df):
        return remove_deviant_ratings(df, self.zscore)

    def __str__(self):
        return "Remove Judgements (z < %f)" % self.zscore

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
        return rebin(df, self.bin_mapping)

    def __str__(self):
        return "Rebin (%s)" % self.bin_names


class CachedSvdCleaner():
    def __init__(self, max_k, learning_rate=.001):
        self.whitened = {}
        self.max_k = max_k
        self.lr = learning_rate

    def scores(self, df, k):
        if id(df) not in self.whitened:
            just_data = df[df.columns[2:]]
            self.whitened[id(df)] = netflix_svd(just_data, self.max_k, epochs=10000, learning_rate=self.lr)

        U, V = self.whitened[id(df)]
        W = np.dot(U[:,:k], V[:k,:])
        whitened_data = df.copy()
        for i, j in enumerate(df.columns[2:]):
            whitened_data[j] = W[:,i]

        return whitened_data

def create_svd_cleaners(max_k):
    cacher = CachedSvdCleaner(max_k)
    cleaners = []

    for k in xrange(1, max_k + 1):
        class _SvdCleaner(Cleaner):
            def __init__(self, k):
                self.k = k
                self.cacher = cacher

            def scores(self, df):
                return self.cacher.scores(df, self.k)

            def parameters(self):
                return dict(svd_k=self.k)

            def __str__(self):
                return "SVD (rank %d)" % self.k
        cleaners.append(_SvdCleaner(k))
    return cleaners

class FillCleaner(Cleaner):
    def __init__(self, fill_value):
        self.fill_value = fill_value

    def scores(self, df):
        return df.fillna(self.fill_value)

    def parameters(self):
        return dict(fill_value=self.fill_value)

    def __str__(self):
        return "Fill w %d" % self.fill_value


def decrange(start, stop, inc):
    out = []
    x = start
    while (inc > 0 and x <= stop) or (inc < 0 and x >= stop):
        out.append(x)
        x += inc
    return out

def combine_measures(agg_heads_and_mods, method='gmean'):
    # returns a new DataFrame that's the exact same format as the whole file
    # method should be either 'sum' or 'prod'.
    grouped = agg_heads_and_mods.groupby('compound')
    if method == 'sum':
        together = grouped.sum()
    elif method == 'prod':
        together = grouped.prod()
    elif method == 'gmean':
        together = grouped.prod().map(sqrt)
    elif method == 'hmean':
        together = 2 * grouped.prod() / grouped.sum()
    else:
        raise ValueError("Invalid method for combining measures.")
    together['compound'] = together.index
    return together

def load_data():
    heads = pd.read_csv(HEAD_FILE)
    mods = pd.read_csv(MOD_FILE)
    whole = pd.read_csv(WHOLE_FILE).sort('compound')
    assoc = pd.read_csv(ASSOC_FILE)

    # make sure we drop the association measures between a compound
    # and itself. they're always 1.
    assoc = assoc[assoc.compound != assoc.const]
    assoc = assoc.sort(['compound', 'const'])

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

    return heads, mods, whole, assoc


if __name__ == '__main__':
    # go ahead and sort the whole judgements and assoc measures
    heads, mods, whole_orig, assoc = load_data()
    whole = aggregate_ratings(whole_orig)
    concatted = pd.concat([heads, mods], ignore_index=True)
    concatted_uncleaned_together = combine_measures(aggregate_ratings(concatted), 'prod').sort('compound')

    setups = []
    setups += [BaselineCleaner()]
    setups += [RemoveDeviantSubjectCleaner(r) for r in decrange(0.10, 0.6, 0.05)]
    setups += [RemoveDeviantRatings(z) for z in decrange(1.0, 4.0, 0.25)]
    #setups += [RebinCleaner(b) for b in ["1144477","1444447","1114777","1122233","1222223","1112333"]]
    setups += create_svd_cleaners(20)
    #setups += [FillCleaner(0), FillCleaner(1), FillCleaner(7)]

    results = []
    parameters = set()

    CONCAT_BEFORE = True

    jcc = concatted.columns[2:]
    blank_concat = float(pd.notnull(concatted[jcc]).sum().sum())
    jcw = whole_orig.columns[2:]
    blank_whole = float(pd.notnull(whole_orig[jcw]).sum().sum())


    for cleaner in setups:
        sys.stderr.write("Evaluating model: %s\n" % cleaner)
        if CONCAT_BEFORE:
            concat_cleaned = cleaner.scores(concatted)
        else:
            concat_cleaned = pd.concat([cleaner.scores(heads), cleaner.scores(mods)], ignore_index=True)
        agg = aggregate_ratings(concat_cleaned).sort(['compound', 'const'])
        sys.stderr.write("Finished cleaning head/const ratings. (%s)\n" % cleaner)
        whole_clean_noagg = cleaner.scores(whole_orig)
        whole_clean = aggregate_ratings(whole_clean_noagg).sort('compound')
        sys.stderr.write("Finished cleaning whole ratings. (%s)\n" % cleaner)
        row = cleaner.parameters()

        blank_cleaned_concat = pd.notnull(concat_cleaned[jcc]).sum().sum()
        blank_cleaned_whole = pd.notnull(whole_clean_noagg[jcw]).sum().sum()

        row['Data Retained (Indiv)'] = blank_cleaned_concat / blank_concat
        row['Data Retained (Whole)'] = blank_cleaned_whole / blank_whole
        row['Data Retained (Both)'] = (blank_cleaned_whole + blank_cleaned_concat) / (blank_concat + blank_whole)


        parameters.update(row.keys())

        together = combine_measures(agg, 'prod').sort('compound')
        rho, p1 = na_spearmanr(together['mean'], whole['mean'])
        row['Parts Cleaned'] = rho

        # and now when we clean up whole measures
        rho, p1 = na_spearmanr(together['mean'], whole_clean['mean'])
        row['Parts & Whole Cleaned'] = rho

        # clean the whole, but not the parts
        rho, p1 = na_spearmanr(concatted_uncleaned_together['mean'], whole_clean['mean'])
        row['Whole Cleaned'] = rho

        rho, p1 = na_spearmanr(agg['mean'], assoc['jaccard'])
        row['Association Norms (Indiv)'] = rho

        combined_assoc = combine_measures(assoc[['compound', 'const', 'jaccard']]).sort('compound')
        rho, p1 = na_spearmanr(combined_assoc['jaccard'], whole_clean['mean'])
        row['Association Norms (Whole)'] = rho

        results.append(row)

    results = pd.DataFrame(results)
    output = melt(results, id_vars=parameters)
    output.to_csv(sys.stdout, index=False)

    # # produce plots
    # from rplots import line_plot
    # import operator
    # for p in parameters:
    #     other_params = parameters - set([p])
    #     if other_params:
    #         experiment = reduce(operator.and_, [output[op].isnull() for op in other_params])
    #         experiment = output[experiment]
    #     else:
    #         experiment = output
    #     line_plot("graphs/" + p + ".pdf", experiment, p, 'value', 'variable',
    #             ylab='Resulting Correlation',
    #             colorname="Eval Method")



