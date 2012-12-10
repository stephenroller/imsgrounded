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
from itertools import izip

TRIALS = 100

# levels of noise for svd and zscore
NOISES = [0.0, 0.05, 0.1, 0.25]
NOISES = [0.0]
# zscore possibilities
ZSCORES = [None] + decrange(4.0, 1.0, -0.25)
# number of dimensions in SVD
K = 10
# levels of noise and number removed for minrho
NUM_DROP = range(25, 0, -1)
PERCENT_REMOVE = decrange(0.00, 0.8, 0.02)


# data loading
heads, mods, whole, assoc = load_data()
concatted = pd.concat([heads, mods], ignore_index=True)

# don't need to process this a bunch of times
agg_concat_orig = combine_measures(aggregate_ratings(concatted))['mean']
agg_whole_orig = aggregate_ratings(whole)['mean']


def run_experiment(randomizer, cleaners, parameters):
    variations_concat = [randomizer(concatted) for i in xrange(TRIALS)]
    variations_whole = [randomizer(whole) for i in xrange(TRIALS)]

    for cleaner, params in izip(cleaners, parameters):
        agg_cl_concat_orig = combine_measures(aggregate_ratings(cleaner(concatted)))['mean']
        agg_cl_whole_orig = aggregate_ratings(cleaner(whole))['mean']

        results = {}
        for noisy_concat, noisy_whole in izip(variations_concat, variations_whole):
            clean_concat = cleaner(noisy_concat)
            clean_whole = cleaner(noisy_whole)

            try:
                agg_concat = combine_measures(aggregate_ratings(noisy_concat))['mean']
                agg_whole = aggregate_ratings(noisy_whole)['mean']

                agg_cl_concat = combine_measures(aggregate_ratings(clean_concat))['mean']
                agg_cl_whole = aggregate_ratings(clean_whole)['mean']
            except KeyError:
                continue

            pairs = {
                'noisy_noisy': (agg_concat, agg_whole),
                'clean_noisy': (agg_cl_concat, agg_whole),
                'noisy_clean': (agg_concat, agg_cl_whole),
                'clean_clean': (agg_cl_concat, agg_cl_whole),
                'noisy_orig':  (agg_concat, agg_whole_orig),
                'clean_orig':  (agg_cl_concat, agg_whole_orig),
                'orig_noisy':  (agg_concat_orig, agg_whole),
                'orig_clean':  (agg_concat_orig, agg_cl_whole),
                'orig_orig':   (agg_concat_orig, agg_whole_orig),
                'cleanorig_orig':  (agg_cl_concat_orig, agg_whole_orig),
                'cleanorig_noisy': (agg_cl_concat_orig, agg_whole),
                'cleanorig_clean': (agg_cl_concat_orig, agg_cl_whole),
                'orig_cleanorig':  (agg_concat_orig, agg_cl_whole_orig),
                'noisy_cleanorig': (agg_concat, agg_cl_whole_orig),
                'clean_cleanorig': (agg_cl_whole, agg_cl_whole_orig),
            }

            for key, (l, r) in pairs.iteritems():
                if key not in results:
                    results[key] = []
                rho, p = na_spearmanr(l, r)
                results[key].append(rho)

            blank_counts = {
                'whole_orig': whole,
                'indiv_orig': concatted,
                'whole_clean': clean_whole,
                'indiv_clean': clean_concat,
                'whole_noisy': noisy_whole,
                'indiv_noisy': noisy_concat,
            }

            for key, x in blank_counts.iteritems():
                jc = x.columns[2:]
                if key + "_notblank" not in results:
                    results[key + "_notblank"] = []
                results[key + "_notblank"].append(pd.notnull(x[jc]).sum().sum())

        row = {}
        trials = 0
        for k, val in results.iteritems():
            v = pd.Series(val)
            mn = v.mean()
            sd = v.std()
            row[k] = mn
            row[k + '_std'] = sd
            row[k + '_low'] = mn - 2 * sd
            row[k + '_high'] = mn + 2 * sd
            trials = len(v)

        for key, val in params.iteritems():
            row[key] = val

        row['trials'] = trials

        yield row

def zscore_run(randomizer, randomizer_name):
    pb = ProgressBar(len(ZSCORES) * len(NOISES))
    sys.stderr.write("Beginning zscore eval with %s randomization.\n" % randomizer_name)
    pb.errput()
    for percent_noise in NOISES:
        this_rand = lambda d: randomizer(d, percent_noise)

        cleaners = [zscore and RemoveDeviantRatings(zscore).scores or BaselineCleaner().scores
                    for zscore in ZSCORES]
        parameters = [dict(cleaner='zscore', p=percent_noise, randomizer=randomizer_name, zscore=str(zscore))
                      for zscore in ZSCORES]

        for row in run_experiment(this_rand, cleaners, parameters):
            yield row
            pb.incr_and_errput()

def svd_run(randomizer, randomizer_name):
    pb = ProgressBar((K + 1) * len(NOISES))
    sys.stderr.write("Beginning SVD eval with %s randomization.\n" % randomizer_name)
    pb.errput()
    for percent_noise in NOISES:
        this_rand = lambda d: randomizer(d, percent_noise)

        parameters = [{
            'cleaner': 'svd',
            'p_noise': percent_noise,
            'randomizer': randomizer_name,
            'k': str(k)
        } for k in [None] + range(1, K + 1)]
        cleaners = [BaselineCleaner().scores] + [c.scores for c in create_svd_cleaners(K)]

        for row in run_experiment(this_rand, cleaners, parameters):
            yield row
            pb.incr_and_errput()


def minrho_run():
    pb = ProgressBar(len(NUM_DROP))
    sys.stderr.write("Beginning minrho eval.\n")
    pb.errput()

    for n in NUM_DROP:
        this_rand = lambda d: replace_subjects(d, n)
        cleaner = lambda d: remove_most_deviant_subjects(d, n)
        params = {'n': n, 'cleaner': 'minrho'}

        for row in run_experiment(this_rand, [cleaner], [params]):
            yield row
            pb.incr_and_errput()

def dropsubj_run():
    pb = ProgressBar(len(PERCENT_REMOVE))
    for p in PERCENT_REMOVE:
        this_rand = lambda d: replace_percent_subjects(d, p)
        cleaner = lambda d: remove_percent_deviant_subjects(d, p)
        params = {'p': p, 'cleaner': 'devsubj'}

        for row in run_experiment(this_rand, [cleaner], [params]):
            yield row
            pb.incr_and_errput()



if __name__ == '__main__':
    method = sys.argv[1]

    if method == 'zscore' or method == 'svd':
        noise_method = sys.argv[2]
        if noise_method == 'offset':
            randomizer = randomize_offsets
        elif noise_method == 'uniform':
            randomizer = randomize_values
        else:
            raise ValueError("Bad noise method.")

        if method == 'zscore':
            runner = zscore_run(randomizer, sys.argv[2])
        elif method == 'svd':
            runner = svd_run(randomizer, sys.argv[2])
        else:
            raise ValueError('Wtf?')

    elif method == 'minrho':
        runner = minrho_run()
    elif method == 'dropsubj':
        runner = dropsubj_run()
    else:
        raise ValueError("Bad processing method.")

    pd.DataFrame(list(runner)).to_csv(sys.stdout, index=False)

