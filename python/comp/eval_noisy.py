#!/usr/bin/env python

import sys
import os, os.path
import pandas as pd

from pandas.core.reshape import melt
from eval_cleanups import load_data, decrange, RemoveDeviantRatings, combine_measures, BaselineCleaner
from standard_cleanup import aggregate_ratings
from noisify import *
from scipy.stats import spearmanr
from progress import ProgressBar

REPEATS = 100


heads, mods, whole, assoc = load_data()
concatted = pd.concat([heads, mods], ignore_index=True)
agg_whole = aggregate_ratings(whole)
agg_concat = combine_measures(aggregate_ratings(concatted))


OUTPUT_FOLDER = "graphs-noisy/"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

output = []
# first zscores
NOISES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]
ZSCORES = [None] + decrange(4.0, 1.0, -0.5)
pb = ProgressBar(len(NOISES) * len(ZSCORES) * REPEATS)
pb.errput()
for percent_noise in NOISES:
    for zscore in ZSCORES:
        if zscore is None:
            cleaner = BaselineCleaner()
        else:
            cleaner = RemoveDeviantRatings(zscore)

        for i in xrange(REPEATS):
            noisy_concat = randomize_values(concatted, percent_noise)
            noisy_whole = randomize_values(whole, percent_noise)
            cleaned_concat = cleaner.scores(noisy_concat)
            cleaned_whole = cleaner.scores(noisy_whole)

            agg_cl_concat = combine_measures(aggregate_ratings(cleaned_concat))
            agg_cl_whole = aggregate_ratings(cleaned_whole)

            rho_clconcat_whole, p = spearmanr(agg_cl_concat['mean'], agg_whole['mean'])
            rho_concat_clwhole, p = spearmanr(agg_concat['mean'], agg_cl_whole['mean'])
            rho_clconcat_clwhole, p = spearmanr(agg_cl_concat['mean'], agg_cl_whole['mean'])

            output.append({
                'noise': str(percent_noise),
                'zscore': zscore,
                'trial': i + 1,
                'concat*-whole':  rho_clconcat_whole,
                'concat-whole*':  rho_concat_clwhole,
                'concat*-whole*': rho_clconcat_clwhole
            })

            pb.incr()
            if i % 5 == 0 : pb.errput()
        pb.errput()


to_graph = melt(pd.DataFrame(output), id_vars=("noise", "zscore", "trial"))
#from rplots import line_plot

#line_plot(OUTPUT_FOLDER + "zscores.pdf", to_graph, 'zscore', 'value', 'noise')
to_graph.to_csv(sys.stdout, index=False)






