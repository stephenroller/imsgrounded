#!/usr/bin/env python

import codecs
import pandas as pd
import numpy as np
import logging

from aesir import row_norm
from assoctest import percentile_ranked, symkldiv, jsdiv, calc_similarities
from nicemodel import load_labels
from scipy.stats import spearmanr
from itertools import chain

COMP_FILE = '/home/01813/roller/tmp/imsgrounded/data/comp/comp-values_all_sorted.tsv'


def utfopenwrite(filename):
    return codecs.getwriter('utf-8')(open(filename, 'w'))

def utfopen(filename):
    return codecs.getreader('utf-8')(open(filename))

def main():
    comps = pd.read_table(COMP_FILE)
    comps = comps[comps.compound != comps.const]
    calcsims = list(chain(*zip(comps['compound'], comps['const'])))
    label_vocab = load_labels("target-labels.txt")
    phi_nn = { w[:w.rindex('/')] : i for i, w in label_vocab.iteritems() if '/NN' in w }
    model = np.load("model_250.npy.npz")
    phi = row_norm(model["phi"].T)

    ranked_sims = {}
    done = set()
    for z, word in enumerate(calcsims):
        if word in done or word not in phi_nn:
            continue
        done.add(word)
        i = phi_nn[word]
        w_dist = phi[i]
        sims = calc_similarities(w_dist, phi)
        percentile = percentile_ranked(sims)
        ranked_sims[word] = sims
        logging.info("Done with %d/%d [%s]" % (z + 1, len(calcsims), word))

    ratings_compound = []
    ratings_const = []
    gold = []
    for compound, const, mean in zip(comps.compound, comps.const, comps['mean']):
        if compound not in ranked_sims or const not in ranked_sims:
            continue
        ranked_sims_compound = ranked_sims[compound]
        ranked_sims_const = ranked_sims[const]

        ratings_compound.append(ranked_sims_compound[phi_nn[const]])
        ratings_const.append(ranked_sims_const[phi_nn[compound]])
        gold.append(mean)

    print ratings_compound
    print ratings_const
    print gold
    print spearmanr(ratings_compound, gold)
    print spearmanr(ratings_const, gold)




if __name__ == '__main__':
    main()



