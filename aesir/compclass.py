#!/usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from aesir import row_norm
from assoctest import jsdiv
from nicemodel import load_labels

def col_norm(a):
    col_sums = a.sum(axis=0)
    return a / col_sums

comp_file = '/home/01813/roller/tmp/imsgrounded/data/comp/comp-values_all_sorted.tsv'

def main():
    parser = argparse.ArgumentParser(description='Outputs a human readable model.')
    parser.add_argument('--model', '-m', metavar='FILE',
                        help='The saved model.')
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    args = parser.parse_args()

    phi = np.ascontiguousarray(np.load(args.model)['phi'])
    topic_normed = col_norm(phi)
    word_normed = row_norm(phi)

    vocab_labels = load_labels(args.vocab)
    vocab_labels = {w : i for i, w in vocab_labels.iteritems()}

    comp_tab = pd.read_table(comp_file, encoding='utf-8')
    comp_tab = comp_tab[comp_tab['const'] != comp_tab['compound']]

    baseline_correct = 0
    jsdiv_correct = 0
    w2w1_correct = 0
    w1w2_correct = 0
    pairs_compared = 0
    for (i, pair1), (j, pair2) in combinations(comp_tab.iterrows(), 2):
        try:
            cnst1_id = vocab_labels[pair1['const'] + '/NN']
            cnst2_id = vocab_labels[pair2['const'] + '/NN']
            cmpd1_id = vocab_labels[pair1['compound'] + '/NN']
            cmpd2_id = vocab_labels[pair2['compound'] + '/NN']
        except KeyError:
            continue

        if pair1['mean'] == pair2['mean']:
            continue

        gold = pair1['mean'] < pair2['mean']

        cmpd1_given_cnst1 = np.dot(topic_normed[:,cmpd1_id], word_normed[:,cnst1_id])
        cmpd2_given_cnst2 = np.dot(topic_normed[:,cmpd2_id], word_normed[:,cnst2_id])
        cnst1_given_cmpd1 = np.dot(topic_normed[:,cnst1_id], word_normed[:,cmpd1_id])
        cnst2_given_cmpd2 = np.dot(topic_normed[:,cnst2_id], word_normed[:,cmpd2_id])

        pairs_compared += 1
        baseline_correct += (gold == 1)

        cnst1 = phi[:,cnst1_id]
        cnst2 = phi[:,cnst2_id]
        cmpd1 = phi[:,cmpd1_id]
        cmpd2 = phi[:,cmpd2_id]
        jsdiv_correct += (gold == (jsdiv(cnst1, cmpd1) > jsdiv(cnst2, cmpd2)))
        w2w1_correct += (gold == (cnst1_given_cmpd1 < cnst2_given_cmpd2))
        w1w2_correct += (gold == (cmpd1_given_cnst1 < cmpd2_given_cnst2))

    print "Acc basel: %5.2f%%" % (100.0 * float(baseline_correct) / pairs_compared)
    print "Acc jsdiv: %5.2f%%" % (100.0 * float(jsdiv_correct) / pairs_compared)
    print "Acc w2|w1: %5.2f%%" % (100.0 * float(w2w1_correct) / pairs_compared)
    print "Acc w1|w2: %5.2f%%" % (100.0 * float(w1w2_correct) / pairs_compared)





if __name__ == '__main__':
    main()


