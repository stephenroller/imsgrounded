#!/usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats
import logging
from itertools import combinations
from aesir import row_norm
from assoctest import jsdiv, kldiv
from nicemodel import load_labels

def col_norm(a):
    col_sums = a.sum(axis=0)
    return a / col_sums

comp_file = '/home/01813/roller/tmp/imsgrounded/data/comp/comp-values_all_sorted.tsv'

def main():
    parser = argparse.ArgumentParser(description='Outputs a human readable model.')
    parser.add_argument('models', metavar='FILE', help='The saved models.', nargs='+')
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    parser.add_argument('--acc-thresh', type=float, default=0,
                        help="Don't include pairwise comparisons whose judgements are closer than this threshold.")
    args = parser.parse_args()

    comp_tab = pd.read_table(comp_file, encoding='utf-8')
    comp_tab = comp_tab[comp_tab['const'] != comp_tab['compound']]

    vocab_labels = load_labels(args.vocab)
    vocab_labels = {w : i for i, w in vocab_labels.iteritems()}

    model_evaluations = []
    for model in args.models:
        logging.info("Processing model '%s'..." % model)
        m = np.load(model)
        k = m['k']
        ll = np.mean('loglikelihoods' in m and m['loglikelihoods'][-5:] or m['perwordbounds'][-5:])
        iter = m['max_iteration']
        time = np.sum(m['timediffs'])
        phi = np.ascontiguousarray(m['phi'])
        topic_normed = col_norm(phi)
        word_normed = row_norm(phi)

        model_eval = {'k': k, 'll': ll, 'iter': iter, 'time': time}

        similarities = {}
        for i, pair in comp_tab.iterrows():
            try:
                cnst_id = vocab_labels[pair['const'] + '/NN']
                cmpd_id = vocab_labels[pair['compound'] + '/NN']
            except KeyError:
                continue

            pair_k = (pair['const'], pair['compound'])
            cmpd_given_cnst = np.dot(topic_normed[:,cmpd_id], word_normed[:,cnst_id])
            cnst_given_cmpd = np.dot(topic_normed[:,cnst_id], word_normed[:,cmpd_id])
            jsdiv_sim = jsdiv(word_normed[:,cmpd_id], word_normed[:,cnst_id])
            kldiv1 = kldiv(word_normed[:,cmpd_id], word_normed[:,cnst_id])
            kldiv2 = kldiv(word_normed[:,cnst_id], word_normed[:,cmpd_id])

            similarities[pair_k] = {'compound':  pair['compound'],
                                    'const':     pair['const'],
                                    'cmpd|cnst': cmpd_given_cnst,
                                    'cnst|cmpd': cnst_given_cmpd,
                                    'jsdiv':     jsdiv_sim,
                                    'kldiv1':    kldiv1,
                                    'kldiv2':    kldiv2,
                                    'human':     pair['mean'],
                                    }

        # let's compute spearman's rho for each of the measures:
        tmp = pd.DataFrame(similarities.values())
        for m in ['cmpd|cnst', 'cnst|cmpd', 'jsdiv', 'kldiv1', 'kldiv2']:
            rho, p = scipy.stats.spearmanr(tmp[m], tmp['human'])
            model_eval['rho_' + m] = rho
            model_eval['p_' + m] = p



        # okay now let's do accuracy style measures
        baseline_correct = 0
        jsdiv_correct = 0
        kldiv1_correct = 0
        kldiv2_correct = 0
        cmpdcnst_correct = 0
        cnstcmpd_correct = 0
        pairs_compared = 0.0
        for (i, pair1), (j, pair2) in combinations(comp_tab.iterrows(), 2):
            if pair1['mean'] == pair2['mean'] or abs(pair1['mean'] - pair2['mean']) < 1.0:
                continue

            try:
                pair1_k = (pair1['const'], pair1['compound'])
                similarities1 = similarities[pair1_k]
                pair2_k = (pair2['const'], pair2['compound'])
                similarities2 = similarities[pair2_k]
            except KeyError:
                continue

            gold = pair1['mean'] < pair2['mean']

            pairs_compared += 1
            baseline_correct += (gold == 1)

            jsdiv_correct += (gold == (similarities1['jsdiv'] > similarities2['jsdiv']))
            cmpdcnst_correct += (gold == (similarities1['cmpd|cnst'] < similarities2['cmpd|cnst']))
            cnstcmpd_correct += (gold == (similarities1['cnst|cmpd'] < similarities2['cnst|cmpd']))

        prod = 100.0 / pairs_compared
        model_eval['acc_baseline'] = baseline_correct / pairs_compared
        model_eval['acc_jsdiv'] = jsdiv_correct / pairs_compared
        model_eval['acc_cmpd|cnst'] = cmpdcnst_correct / pairs_compared
        model_eval['acc_cnst|cmpd'] = cnstcmpd_correct / pairs_compared

        model_evaluations.append(model_eval)

    pd.DataFrame(model_evaluations).to_csv(sys.stdout, index=False)





if __name__ == '__main__':
    main()


