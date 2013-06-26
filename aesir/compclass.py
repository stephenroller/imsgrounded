#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os.path
import argparse
import numpy as np
import pandas as pd
import scipy.stats
import logging
from itertools import combinations
from aesir import row_norm
from assoctest import jsdiv, kldiv, symkldiv
from nicemodel import load_labels

DATA_FOLDER = "/home/01813/roller/tmp/imsgrounded/data/"
COMP_HEAD   = "comp/comp-values_heads.tsv"
COMP_MODS   = "comp/comp-values_modifiers.tsv"
COMP_ALL    = "comp/comp-values_all_sorted.tsv"
DISCO_TRAIN = "other/disco/DISCo_num_DE_train.tsv"
DISCO_VALID = "other/disco/DISCo_num_DE_validation.tsv"
DISCO_TEST  = "other/disco/DISCo_num_DE_test.tsv"
SCHM280     = "other/Schm280.csv"

def utf8(s):
    return unicode(s, encoding='utf-8')

def col_norm(a):
    col_sums = a.sum(axis=0)
    return a / col_sums

def topic_lmi(phi):
    # lmi(p, x) = p(x, y) * pmi(x, y)
    #           = p(x, y) * log[ p(x | y) / p(x)]
    #           = p(x, y) * log[ p(x | y) ] - log [ p(x) ]
    #           = p(w, t) * log[ p(w | t) ] - log [ p(t) ]

    pwt = phi / phi.sum()
    pwgivent = row_norm(phi)
    pt = phi.sum(axis=1)
    pt = pt / pt.sum()
    pt = np.array([pt]).T

    pmi = np.log(pwgivent) - np.log(pt)
    lmi = pwt * pmi
    lmi[lmi < 0] = 0
    return lmi

def cos(v1, v2):
    v1 = v1 / np.sqrt(v1.dot(v1))
    v2 = v2 / np.sqrt(v2.dot(v2))
    return v1.dot(v2)

def _load_disco_raw(mode):
    if mode == "train":
        filename = os.path.join(DATA_FOLDER, DISCO_TRAIN)
    elif mode == "test":
        filename = os.path.join(DATA_FOLDER, DISCO_TEST)
    elif mode == "valid":
        filename = os.path.join(DATA_FOLDER, DISCO_VALID)
    elif mode == "trainvalid":
        l = _load_disco_raw("train")
        r = _load_disco_raw("test")
        return l.append(r)
    elif mode == "all":
        l = _load_disco_raw("trainvalid")
        r = _load_disco_raw("test")
        return l.append(r)
    else:
        raise ValueError("Data set can only be 'train', 'test', 'valid', 'trainvalid', or 'all'")

    # lexical substitution
    table = pd.read_table(filename, names=("pos", "pair", "similarity"), encoding="utf-8")

    spell_mappings = [
            ("&uuml;", u"ü"), ("&ouml;", u"ö"), ("&auml;", u"ä"),
            ("&Uuml;", u"Ü"), ("&Ouml;", u"Ö"), ("&Auml;", u"Ä"),
            ("%szlig;", u"ß"),
    ]

    for encoded, corrected in spell_mappings:
        table["pair"] = table["pair"].apply(lambda x: unicode(x).replace(unicode(encoded), corrected))

    return table

def load_disco(target_labels, mode="train+valid"):
    table = _load_disco_raw(mode)
    table["leftnopos"] = table["pair"].apply(lambda x: x.split(" ")[0])
    table["rightnopos"] = table["pair"].apply(lambda x: x.split(" ")[1])
    table["leftpos"] = table["pos"].apply(lambda x: x.split("_")[1])
    table["rightpos"] = table["pos"].apply(lambda x: x.split("_")[2])

    posmaps = {
        'ADJ' : ['ADJA', 'ADJD'],
        'NN' : ['NN'],
        'SUBJ' : ['NN'],
        'OBJ' : ['NN'],
        'V' : ['VVFIN', 'VVINF', 'VVPP', 'VVIZU', 'VVIMP'],
    }

    left_with_pos = []
    right_with_pos = []

    for i, r in table.iterrows():
        left = r["leftnopos"]
        for corpuspos in posmaps[r["leftpos"]]:
            if left + "/" + corpuspos in target_labels:
                left_with_pos.append(left + "/" + corpuspos)
                break
        else:
            left_with_pos.append("")

    for i, r in table.iterrows():
        right = r["rightnopos"]
        for corpuspos in posmaps[r["rightpos"]]:
            if right + "/" + corpuspos in target_labels:
                right_with_pos.append(right + "/" + corpuspos)
                break
        else:
            right_with_pos.append("")

    table["left"] = left_with_pos
    table["right"] = right_with_pos

    table = table[table["left"] != ""]
    table = table[table["right"] != ""]

    return table

def load_comp(target_labels, mode="all"):
    if mode == "head":
        table = pd.read_table(os.path.join(DATA_FOLDER, COMP_HEAD))
    elif mode == "mod":
        table = pd.read_table(os.path.join(DATA_FOLDER, COMP_MODS))
    elif mode == "all":
        table = pd.read_table(os.path.join(DATA_FOLDER, COMP_ALL))
    else:
        raise ValueError("Mode must be 'head', 'mod', or 'all'.")

    table = table[table['const'] != table['compound']]
    table["left"] = table["const"].apply(lambda x: utf8(x + "/NN"))
    table["right"] = table["compound"].apply(lambda x: utf8(x + "/NN"))
    table["similarity"] = table["mean"]

    return table

def load_schm280(target_labels):
    table = pd.read_table(os.path.join(DATA_FOLDER, SCHM280), encoding="utf-8")
    table["leftnopos"] = table["Translation Word 1"]
    table["rightnopos"] = table["Translation Word 2"]
    table["similarity"] = table["Human (mean)"]

    postypes =  ['NN', 'ADJA', 'ADJD', 'VVFIN', 'VVINF', 'VVPP', 'VVIZU', 'VVIMP']

    left_with_pos = []
    right_with_pos = []
    for i, r in table.iterrows():
        left = r["leftnopos"]
        for pos in postypes:
            if left + "/" + pos in target_labels:
                left_with_pos.append(left + "/" + pos)
                break
        else:
            left_with_pos.append("")

        right = r["rightnopos"]
        for pos in postypes:
            if right + "/" + pos in target_labels:
                right_with_pos.append(right + "/" + pos)
                break
        else:
            right_with_pos.append("")

    table["left"] = left_with_pos
    table["right"] = right_with_pos

    table = table[table["left"] != ""]
    table = table[table["right"] != ""]

    return table

def load_eval_table(target_labels, method):
    if method == 'disco':
        return load_disco(target_labels, 'all')
    elif method in ('discotrain', 'discotest', 'discovalid', 'discotrainvalid'):
        return load_disco(target_labels, method[6:])
    elif method == 'comp':
        return load_comp(target_labels, 'all')
    elif method in ('comphead', 'compmod'):
        return load_comp(target_labels, method[5:])
    elif method == 'schm280':
        return load_schm280(target_labels)
    else:
        raise ValueError("Method '%s' not supported." % method)

def main():
    parser = argparse.ArgumentParser(description='Outputs a human readable model.')
    parser.add_argument('models', metavar='FILE', help='The saved models.', nargs='+')
    parser.add_argument('--eval', '-e', metavar='EVALDATA', default='comp',
                        choices=['disco', 'discotrain', 'discovalid', 'discotest', 'discotrainvalid',
                                 'comp', 'compmod', 'comphead',
                                 'schm280'],
                        help="The data set to evaluate against.")
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    parser.add_argument('--acc-thresh', type=float, default=0,
                        help="Don't include pairwise comparisons whose judgements are closer than this threshold.")
    args = parser.parse_args()

    vocab_labels = load_labels(args.vocab)
    vocab_labels = {w : i for i, w in vocab_labels.iteritems()}
    eval_tab = load_eval_table(vocab_labels, args.eval)

    model_evaluations = []
    for model in args.models:
        logging.info("Processing model '%s'..." % model)
        m = np.load(model)
        k = m['k']
        ll = np.mean('loglikelihoods' in m and m['loglikelihoods'][-5:] or m['perwordbounds'][-5:])
        iter = m['max_iteration']
        time = np.sum(m['timediffs'])
        phi = np.ascontiguousarray(m['phi'])
        topic_normed = row_norm(phi)
        word_normed = col_norm(phi)

        lmid = topic_lmi(phi)

        model_eval = dict(k=k, ll=ll, iter=iter, time=time, 
                          alpha=m['alpha'], eta=m['eta'], mu=m['mu'],
                          eval=args.eval, input=m['input_filename'])

        similarities = {}
        for i, pair in eval_tab.iterrows():
            try:
                left_id = vocab_labels[pair['left']]
                right_id = vocab_labels[pair['right']]
            except KeyError:
                continue

            pair_k = (pair['left'], pair['right'])
            right_given_left = np.dot(topic_normed[:,right_id], word_normed[:,left_id])
            left_given_right = np.dot(topic_normed[:,left_id], word_normed[:,right_id])
            jsdiv_sim = jsdiv(word_normed[:,right_id], word_normed[:,left_id])
            symkldiv_sim = symkldiv(word_normed[:,right_id], word_normed[:,left_id])
            kldiv1 = kldiv(word_normed[:,right_id], word_normed[:,left_id])
            kldiv2 = kldiv(word_normed[:,left_id], word_normed[:,right_id])
            cos_lmi = cos(lmid[:,right_id], lmid[:,left_id])

            similarities[pair_k] = {'right':  pair['right'],
                                    'left':     pair['left'],
                                    'right|left': right_given_left,
                                    'left|right': left_given_right,
                                    'jsdiv':     jsdiv_sim,
                                    'symkldiv':  symkldiv_sim,
                                    'kldiv1':    kldiv1,
                                    'kldiv2':    kldiv2,
                                    'coslmi':    cos_lmi,
                                    'human':     pair['similarity'],
                                    }

        # let's compute spearman's rho for each of the measures:
        tmp = pd.DataFrame(similarities.values())
        for m in ['right|left', 'left|right', 'jsdiv', 'symkldiv', 'kldiv1', 'kldiv2', 'coslmi']:
            rho, p = scipy.stats.spearmanr(tmp[m], tmp['human'])
            model_eval['rho_' + m] = rho
            model_eval['p_' + m] = p
            model_eval['n'] = len(tmp[m])

        # okay now let's do accuracy style measures
        baseline_correct = 0
        jsdiv_correct = 0
        symkldiv_correct = 0
        kldiv1_correct = 0
        kldiv2_correct = 0
        rightleft_correct = 0
        leftright_correct = 0
        lmicos_correct = 0
        pairs_compared = 0.0
        for (i, pair1), (j, pair2) in combinations(eval_tab.iterrows(), 2):
            if pair1['similarity'] == pair2['similarity'] or abs(pair1['similarity'] - pair2['similarity']) < 1.0:
                continue

            try:
                pair1_k = (pair1['left'], pair1['right'])
                similarities1 = similarities[pair1_k]
                pair2_k = (pair2['left'], pair2['right'])
                similarities2 = similarities[pair2_k]
            except KeyError:
                continue

            gold = pair1['similarity'] < pair2['similarity']

            pairs_compared += 1
            baseline_correct += (gold == 1)

            jsdiv_correct += (gold == (similarities1['jsdiv'] > similarities2['jsdiv']))
            symkldiv_correct += (gold == (similarities1['symkldiv'] > similarities2['symkldiv']))
            rightleft_correct += (gold == (similarities1['right|left'] < similarities2['right|left']))
            leftright_correct += (gold == (similarities1['left|right'] < similarities2['left|right']))
            lmicos_correct += (gold == (similarities1['coslmi'] < similarities2['coslmi']))

        prod = 100.0 / pairs_compared
        model_eval['filename'] = model
        model_eval['model_type'] = os.path.dirname(model)
        model_eval['acc_baseline'] = baseline_correct / pairs_compared
        model_eval['acc_jsdiv'] = jsdiv_correct / pairs_compared
        model_eval['acc_symkldiv'] = jsdiv_correct / pairs_compared
        model_eval['acc_right|left'] = rightleft_correct / pairs_compared
        model_eval['acc_left|right'] = leftright_correct / pairs_compared
        model_eval['acc_coslmi'] = lmicos_correct / pairs_compared

        model_evaluations.append(model_eval)

    pd.DataFrame(model_evaluations).to_csv(sys.stdout, index=False)





if __name__ == '__main__':
    main()


