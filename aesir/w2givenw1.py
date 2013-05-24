#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import codecs
from scipy.stats import spearmanr
from nicemodel import load_labels

def row_norm(a):
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]

def col_norm(a):
    col_sums = a.sum(axis=0)
    return a / col_sums

model_file = sys.argv[1]
comp_file = '/home/01813/roller/tmp/imsgrounded/data/comp/comp-values_all_sorted.tsv'
target_labels_file = '/scratch/01813/roller/corpora/webko/TermDoc/target-labels.txt'

vocab_labels = load_labels(target_labels_file)
vocab_labels = {w : i for i, w in vocab_labels.iteritems()}

phi = np.ascontiguousarray(np.load(model_file)['phi'])

topic_normed = col_norm(phi)
word_normed = row_norm(phi)

comp_tab = pd.read_table(comp_file, encoding='utf-8')
comp_tab = comp_tab[comp_tab['const'] != comp_tab['compound']]

compound = []
const = []
ratings = []
w2givenw1 = []
w1givenw2 = []
for i, row in comp_tab.iterrows():
    try:
        cmpd_id = vocab_labels[row['compound'] + '/NN']
        const_id = vocab_labels[row['const'] + '/NN']
    except KeyError:
        pass

    compound.append(row['compound'])
    const.append(row['const'])

    top_given_w1 = word_normed[:,cmpd_id]
    w2_given_top = topic_normed[:,const_id]

    top_given_w2 = word_normed[:,const_id]
    w1_given_top = topic_normed[:,cmpd_id]

    ratings.append(row['mean'])
    w2givenw1.append(np.dot(top_given_w1, w2_given_top))
    w1givenw2.append(np.dot(top_given_w2, w1_given_top))

disp_tab = pd.DataFrame(dict(compound=compound, const=const, ratings=ratings, w2givenw1=w2givenw1, w1givenw2=w1givenw2))
#print disp_tab.sort('ratings').to_string()
disp_tab.to_csv(sys.argv[2], index=False, encoding='utf-8')

print
print "shape =", disp_tab.shape
print "cpmd|const =", spearmanr(ratings, w1givenw2)
print "const|cmpd =", spearmanr(ratings, w2givenw1)
