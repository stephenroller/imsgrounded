#!/usr/bin/env python

import sys
import argparse

import pandas as pd
from pandas.core.reshape import melt

from collections import Counter

from whiten import whiten
from math import sqrt, isnan
from random import choice


HEAD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_head.csv'
MOD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_const.csv'
WHOLE_FILE = '/Users/stephen/Working/imsgrounded/data/comp/amt_reshaped.csv'



class Prediction(object):
    def __init__(self, data):
        self.data = data

    def predict(self, item, subject):
        return 0

class MostCommonPrediction(Prediction):
    def __init__(self, data):
        judgement_columns = data.columns[2:]
        judgements = [v for v in data.values.flatten() if not isinstance(v, str) and not isnan(v)]
        counts = Counter(judgements)
        self.guess = max(counts.keys(), key=counts.__getitem__)

    def predict(self, item, subject):
        return self.guess

class ItemAveragePrediction(Prediction):
    def __init__(self, data):
        judgement_columns = data.columns[2:]
        self.item_avgs = data[judgement_columns].T.mean()

    def predict(self, item, subject):
        return self.item_avgs[item]


class SubjectAveragePrediction(Prediction):
    def __init__(self, data):
        self.subj_avgs = data.mean()

    def predict(self, item, subject):
        return self.subj_avgs[subject]

class ItemSubjectAveragePrediction(Prediction):
    def __init__(self, data):
        self.iap = ItemAveragePrediction(data)
        self.sap = SubjectAveragePrediction(data)

    def predict(self, item, subject):
        return sqrt(self.iap.predict(item, subject) *
                    self.sap.predict(item, subject))


class SvdPrediction(Prediction):
    def __init__(self, data, k):
        self.whitened = whiten(data, k)

    def predict(self, item, subject):
        return self.whitened[subject][item]


def predict(data, predictor, num_blank=50, trials=100):
    judgement_columns = data.columns[2:]
    measurements = []
    for t in xrange(trials):
        # sys.stderr.write("Trial %d of %d (%s): \n" % (t, trials, predictor))
        # pick some random blanks
        copy = data.copy()
        for j in judgement_columns:
            copy[j] = copy[j].astype('float')
        nulls = pd.DataFrame({c : data[c].isnull() for c in data.columns})
        blanked = []
        while len(blanked) < num_blank:
            subj = choice(judgement_columns)
            item = choice(data.index)
            if nulls[subj][item]:
                continue
            copy[subj][item] = float('nan')
            nulls[subj][item] = True
            blanked.append((subj, item))

        model = predictor(copy)
        predictions = pd.Series([model.predict(item, subj) for subj, item in blanked])
        correct = pd.Series([data[subj][item] for subj, item in blanked])
        #print correct

        rmse = sqrt(((correct - predictions) ** 2).mean())
        measurements.append(rmse)

    return pd.Series(measurements).mean()











heads = pd.read_csv(HEAD_FILE)
mods = pd.read_csv(MOD_FILE)
whole = pd.read_csv(WHOLE_FILE)

# we can only work with the intersection of all 3 files in terms of
# judgements
good_compounds = reduce(
    set.intersection,
    [set(x.compound) for x in [heads, mods, whole]]
)
heads, mods, whole = [
    d[d.compound.map(good_compounds.__contains__)]
    for d in [heads, mods, whole]
]

concatted = pd.concat([heads, mods], ignore_index=True)

datasets = [
    ('head', heads),
    ('mod', mods),
    ('whole', whole),
    ('head+mod', concatted)
]

for dataset_name, dataset in datasets:
    print "DATA:", dataset_name
    print "  common:", predict(dataset, MostCommonPrediction)
    print "  item avg:", predict(dataset, ItemAveragePrediction)
    print "  subj avg:", predict(dataset, SubjectAveragePrediction)
    print "  item*subj avg:", predict(dataset, ItemSubjectAveragePrediction)
    for k in xrange(1, 11):
        model = lambda d: SvdPrediction(d, k)
        print "  svd", k, ":", predict(dataset, model)
    print



