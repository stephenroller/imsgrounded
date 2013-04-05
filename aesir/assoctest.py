#!/usr/bin/env python

import argparse
import codecs
import numpy as np
import logging
from itertools import izip
from nicemodel import load_labels
from aesir import row_norm

logging.basicConfig(
    format="[ %(levelname)-10s %(module)-8s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m",
    level=logging.DEBUG)

ASSOC_FILE_NAME = '/home/01813/roller/tmp/imsgrounded/data/associations/big_assoc_vectors.txt'
KLDIV_EPS = 1e-8

class mdict(dict):
    def __setitem__(self, key, value):
        """add the given value to the list of values for this key"""
        self.setdefault(key, []).append(value)


def norm1(v):
    return v / np.sum(v)

def kldiv(p, q):
    return np.sum([pi * np.log2(pi / qi) for pi, qi in zip(p, q) if pi > KLDIV_EPS or qi > KLDIV_EPS])

def symkldiv(p, q):
    return kldiv(p, q) + kldiv(q, p)

def jsdiv(p, q):
    M = 0.5 * (p + q)
    return 0.5 * kldiv(p, M) + 0.5 * kldiv(q, M)

def load_associations():
    assocs = []
    with codecs.getreader('utf-8')(open(ASSOC_FILE_NAME)) as f:
        for line in f:
            line = line.rstrip()
            target, assoc, count = line.split("\t")
            count = int(count)
            for n in xrange(count):
                assocs.append((target, assoc))
    return assocs

def percentile_ranked(similarities):
    ranked_sims = np.zeros(len(similarities))
    cached = {}
    for i, s in enumerate(similarities):
        if s in cached:
            c = cached[s]
        else:
            c = np.sum(similarities >= s)
            cached[s] = c
        ranked_sims[i] = float(c)/len(similarities)

    return ranked_sims




def main():
    parser = argparse.ArgumentParser(description='Checks for prediction of association norms.')
    parser.add_argument('--model', '-m', metavar='FILE',
                        help='The saved model.')
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    parser.add_argument('--features', '-f', metavar='FILE',
                        help='The feature labels.')
    parser.add_argument('--docs', '-D', metavar='FILE',
                        help='Output the document distributions for these documents.')
    parser.add_argument('--docids', '-d', metavar='FILE',
                        help='The document labels.')
    args = parser.parse_args()

    model = np.load(args.model)
    phi = row_norm(model["phi"].T)
    pi = model["pi"]

    label_vocab = load_labels(args.vocab)
    #docids = codecs.getreader('utf-8')(open(args.docids)).readlines()

    phi_nn = { w[:w.rindex('/')] : i for i, w in label_vocab.iteritems() if '/NN' in w }

    nopos_labels = mdict()
    for i, v in label_vocab.iteritems():
        nopos = v[:v.rindex('/')]
        nopos_labels[nopos] = i

    assocs = load_associations()
    logging.info("assoc's loaded.")
    to_compute_similarities = set(t for t, a in assocs)

    ranked_sims = {}

    logging.info("compute similarities...")

    for z, w_i in enumerate(to_compute_similarities):
        if w_i not in phi_nn:
            continue
        i = phi_nn[w_i]
        w_i_dist = norm1(phi[i])
        similarities = np.array([jsdiv(w_i_dist, w_j_dist) for w_j_dist in phi])
        ranked_sims[w_i] = percentile_ranked(similarities)
        logging.debug("%d / %d done." % (z + 1, len(to_compute_similarities)))

    logging.info("finished computing similarities.")

    measures = []
    for t, a in assocs:
        if t not in ranked_sims or a not in nopos_labels:
            continue
        ranked = ranked_sims[t]
        m = max(ranked[i] for i in nopos_labels[a])
        measures.append(m)

    measures = np.array(measures)
    print "mean: %f" % measures.mean()
    print "std: %f" % measures.std()


    np.savez_compressed("assoc.npz", measures=measures)




if __name__ == '__main__':
    main()



