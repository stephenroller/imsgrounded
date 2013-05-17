#!/usr/bin/env python

import argparse
import codecs
import numpy as np
import logging
import scipy.stats
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
    mask = (p > KLDIV_EPS) | (q > KLDIV_EPS)
    mp = p[mask]
    mq = q[mask]
    return np.dot(mp, np.log2(mp / mq))

def symkldiv(p, q):
    return kldiv(p, q) + kldiv(q, p)

jsdiv_cache = {}
def cached_jsdiv(i, j, p, q):
    if (i, j) in jsdiv_cache:
        return jsdiv_cache[(i, j)]
    elif (j, i) in jsdiv_cache:
        return jsdiv_cache[(j, i)]
    else:
        d = jsdiv(p, q)
        x, y = i < j and (i, j) or (j, i)
        jsdiv_cache[(x, y)] = d
        return d


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
            assocs.append((target, assoc, count))
    return assocs

def percentile_ranked(similarities):
    ranked = np.ceil(len(similarities) + 1 - scipy.stats.rankdata(similarities))
    return ranked/len(similarities)

def calc_similarities(dist, other_dists, measure=jsdiv):
    return np.array([measure(dist, other) for other in other_dists])

def main():
    parser = argparse.ArgumentParser(description='Checks for prediction of association norms.')
    parser.add_argument('--model', '-m', metavar='FILE',
                        help='The saved model.')
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    parser.add_argument('--features', '-f', metavar='FILE',
                        help='The feature labels.')
    #parser.add_argument('--docs', '-D', metavar='FILE',
    #                    help='Output the document distributions for these documents.')
    #parser.add_argument('--docids', '-d', metavar='FILE',
    #                    help='The document labels.')
    args = parser.parse_args()

    model = np.load(args.model)
    phi = row_norm(model["phi"].T)
    #pi = safe_pi_read(args.model)

    label_vocab = load_labels(args.vocab)
    #docids = codecs.getreader('utf-8')(open(args.docids)).readlines()

    phi_nn = { w[:w.rindex('/')] : i for i, w in label_vocab.iteritems() if '/NN' in w }

    nopos_labels = mdict()
    for i, v in label_vocab.iteritems():
        nopos = v[:v.rindex('/')]
        nopos_labels[nopos] = i

    assocs = load_associations()
    to_compute_similarities = list(set(t for t, a, c in assocs))

    ranked_sims = {}

    logging.info("compute similarities...")

    for z, w_i in enumerate(to_compute_similarities):
        if w_i not in phi_nn:
            continue
        i = phi_nn[w_i]
        w_i_dist = norm1(phi[i])
        similarities = np.array([cached_jsdiv(i, j, w_i_dist, w_j_dist) for j, w_j_dist in enumerate(phi)])
        ranked_sims[w_i] = percentile_ranked(similarities)
        logging.debug("%d / %d done." % (z + 1, len(to_compute_similarities)))

    logging.info("finished computing similarities.")

    measures = []
    oov_count = 0
    noov_count = 0
    for t, a, c in assocs:
        if t not in ranked_sims or a not in nopos_labels:
            oov_count += 1
            continue
        noov_count += 1
        ranked = ranked_sims[t]
        m = max(ranked[i] for i in nopos_labels[a])
        measures += [m] * c

    measures = np.array(measures)
    print "mean: %f" % measures.mean()
    print "std: %f" % measures.std()
    print "oov: %d" % oov_count
    print "len(measures) = %d" % len(measures)
    print "# hit: %d" % noov_count
    print "Percentiles [.05, .10, .25, .5, .75, .90, .95] ="
    print "     [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]" % tuple([scipy.stats.scoreatpercentile(measures, p) for p in [5, 10, 25, 50, 75, 90, 95]])




if __name__ == '__main__':
    main()



