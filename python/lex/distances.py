#!/usr/bin/env python

"""
Performs distance metrics between all the specified word pairs.

Word pairs should be fed in from a file or stdin and tab seperated.
The vector spaces should also be fed in.

One should note that this loads the entire vector space into memory.
Therefore one should probably use extract_vectors first to make sure
that you are only loading what is needed into memory.
"""

import sys
import argparse
import logging
from math import sqrt
from itertools import groupby

import scipy.stats
import pandas as pd

import tsv
from util import remove_pos, normalize, openfile, read_vector_file, df_remove_pos, norm2

# this will automatically be filled in by the @distance_metric
# decorator
METRICS = dict()

def distance_metric(func, name=None):
    func.name = name and name or func.func_name
    METRICS[func.name] = func
    return func

# distance measures
@distance_metric
def cosine(vec1, vec2):
    return norm2(vec1).dot(norm2(vec2))

@distance_metric
def euclid(vec1, vec2):
    d = vec1 - vec2
    return sqrt(d.dot(d))

@distance_metric
def normeuclid(vec1, vec2):
    n1 = norm2(vec1)
    n2 = norm2(vec2)
    return euclid(n1, n2)

@distance_metric
def jaccard(vec1, vec2):
    """Jaccard overlap of key presence."""
    leftkeys = set(vec1.nonzero()[0])
    rightkeys = set(vec2.nonzero()[0])
    inter = leftkeys.intersection(rightkeys)
    outer = leftkeys.union(rightkeys)
    return float(len(inter)) / len(outer)

@distance_metric
def dot(vec1, vec2):
    return vec1.dot(vec2)

@distance_metric
def spearman(vec1, vec2, also_give_p=False):
    """
    Considers the vectors to be ranked lists and calculates
    spearmann's rho.

    If also_give_p is False, then it just returns rho. Otherwise,
    it returns the (rho, pvalue) tuple.
    """
    rho, p = scipy.stats.spearmanr(vec1, vec2)
    if also_give_p:
        return (rho, p)
    else:
        return rho

# okay onto drivers and such
def calculate_distance_metrics(vecspace, pairs, metrics):
    for word1, word2 in pairs:
        try:
            vec1 = vecspace[word1].to_dense()
        except KeyError:
            logging.warning("Couldn't find vector for '%s'. Skipping..." % word1)
            continue
        try:
            vec2 = vecspace[word2].to_dense()
        except KeyError:
            logging.warning("Couldn't find vector for '%s'. Skipping..." % word2)
            continue

        results = {dm.name : dm(vec1, vec2) for dm in metrics}
        results['left'] = word1
        results['right'] = word2

        yield results


def read_pairs(file):
    return list(tsv.read_tsv(file, False))

def main():
    parser = argparse.ArgumentParser(
                description='Computes distances between word pairs.')
    parser.add_argument("--input", "-i", type=openfile,
                        metavar="FILE", help='The input vector space.')
    parser.add_argument('--pairsfile', '-w', metavar='FILE', type=openfile,
                        help='The list of tab separated word pairs.')
    parser.add_argument('words', nargs='*', metavar='WORD',
                        help=('Additional word pairs specified at the command line.  '
                              'Every two specifies an additional word pair. Must be '
                              'given an even number of words.'))
    parser.add_argument('--pos', '-p', action='store_true',
                        help='Marks that the word pairs are POS tagged.')
    parser.add_argument('--distance-metric', '-d', action='append',
                        choices=METRICS.keys(),
                        help='Distance metrics to use.')
    args = parser.parse_args()

    pairs = set()
    if args.pairsfile:
        pairs.update(read_pairs(args.pairsfile))

    if len(args.words) % 2 != 0:
        raise ValueError, "You need to specify an even number of pair words."

    if not args.distance_metric:
        args.distance_metric = ['cosine']

    pairs.update(zip(args.words[::2], args.words[1::2]))

    vecspace = read_vector_file(args.input)
    if not args.pos:
        # need to strip the POS from the targets
        vecspace = df_remove_pos(vecspace)

    distance_metric_names = args.distance_metric
    distance_metrics = [METRICS[name] for name in distance_metric_names]
    out_tsv = calculate_distance_metrics(vecspace, pairs, distance_metrics)

    tsv.print_tsv(out_tsv, headers=['left', 'right'] + distance_metric_names)




if __name__ == '__main__':
    main()


