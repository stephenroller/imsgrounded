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
import heapq
import logging
from math import sqrt
from itertools import groupby

import scipy.stats

import tsv
from util import remove_pos, normalize, openfile

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
    n1 = sqrt(sum(v ** 2 for v in vec1.values()))
    n2 = sqrt(sum(v ** 2 for v in vec2.values()))
    intersection = set(vec1.keys()).intersection(set(vec2.keys()))
    dotprod = sum(vec1[k] * vec2[k] for k in intersection)
    return dotprod / (n1 * n2)

@distance_metric
def euclid(vec1, vec2):
    keys = set(vec1.keys()).union(set(vec2.keys()))
    totalsum = sum((vec1.get(k, 0) - vec2.get(k, 0)) ** 2 for k in keys)
    return sqrt(totalsum)

@distance_metric
def normeuclid(vec1, vec2):
    n1 = sqrt(sum(v ** 2 for v in vec1.values()))
    n2 = sqrt(sum(v ** 2 for v in vec2.values()))
    keys = set(vec1.keys()).union(set(vec2.keys()))
    totalsum = sum((vec1.get(k, 0) / n1 - vec2.get(k, 0) / n2) ** 2 for k in keys)
    return totalsum

@distance_metric
def jaccard(vec1, vec2):
    """Jaccard overlap of key presence."""
    inter = set(vec1.keys()).intersection(set(vec2.keys()))
    outer = set(vec1.keys()).union(set(vec2.keys()))
    return float(len(inter)) / len(outer)

@distance_metric
def dot(vec1, vec2):
    intersection = set(vec1.keys()).intersection(set(vec2.keys()))
    dotprod = sum(vec1[k] * vec2[k] for k in intersection)
    return dotprod

@distance_metric
def spearman(vec1, vec2, also_give_p=False):
    """
    Considers the vectors to be ranked lists and calculates
    spearmann's rho.

    If also_give_p is False, then it just returns rho. Otherwise,
    it returns the (rho, pvalue) tuple.
    """
    keys = set(vec1.keys()).union(set(vec2.keys()))
    lst1 = [vec1.get(k, 0) for k in keys]
    lst2 = [vec2.get(k, 0) for k in keys]
    rho, p = scipy.stats.spearmanr(lst1, lst2)
    if also_give_p:
        return (rho, p)
    else:
        return rho


# okay onto drivers and such

def calculate_distance_metrics(corpus_mem, pairs, metrics):
    for word1, word2 in pairs:
        try:
            vec1 = corpus_mem[word1]
        except KeyError:
            logging.warning("Couldn't find vector for '%s'. Skipping..." % word1)
        try:
            vec2 = corpus_mem[word2]
        except KeyError:
            logging.warning("Couldn't find vector for '%s'. Skipping..." % word2)

        results = [dm(vec1, vec2) for dm in metrics]

        yield [word1, word2] + results


def read_pairs(file):
    return list(tsv.read_tsv(file, False))

def main():
    parser = argparse.ArgumentParser(
                description='Computes distances between word pairs.')
    parser.add_argument("--input", "-i", action="append", type=openfile,
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

    corpus = tsv.read_many_tsv(args.input, ('target', 'context', 'value'), parsers={'value': float})
    corpus_mem = tsv_to_dict(corpus, keep_pos=args.pos)

    distance_metric_names = args.distance_metric
    distance_metrics = [METRICS[name] for name in distance_metric_names]
    out_tsv = calculate_distance_metrics(corpus_mem, pairs, distance_metrics)

    tsv.print_tsv(out_tsv, headers=['left', 'right'] + distance_metric_names)




if __name__ == '__main__':
    main()


