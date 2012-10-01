#!/usr/bin/env python

"""
Script for concatenating vectors.
"""

import argparse
from math import sqrt

import tsv
from util import openfile, remove_pos

def normalize_vectors(corpus_mem):
    new_corpus = {}
    for target, vector in corpus_mem.iteritems():
        n = sqrt(sum(v ** 2 for v in vector.values()))
        new_corpus[target] = { k : v / n for k, v in vector.iteritems() }
    return new_corpus

def concat_spaces(corpora_mem):
    new_corpus = {}
    targets = reduce(set.union, (set(c.keys()) for c in corpora_mem))
    for target in targets:
        new_corpus[target] = {}
        for i, corpus in enumerate(corpora_mem):
            if target not in corpus:
                continue
            vector_piece = corpus[target]
            for k, v in vector_piece.iteritems():
                new_corpus[target]["%s_%d" % (k, i)] = v
    return new_corpus

def corpus_mem_to_tsv(corpus_mem):
    for k1, vec in corpus_mem.iteritems():
        for k2, val in vec.iteritems():
            yield dict(target=k1, context=k2, value=val)


def main():
    parser = argparse.ArgumentParser(
                description='Concatenates two or more vector spaces.')
    parser.add_argument("--input", "-i", action="append", type=openfile,
                        metavar="FILE", help='The input vector spaces.')
    parser.add_argument('--strip-pos', '-P', action='store_true',
                        help='Indicates we should strip POS tags from target dimensions when possible.')
    parser.add_argument('--norm', '-n', action='store_true',
                        help='Indicates we should normalize before concatenating')
    args = parser.parse_args()

    parsers={'value': float}
    if args.strip_pos:
        parsers['target'] = remove_pos
    corpuses = [tsv.read_tsv(i, ('target', 'context', 'value'), parsers=parsers) for i in args.input]
    corpuses_mem = [tsv_to_dict(c) for c in corpuses]

    if args.norm:
        corpuses_mem = [normalize_vectors(cm) for cm in corpuses_mem]

    new_space = concat_spaces(corpuses_mem)
    out_tsv = corpus_mem_to_tsv(new_space)

    tsv.print_tsv(out_tsv, headers=('target', 'context', 'value'), write_header=False)




if __name__ == '__main__':
    main()





