#!/usr/bin/env python

"""
Single script for computing spearman correlation between different models
and the compositionality ratings.
"""

import argparse
from os.path import basename
from itertools import izip
from collections import OrderedDict

from util import openfile, remove_pos, tsv_to_dict
from tsv import read_tsv, print_tsv
from distances import cosine, calculate_distance_metrics
from join import union_records, join_on
from concat import concat_spaces, normalize_vectors

DISTANCE_METRIC = cosine

def join_distance_records(*distance_measures):
    for distance_rows in izip(*distance_measures):
        lefts = [z[0] for z in distance_rows]
        rights = [z[1] for z in distance_rows]
        values = [z[2] for z in distance_rows]
        assert (len(set(lefts)) == 1 and len(set(rights)) == 1), (
                "Some sort of strange non-matchup. Crashing hard.")
        new_record = union_records(lefts[:1], rights[:1], values)
        yield new_record

def main():
    parser = argparse.ArgumentParser(
                description='Computes correlations with compositionality ratings.')
    parser.add_argument('--input', '-i', action="append", type=openfile,
                            metavar="FILE", help='Input vector space.')
    parser.add_argument('--ratings', '-r', metavar='COMPFILE', type=openfile,
                            help='The compositionality ratings file.')
    parser.add_argument('--self', '-s', action="store_true",
                            help='Whether we should include self-comp ratings.')
    args = parser.parse_args()

    compratings = read_tsv(args.ratings, headers=True)
    if not args.self:
        compratings = [cr for cr in compratings if cr['compound'] != cr['const']]

    word_pairs = set((cr['compound'], cr['const']) for cr in compratings)

    vector_spaces_tsvs = (
        (basename(f.name),
            read_tsv(f, headers=('target', 'context', 'value'), 
                        parsers={'value': float}))
        for f in args.input
    )
    vector_spaces_mem = [
        (fn, normalize_vectors(tsv_to_dict(c, keep_pos=False)))
        for fn, c in vector_spaces_tsvs
    ]
    if len(vector_spaces_mem) > 1:
        # need to do concatenation
        concat_space = concat_spaces([vsm for fn, vsm in vector_spaces_mem])
        vector_spaces_mem.append(("<concat>", concat_space))

    distances_all = [
        (fn, calculate_distance_metrics(vs, word_pairs, [DISTANCE_METRIC]))
        for fn, vs in vector_spaces_mem
    ]
    # now we need to join all the distance calculations:
    filenames = [fn for fn, ds in distances_all]
    distance_measures = [ds for fn, ds in distances_all]
    keys = ['left', 'right'] + [DISTANCE_METRIC.name + ":" + fn for fn in filenames]
    joined_distance_measures = (OrderedDict(zip(keys, r)) for r in join_distance_records(*distance_measures))

    # okay, so we've got all our distance metrics. now to join with the comp ratings.
    join_keys = set([('left', 'compound'), ('right', 'const')])
    dm_and_comp = join_on(join_keys, joined_distance_measures, compratings)

    print_tsv(dm_and_comp, headers=True)




if __name__ == '__main__':
    main()
