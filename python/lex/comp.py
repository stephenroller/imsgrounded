#!/usr/bin/env python

"""
Single script for computing spearman correlation between different models
and the compositionality ratings.
"""

import sys
import argparse

from os.path import basename
from numbers import Number

import pandas as pd

from util import openfile, df_remove_pos, read_vector_file
from distances import cosine, calculate_distance_metrics as cdm
from matrix import norm2_matrix

DISTANCE_METRIC = cosine

def correlations(dataframe):
    from scipy.stats import spearmanr
    output = []
    columns = list(dataframe.columns)
    for i, colname1 in enumerate(columns):
        col1 = dataframe[colname1]
        if not isinstance(col1[0], Number):
            continue
        for colname2 in columns[i+1:]:
            col2 = dataframe[colname2]
            if not isinstance(col2[0], Number):
                continue
            # okay, we want correlation here
            rho, p = spearmanr(col1, col2)
            output.append(dict(col1=colname1, col2=colname2, rho=rho, p=p))
    return pd.DataFrame(output, columns=("col1", "col2", "rho", "p"))



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

    compratings = pd.read_table(args.ratings)
    if not args.self:
        compratings = compratings[compratings["compound"] != compratings["const"]]

    word_pairs = set(zip(compratings['compound'], compratings['const']))

    named_vector_spaces = [
        (basename(f.name), norm2_matrix(df_remove_pos(read_vector_file(f))))
        for f in args.input
    ]

    if len(named_vector_spaces) > 1:
        # need to do concatenation
        names, vses = zip(*named_vector_spaces)
        concat_space = pd.concat(vses, keys=names)
        named_vector_spaces.append(("<concat>", concat_space))

    # compute all the distances AND keep the different measures independently named
    distances = [
        cdm(vs, word_pairs, [DISTANCE_METRIC])
          .rename(columns={DISTANCE_METRIC.name: fn + ":" + DISTANCE_METRIC.name})
        for fn, vs in named_vector_spaces
    ]
    # now we need to join all the distance calculations:
    joined_measures = reduce(pd.merge, distances).rename(
                        columns={"left": "compound", "right": "const"})

    dm_and_comp = pd.merge(compratings, joined_measures)

    dm_and_comp.to_csv(sys.stdout, index=False, sep="\t")

    # let's compute our correlations
    print "\n" + "-" * 80 + "\n"

    corrs = correlations(dm_and_comp).to_csv(sys.stdout, index=False, sep="\t")




if __name__ == '__main__':
    main()
