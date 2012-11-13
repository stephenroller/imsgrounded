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

def numeric_columns(dataframe):
    return [dataframe[c] for c in dataframe.columns if isinstance(dataframe[c][0], Number)]

def pairs(lst):
    for i, x in enumerate(lst):
        for y in lst[i+1:]:
            yield x, y

def correlations(dataframe):
    from scipy.stats import spearmanr
    output = []
    columns = list(numeric_columns(dataframe))
    for col1, col2 in pairs(columns):
        rho, p = spearmanr(col1, col2)
        output.append(dict(col1=col1.name, col2=col2.name, rho=rho, p=p))
    return pd.DataFrame(output, columns=("col1", "col2", "rho", "p"))

def scatters(dataframe, filename):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)
    plt.locator_params(tight=True)
    columns = list(numeric_columns(dataframe))
    for col1, col2 in pairs(columns):
        plt.plot(col1, col2, 'o')
        xspace = 0.05 * (col1.max() - col1.min())
        yspace = 0.05 * (col2.max() - col2.min())
        plt.axis([col1.min() - xspace, col1.max() + xspace, col2.min() - yspace, col2.max() + yspace])
        plt.xlabel(col1.name)
        plt.ylabel(col2.name)
        pp.savefig()
        plt.clf()
    pp.close()



def main():
    parser = argparse.ArgumentParser(
                description='Computes correlations with compositionality ratings.')
    parser.add_argument('--input', '-i', action="append", type=openfile,
                        metavar="FILE", help='Input vector space.')
    parser.add_argument('--ratings', '-r', metavar='COMPFILE', type=openfile,
                        help='The compositionality ratings file.')
    parser.add_argument('--self', '-s', action="store_true",
                        help='Whether we should include self-comp ratings.')
    parser.add_argument('--no-tsv', '-T', action="store_true",
                        help="*Don't* output the TSV containing comp and model ratings.")
    parser.add_argument('--corrs', '-c', action="store_true",
                        help='Specifies whether correlations should be computed and outputed.')
    parser.add_argument('--pdf', '-p', metavar="FILE", default=None,
                        help='Output plots as a PDF to the given filename.')

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

    # finally join the similarity measures with the human ratings
    dm_and_comp = pd.merge(compratings, joined_measures)

    # output dm_and_comp unless the user specified not to
    if not args.no_tsv:
        dm_and_comp.to_csv(sys.stdout, index=False, sep="\t")

    # nicer output
    if not args.no_tsv and args.corrs:
        # let's compute our correlations
        print "\n" + "-" * 80 + "\n"

    # compute and output correlations if the user asked
    if args.corrs:
        corrs = correlations(dm_and_comp).to_csv(sys.stdout, index=False, sep="\t")

    # plot the measures if the user asked.
    if args.pdf:
        scatters(dm_and_comp, args.pdf)




if __name__ == '__main__':
    main()
