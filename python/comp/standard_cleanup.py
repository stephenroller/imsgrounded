#!/usr/bin/env python

import sys
import pandas as pd
import argparse
import math
from os.path import basename
from scipy.stats import spearmanr

DEFAULT_MIN_CORR = 0.5
DEFAULT_ZSCORE = 1.0

def na_spearmanr(a, b):
    mask = a.notnull().values & b.notnull().values
    rho, p = spearmanr(a[mask], b[mask])
    if not isinstance(rho, float) or math.isnan(rho):
        return -1.0, 1.0
    else:
        return rho, p

deviant_subject_cache = {}
def remove_most_deviant_subjects(data, n):
    if id(data) in deviant_subject_cache:
        by_rho = deviant_subject_cache[id(data)]
    else:
        judgement_columns = data.columns[2:]
        agreements = {}
        for j in judgement_columns:
            ratings = data[j]
            other_subjects = list(set(judgement_columns) - set([j]))
            exclusive_means = data[other_subjects].transpose().mean()
            rho, p = na_spearmanr(ratings, exclusive_means)
            # sys.stderr.write("%s with others: %f\n" % (j, rho))
            agreements[j] = rho

        by_rho = sorted(agreements.keys(), key=agreements.__getitem__)
        deviant_subject_cache[id(data)] = by_rho

    out_data = data.copy()
    for j in by_rho[:n]:
        out_data[j] = float('nan')

    return out_data

def remove_percent_deviant_subjects(data, p):
    judgement_columns = data.columns[2:]
    n = int(math.ceil(p * len(judgement_columns)))
    return remove_most_deviant_subjects(data, n)



def remove_deviant_subjects(data, min_corr=DEFAULT_MIN_CORR):
    # TODO: don't hardcode this
    judgement_columns = data.columns[2:]

    to_remove = []
    for j in judgement_columns:
        ratings = data[j]
        other_subjects = list(set(judgement_columns) - set([j]))
        exclusive_means = data[other_subjects].transpose().mean()
        rho, p = na_spearmanr(ratings, exclusive_means)
        # sys.stderr.write("%s with others: %f\n" % (j, rho))
        if rho < min_corr:
            # sys.stderr.write("Removing %s\n" % j)
            to_remove.append(j)

    out_data = data.copy()
    for j in to_remove:
        out_data[j] = float('nan')

    return out_data

deviant_ratings_cache = {}
def remove_deviant_ratings(data, outlier_zscore=DEFAULT_ZSCORE):
    judgement_columns = data.columns[2:]
    if id(data) in deviant_ratings_cache:
        zscores = deviant_ratings_cache[id(data)]
    else:
        rows = []
        for i, row in data.iterrows():
            new_row = row.copy()
            mean = new_row[judgement_columns].transpose().mean()
            stddev = new_row[judgement_columns].transpose().std()
            for j in judgement_columns:
                zscore = (row[j] - mean) / stddev
                new_row[j] = abs(zscore)
            rows.append(new_row)
        zscores = pd.DataFrame(rows)[judgement_columns]
        deviant_ratings_cache[id(data)] = zscores

    data = data.copy()
    data[judgement_columns] = data[judgement_columns].where(zscores <= outlier_zscore)
    return data



def aggregate_ratings(data):
    output_data = []
    judgement_columns = data.columns[2:]
    for i, row in data.iterrows():
        ratings = row[judgement_columns].transpose()
        nonnull = sum(ratings.notnull())
        if nonnull == 0:
            # skip data points with only 0 or 1 ratings.
            continue
        mean = ratings.mean()
        stddev = nonnull > 1 and ratings.std() or 0
        median = nonnull > 1 and ratings.median() or ratings[0]
        output_data.append({
            'compound': row['compound'],
            'const': row['const'],
            'mean': mean,
            'median': median,
            'stddev': stddev,
            'var': stddev * stddev
        })
    return pd.DataFrame(output_data)


def main():
    parser = argparse.ArgumentParser(
                description='Filters out abnormal ratings from comp ratings.')
    parser.add_argument('--input', '-i', metavar="FILE", action="append", default=[],
                        help="Input comp ratings.")
    parser.add_argument('--filter-subjects', '-r', metavar="CORR", type=float,
                        help=("Filters subjects who do not correlate "
                              "with the others by at least CORR."))
    parser.add_argument('--filter-deviations', '-z', metavar="SIGMAs", type=float,
                        help=("Filters individual judgements that deviate "
                              "from the average by SIGMA std devs."))
    parser.add_argument('--output-aggregate', '-a', action='store_true',
                        help=("Output aggregate statistics only. (Default: "
                              "outputs new judgements with blanks)."))
    parser.add_argument('--compare-whole', '-w', metavar='FILE',
                        help='Compute correlations with the whole compound ratings.')

    args = parser.parse_args()
    input = pd.concat(map(pd.read_csv, args.input))
    output = input
    if args.filter_subjects:
        output = remove_deviant_subjects(output, args.filter_subjects)
    if args.filter_deviations:
        output = remove_deviant_ratings(output, args.filter_deviations)
    if args.output_aggregate:
        aggregate_ratings(output).to_csv(sys.stdout, index=False)
    else:
        output.to_csv(sys.stdout, index=False)

    if args.compare_whole:
        agg = aggregate_ratings(output)
        whole_compounds = pd.read_csv(args.compare_whole)
        # we only want the intersection of the columns
        keepers = set(output.compound).intersection(set(whole_compounds.compound))
        agg = agg[agg.compound.map(keepers.__contains__)]
        whole_compounds = whole_compounds[whole_compounds.compound.map(keepers.__contains__)]
        whole_compounds = whole_compounds.sort('compound')

        by_sum = agg.groupby('compound').sum()
        by_sum['compound'] = by_sum.index
        by_sum = by_sum.sort('compound')
        rho_sum, p_sum = na_spearmanr(by_sum['mean'], whole_compounds['mean'])

        by_prod = agg.groupby('compound').prod()
        by_prod['compound'] = by_prod.index
        by_prod = by_prod.sort('compound')
        rho_prod, p_prod = na_spearmanr(by_prod['mean'], whole_compounds['mean'])

        sys.stderr.write("rho w/ whole compounds by sum: %f\n" % rho_sum)
        sys.stderr.write("rho w/ whole compounds by prod: %f\n" % rho_prod)


if __name__ == '__main__':
    main()

