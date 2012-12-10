#!/usr/bin/env python

import sys
import pandas as pd
from itertools import islice
from random import choice, randint

# ratings data is a DataFrame with the following columns:
# compound, const, user1, user2, user3, ...
# and the user* columns report numeric assignments of ratings
# given by said user.

# these yield tuples (cost1, const2, const1score, const2score)

def gen_matches_randInf(ratings_data, trials=5000000):
    L = len(ratings_data)
    judgement_columns = ratings_data.columns[2:]
    for t in xrange(trials):
        j = choice(judgement_columns)
        user_ratings = ratings_data[j]
        i = choice(user_ratings.index)
        k = choice(user_ratings.index)
        nn = user_ratings.notnull()
        if not nn[i] or not nn[k]:
            continue
        a = user_ratings[i]
        b = user_ratings[k]
        # yield (i, k, a - b, b - a)
        if a == b:
            yield (i, k, 0.5, 0.5)
        elif a < b:
            yield (i, k, 0.0, 1.0)
        elif a > b:
            yield (i, k, 1.0, 0.0)


def compute_elo(matches, kfactor, start, spread):
    elos = {}
    i = 0
    for a, b, s_a, s_b in matches:
        i += 1
        if a not in elos:
            elos[a] = start
        if b not in elos:
            elos[b] = start

        # quotient factors
        q_a = 10 ** (elos[a] / spread)
        q_b = 10 ** (elos[b] / spread)

        # compute expected scores
        e_a = q_a / (q_a + q_b)
        e_b = q_b / (q_a + q_b)

        # update elos
        elos[a] = elos[a] + kfactor * (s_a - e_a)
        elos[b] = elos[b] + kfactor * (s_b - e_b)
        if i % 10000 == 0:
            sys.stderr.write("elo loop / t: %d, a: %f, b: %f\n" % (i, elos[a], elos[b]))

    return elos

def elos_to_df(elo_dict, orig_df):
    results = []
    for item, elo in elo_dict.iteritems():
        results.append({
            'compound': orig_df['compound'][item],
            'const': orig_df['const'][item],
            'elo': elo
        })
    return pd.DataFrame(results, columns=('compound', 'const', 'elo'))


def elo(df, kfactor=0.03, start=1500.0, spread=400.0):
    return elos_to_df(compute_elo(gen_matches_randInf(df), kfactor, start, spread), df)

if __name__ == '__main__':
    HEAD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_head.csv'
    MOD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_const.csv'

    df = pd.concat([pd.read_csv(HEAD_FILE), pd.read_csv(MOD_FILE)], ignore_index=True)
    elo(df).to_csv(sys.stdout, index=False)

