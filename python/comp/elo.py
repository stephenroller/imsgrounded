#!/usr/bin/env python

import sys
import pandas as pd
from itertools import islice

# ratings data is a DataFrame with the following columns:
# compound, const, user1, user2, user3, ...
# and the user* columns report numeric assignments of ratings
# given by said user.

# these yield tuples (cost1, const2, const1score, const2score)

def gen_matches_randInf(ratings_data):
    L = len(ratings_data)
    from random import choice, randint
    judgement_columns = ratings_data.columns[2:]
    while True:
        j = choice(judgement_columns)
        i = randint(0, L - 1)
        k = randint(0, L - 1)
        user_ratings = ratings_data[j]
        nn = user_ratings.notnull()
        if not nn[i] or not nn[k]:
            continue
        a = user_ratings[i]
        b = user_ratings[k]
        if a == b:
            yield (i, k, 0.5, 0.5)
        elif a < b:
            yield (i, k, 0.0, 1.0)
        elif a > b:
            yield (i, k, 1.0, 0.0)
        else:
            assert False, "wtf?"


def compute_elo(matches, kfactor=10.0, start=1500.0, spread=400.0):
    elos = {}
    for a, b, s_a, s_b in matches:
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

    return elos

HEAD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_head.csv'
MOD_FILE = '/Users/stephen/Working/imsgrounded/data/comp/comp_ratings_const.csv'

df = pd.read_csv(HEAD_FILE)
#elos = compute_elo(gen_matches_seq(df))
elos = compute_elo(islice(gen_matches_randInf(df), 5000000))

results = []
for item, elo in elos.iteritems():
    results.append({
        'compound': df['compound'][item],
        'const': df['const'][item],
        'elo': elo
    })

pd.DataFrame(results).to_csv(sys.stdout, index=False)

