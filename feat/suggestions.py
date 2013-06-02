#!/usr/bin/env python

import sys
import pandas as pd
from math import log



def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return float(current[n])


inpt_table = pd.read_table(sys.argv[1], names=("target", "feature", "cnt"))

features = inpt_table.groupby('feature').sum().rename(columns={'cnt': 'total_count'}).sort('total_count')
feature_counts = { fn: row['total_count'] for fn, row in features.iterrows() }
feature_names = feature_counts.keys()

sorted_inpt = inpt_table.join(features, on='feature').sort(columns=('total_count', 'target', 'feature'))

cache = {}


for i, row in sorted_inpt.iterrows():
    fn = row['feature']
    target = row['target']
    cnt = row['cnt']
    total_count = row['total_count']

    if fn in cache:
        weighted_distances = cache[fn]
    else:
        distances = { fn2: levenshtein(fn, fn2) for fn2 in feature_names if fn2 != fn }
        weighted_distances = { fn2 : log(feature_counts[fn2])/d for fn2, d in distances.iteritems() }

        cache[fn] = weighted_distances


    weighted = sorted(weighted_distances.iteritems(), reverse=True, key=lambda x: x[1])[:5]

    r = "    ".join("%s %.3f" % w for w in weighted)
    print "%s\t%s\t%d\t%d\t%s" % (target, fn, cnt, total_count, r)


