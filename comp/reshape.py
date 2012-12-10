#!/usr/bin/env python

import sys
import pandas as pd
from pandas.core.reshape import melt

df = pd.read_csv(sys.argv[1])

meas = [
    'orig_clean',
    'clean_orig',
    'orig_noisy',
    'noisy_orig',
    'clean_noisy',
    'noisy_clean',
    'clean_clean',
    'noisy_noisy',
    'orig_orig',
    'cleanorig_orig',
    'cleanorig_noisy',
    'cleanorig_clean',
    'orig_cleanorig',
    'noisy_cleanorig',
    'clean_cleanorig',
]

# meas += [
#     'whole_orig_notblank',
#     'indiv_orig_notblank',
#     'whole_clean_notblank',
#     'indiv_clean_notblank',
#     'whole_noisy_notblank',
#     'indiv_noisy_notblank',
# ]

all_columns = set(df.columns)
meas = all_columns.intersection(set(meas))
accounted_for = set(["%s%s" % (c, s) for c in meas for s in ["", "_low", "_high", "_std"]])
parameters = all_columns - accounted_for

output = []

for i, row in df.iterrows():
    for m in meas:
        out = {k : row[k] for k in parameters}
        out['variable'] = m
        out['mean'] = row[m]
        out['std']  = row[m + "_std"]
        out['high'] = row[m + "_high"]
        out['low']  = row[m + "_low"]
        output.append(out)




pd.DataFrame(output).to_csv(sys.stdout, index=False)

