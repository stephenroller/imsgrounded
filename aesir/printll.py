#!/usr/bin/env python

import sys
import numpy as np

HUMAN=True

for f in sys.argv[1:]:
    try:
        m = np.load(f)
    except:
        print "%s didn't work, skipping" % f
        continue

    if HUMAN:
        i = len(m["loglikelihoods"])
        print "%s [%d]: %f" % (f, i, m["loglikelihoods"][-1])
    else:
        ll = data["loglikelihoods"]
        k = data["k"]
        for i, l in enumerate(ll[1:], 2):
            print "%d,%d,%f" % (i, k, l)

