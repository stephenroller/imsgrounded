#!/usr/bin/env python

import sys
import numpy as np
from itertools import izip

HUMAN=False

if not HUMAN:
    print "iteration,k,time,eval"

for f in sys.argv[1:]:
    try:
        m = np.load(f)
    except:
        print "%s didn't work, skipping" % f
        continue

    if "loglikelihoods" in m:
        key = "loglikelihoods"
    else:
        key = "perwordbounds"

    timediffs = np.cumsum(m['timediffs'])


    if HUMAN:
        i = len(m[key])
        print "%s [%d]: %f" % (f, i, m[key][-1])
    else:
        ll = m[key]
        k = m["k"]
        for i, (l, t) in enumerate(izip(ll, timediffs), 1):
            print "%d,%d,%f,%f" % (i, k, t, l)

