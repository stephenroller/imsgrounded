#!/usr/bin/env python

import sys
import numpy as np
import os.path
from itertools import izip

HUMAN=False

if not HUMAN:
    print "model,type,iteration,k,time,eval,mu,eta,alpha"

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

    nmn=os.path.dirname(f)

    if HUMAN:
        i = len(m[key])
        print "%s [%d]: %f" % (f, i, m[key][-1])
    else:
        ll = m[key]
        k = m["k"]
        for i, (l, t) in enumerate(izip(ll, timediffs), 1):
            print "%s,%s,%d,%d,%f,%f,%f,%f,%f" % (f, nmn, i, k, t, l, m["mu"], m["eta"], m["alpha"])

