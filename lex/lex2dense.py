#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
from util import *
from collections import Counter

def tryint(v):
    try:
        return int(v)
    except ValueError:
        return float(v)

MIN_COUNT = 2
ADD_MULTIPLE = True
IGNORE_NONNN = False
PRINT_IDS = True

f = openfile(sys.argv[1])
f.next()
data = {}

cols = Counter()
tars = set()
for line in f:
    if IGNORE_NONNN and '/NN' not in line:
        continue
    t, c, v = line.strip().split("\t")
    v = float(v)
    t, pos = extract_word_pos(t)
    if pos != 'NN':
        #continue
        pass
    if t not in data:
        data[t] = {}
    if c in data[t]:
        if ADD_MULTIPLE:
            data[t][c] += v
        else:
            raise ValueError, 'Already found context "%s" for target "%s".' % (c, t)
    else:
        data[t][c] = v
    cols.update([c])
    tars.add(t)

cols = sorted(w for w, v in cols.iteritems() if v >= MIN_COUNT)
tars = sorted(tars)

if not PRINT_IDS:
    for t in tars:
        as_str = " ".join(repr(tryint(data[t].get(c, 0))) for c in cols)
        print "%s\t%s" % (t, as_str)
else:
    for i, w in enumerate(cols, 1):
        print '%d\t%s' % (i, w)


