#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
from util import *
from collections import Counter

MIN_COUNT = 0

f = openfile(sys.argv[1])
f.next()
data = {}

cols = Counter()
tars = set()
for line in f:
    t, c, v = line.strip().split("\t")
    t, pos = extract_word_pos(t)
    if pos != 'NN':
        #continue
        pass
    if t not in data:
        data[t] = {}
    assert c not in data[t], "already found context %s for target %s" % (c, t)
    data[t][c] = v
    cols.update([c])
    tars.add(t)

cols = sorted(w for w, v in cols.iteritems() if v >= MIN_COUNT)
tars = sorted(tars)

for t in tars:
    as_str = " ".join(data[t].get(c, "0") for c in cols)
    print "%s\t%s" % (t, as_str)




