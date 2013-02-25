#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import os.path
import bz2
from math import sqrt
from scipy.stats import spearmanr

storedir = "/Users/stephen/Working/imsgrounded/data"

#vectors = pd.read_table(sys.argv[1], header=False, names=("Synset", "Vector"))
if sys.argv[1].endswith(".bz2"):
    f = bz2.BZ2File(sys.argv[1])
else:
    f = open(sys.argv[1])

lines = [line.split("\t") for line in f.read().split("\n") if line.strip()]
vectors = pd.DataFrame([{'Synset': a, 'Vector': b} for a, b in lines])


mappings_raw = pd.read_table(os.path.join(storedir, "images/imagenet_mappings.tsv"))
nonempty = mappings_raw[mappings_raw.Synset.notnull()]
splitted = nonempty.Synset.map(lambda x: x.split())

aftersplit = [{'Compound': cmp, 'OrigSynset': ss} for cmp, sss in zip(nonempty.Compound, splitted) for ss in sss]
remapped = pd.DataFrame(aftersplit)

extender = pd.read_table(os.path.join(storedir, "images/imagenet_hypos.txt"), header=False, names=("OrigSynset", "Synset"))
remapped = remapped.merge(extender)

def norm1(v):
    return v
    return v / float(sum(v))

try:
    parsed = vectors.Vector.map(lambda x: norm1(np.array(map(int, x.split(" ")))))
except:
    parsed = vectors.Vector.map(lambda x: norm1(np.array(map(float, x.split(" ")))))

del vectors["Vector"]
vectors["Vector"] = parsed

joined = remapped.merge(vectors)

def aggfunc(x):
    return {"Vector": x.Vector.sum()}

final_vectors = joined.groupby("Compound").aggregate(aggfunc).Vector
#final_vectors = pd.DataFrame({"word": final_vectors.index, "vectors": final_vectors})

for compound, vector in final_vectors.iteritems():
    print "%s\t%s" % (compound, " ".join(map(str, vector)))





