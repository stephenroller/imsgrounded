#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import os.path
from math import sqrt
from scipy.stats import spearmanr

storedir = "/Users/stephen/Working/imsgrounded/data"

vectors = pd.read_table(sys.argv[1], header=False, names=("Synset", "Vector"))


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





