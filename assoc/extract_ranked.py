#!/usr/bin/env python

import numpy as np
import scipy.stats as stats
import sys

ranked_list = sys.argv[1]

vocab_file = "/scratch/01813/roller/corpora/webko/TermDoc/target-labels.txt"
assoc_file = "/home/01813/roller/tmp/imsgrounded/data/associations/big_assoc_vectors.txt"

assoc_counts = {}
resp_counts = {}
sims = {}

with open(vocab_file) as f:
    lines = f.readlines()
    lines = (l.strip() for l in lines)
    lines = (l.split("\t")[1] for l in lines if l)
    lines = list(lines)
    failing = [l for l in lines if '/' not in l]
    print failing
    lines = (l[:l.rindex("/")] for l in lines)
    vocab = set(lines)


num_ignored = 0
num_oov = 0
with open(assoc_file) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        cue, resp, count = line.split("\t")
        assert (cue, resp) not in assoc_counts
        if cue == resp:
            print "ignoring line b/c the response is the cue..."
            num_ignored += 1
            continue
        if resp not in vocab:
            print "ignoring line b/c response (%s) is OOV..." % line
            num_oov += 1
            continue
        assoc_counts[(cue, resp)] = int(count)
        resp_counts[resp] = resp_counts.get(resp, 0) + int(count)

num_foundhigher = 0
with open(ranked_list) as f:
    for i, line in enumerate(f, 1):
        if i % 100000 == 0: print "At doc #%d/31,927,108" % i
        line = line.strip()
        if not line: continue
        left, right, sim, rank, pct = line.split("\t")
        (sim, rank, pct) = map(float, (sim, rank, pct))
        left = left[:left.rindex('/')]
        right = right[:right.rindex('/')]
        if (left, right) in assoc_counts:
            #print "%s\t%d" % (line, assoc_counts[(left, right)])
            if sims.get((left, right), 0) < pct:
                if (left, right) in sims: num_foundhigher += 1
                sims[(left, right)] = pct

print "Uncounted pairs: %d" % (len(assoc_counts) - len(sims))
print "Counted pairs: %d" % len(sims)


hits = []
for pair, sim in sims.iteritems():
    hits += [sim] * assoc_counts[pair]

hits = np.array(hits)

print "num skipped: %d" % num_ignored
print "num oov: %d" % num_oov
print "num higher: %d" % num_foundhigher
print "Average sim of located associations: %f" % np.average(hits)
print "Std dev of located associations: %f" % np.std(hits)
print "Percentiles [.05, .10, .25, .5, .75, .90, .95] ="
print "     [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]" % tuple([stats.scoreatpercentile(hits, p) for p in [5, 10, 25, 50, 75, 90, 95]])
print "Average sim of all allocations: %f" % (sum(hits) / sum(assoc_counts.values()))






