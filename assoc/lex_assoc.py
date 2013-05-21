#!/usr/bin/env python

import sys
import numpy as np
import scipy.stats
import logging
import codecs
from numpy import float64
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
from sklearn.preprocessing import normalize

logging.basicConfig(
    format="[ %(levelname)-10s %(module)-8s %(asctime)s  %(relativeCreated)-10d ]  %(message)s",
    datefmt="%H:%M:%S:%m",
    level=logging.DEBUG)

vocab_file = "/scratch/01813/roller/corpora/webko/TermDoc/target-labels.txt"
assoc_file = "/home/01813/roller/tmp/imsgrounded/data/associations/big_assoc_vectors.txt"
nn_file = "/home/01813/roller/tmp/imsgrounded/data/nn.txt"

assoc_counts = {}
resp_counts = {}
dots_to_compute = set()

nns = set([l.strip() for l in codecs.getreader('utf-8')(open(nn_file)).readlines()])

logging.info("reading vocab")
with codecs.getreader('utf-8')(open(vocab_file)) as f:
    lines = f.readlines()
    lines = (l.strip() for l in lines)
    lines = (l.split("\t")[1] for l in lines if l)
    lines = list(lines)
    lines = (l[:l.rindex("/")] for l in lines)
    vocab = set(lines)


logging.info("reading assoc")
num_ignored = 0
num_oov = 0
with codecs.getreader('utf-8')(open(assoc_file)) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        cue, resp, count = line.split("\t")
        assert (cue, resp) not in assoc_counts
        if cue not in nns:
            #logging.info("ignoring b/c not in list of NN compounds.")
            num_ignored += 1
            continue
        if cue == resp:
            logging.info( "ignoring line b/c the response is the cue...")
            num_ignored += 1
            continue
        if resp not in vocab:
            #logging.info( "ignoring line b/c response (%s) is OOV..." % line)
            num_oov += 1
            continue
        dots_to_compute.add(cue)
        assoc_counts[(cue, resp)] = int(count)
        resp_counts[resp] = resp_counts.get(resp, 0) + int(count)

logging.info("reading vs (pass 1)")
# get ids for dimensions and shape of matrix
targets = dict()
dims = dict()
R = 75678
C = 1038883
data, rows, cols = [], [], []
from collections import defaultdict
fast_target_lookup = defaultdict(list)
with codecs.getreader('utf-8')(sys.stdin) as vs:
#with codecs.getreader('utf-8')(open(sys.argv[1])) as vs:
    for line in vs:
        line = line.strip()
        dim, target, weight = line.split("\t")
        weight = float(weight)
        if target not in targets:
            fast_target_lookup[target[:target.rindex('/')]].append(len(targets))
            targets[target] = len(targets)
        t = targets[target]
        if dim not in dims:
            dims[dim] = len(dims)
        d = dims[dim]

        data.append(weight)
        rows.append(t)
        cols.append(d)

coo = coo_matrix((data, (rows, cols)))
del data, rows, cols
sp_vectors = coo.tocsr()
del coo

# now we need to get out only the vectors for which we need ranked cosines:
logging.info("Looking up vectors we need ranks for")
row_ids = []
row_names = []
baddies = []
for word in dots_to_compute:
    if word + '/NN' in targets:
        row_ids.append(targets[word + '/NN'])
        row_names.append(word)
    else:
        baddies.append(word)

from itertools import count, izip
row_lookup = dict(izip(row_names, count()))

logging.warning("bads: %d  / goods: %d" % (len(baddies), len(row_ids)))
logging.warning("baddies: %s" % baddies)

logging.info("extracting just nn vectors.")
nn_vectors = sp_vectors[row_ids]

# pairwise cosine
from sklearn.metrics.pairwise import cosine_similarity

logging.info("computing pairwise cosines")
coses = cosine_similarity(nn_vectors, sp_vectors)
del sp_vectors

def percentile_ranked(similarities):
    return np.ceil(scipy.stats.rankdata(similarities)) / len(similarities)

logging.info("ranking...")
ranked_coses = np.array(map(percentile_ranked, coses))

# okay, got all our cosines. let's add up the averages with assocs
logging.info("Now computing ranked averages.")
hits = []

num_foundhigher = 0
for (cue, resp), cnt in assoc_counts.iteritems():
    if cue not in row_lookup:
        continue
    rightids = fast_target_lookup[resp]
    if not rightids:
        continue
    leftid = row_lookup[cue]
    sims = ranked_coses[leftid,:]
    options = sims[rightids]
    num_foundhigher += len(options) - 1
    hits += [max(options)] * cnt

print "num skipped: %d" % num_ignored
print "num oov: %d" % num_oov
print "num higher: %d" % num_foundhigher
print "Average sim of located associations: %f" % np.average(hits)
print "Std dev of located associations: %f" % np.std(hits)
print "Percentiles [.05, .10, .25, .5, .75, .90, .95] ="
print "     [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]" % tuple([scipy.stats.scoreatpercentile(hits, p) for p in [5, 10, 25, 50, 75, 90, 95]])
mean_ctr95, var_ctr, std_ctr = scipy.stats.bayes_mvs(hits, alpha=0.95)
mean_ctr50, var_ctr, std_ctr = scipy.stats.bayes_mvs(hits, alpha=0.50)
(m, (lb, ub)) = mean_ctr95
(m, (mlb, mub)) = mean_ctr50
print "bayes_mvs = [%.8f  %.8f  %.8f  %.8f  %.8f]" % (lb, mlb, m, mub, ub)
#print "Average sim of all allocations: %f" % (sum(hits) / sum(assoc_counts.values()))






