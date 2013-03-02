#!/usr/bin/env python

import sys
import pandas as pd
import operator
import numpy as np
import os.path
from os.path import basename
from math import sqrt
from scipy.stats import spearmanr
from itertools import combinations as icombinations
from itertools import chain
from functools import partial
from math import sqrt
import bz2

from apgl.features.KernelCCA import KernelCCA
from apgl.kernel.LinearKernel import LinearKernel


STOREDIR = "/Users/stephen/Working/imsgrounded/data"
comp_values = pd.read_table(os.path.join(STOREDIR, "comp/comp-values_all_sorted.tsv"))
comp_values = comp_values[comp_values.const != comp_values.compound]

def parse_vector(v):
    return np.array(map(float, v.split(" ")))

def read_space(filename):
    if filename.endswith(".bz2"):
        f = bz2.BZ2File(filename)
    else:
        f = open(filename)
    data = {}
    for line in f:
        line = line.strip()
        word, vector = line.split("\t")
        vector = parse_vector(vector)
        data[word] = vector
    return data

def combinations(lst):
    return chain(*(icombinations(lst, i) for i in xrange(1, len(lst)+1)))

def norm2(vector):
    return vector / sqrt(vector.dot(vector))

def join_vectors(vectors, norm=True):
    if norm:
        vectors = (norm2(v) for v in vectors)
    vectors = (list(v) for v in vectors)
    return np.array(reduce(operator.add, vectors))

def cosine(v1, v2):
    return v1.dot(v2) / sqrt(v1.dot(v1) * v2.dot(v2))

def lmi(space):
    M = np.matrix(list(space.vector))
    tt = M.sum()
    py = sum(M) / tt
    px = sum(M.T).T / tt
    pxy = M / float(tt)
    pxpy = np.dot(px, py)
    pmi = np.log2(pxy) - np.log2(pxpy)
    lmi = np.multiply(pxy, pmi)
    nnlmi = lmi.copy()
    nnlmi[nnlmi < 0] = 0
    nnlmi[np.isnan(nnlmi)] = 0
    vectors_lmied = pd.DataFrame({'word': space.word, 'vector': map(np.array, nnlmi.tolist())})
    return vectors_lmied

def perform_svd(space):
    M = np.matrix(list(space.vector))
    U, S, V = np.linalg.svd(M, full_matrices=False)
    return U, S, V

def reduce_space(space, U, S, V, min_sigma=None, max_dim=None, latent_only=False):
    S = S.copy()
    if min_sigma:
        S[s < min_sigma] = 0
    if max_dim:
        S[max_dim:] = 0
    if latent_only:
        Ma = np.dot(U, np.diag(S))
    else:
        Ma = np.dot(U, np.dot(np.diag(S), V))
    vectors_svded = pd.DataFrame({'word': space.word, 'vector': map(np.array, Ma.tolist())})
    return vectors_svded


def eval_space(vectors, comps):
    joined = comps.merge(vectors, left_on='compound', right_on='word').merge(vectors, left_on='const', right_on='word')
    joined['cos'] = [cosine(v1, v2) for v1, v2 in zip(joined['vector_y'], joined['vector_x'])]
    rho, p = spearmanr(joined['mean'], joined['cos'])
    return rho, p, len(joined['const'])


def all_params(vectors):
    rho, p, n = eval_space(vectors, comp_values)
    print "  Basic Concat (With all pairs):"
    print "    Number pairs compared:", n
    print "    rho =", rho, "   p =", p

    return

    U, S, V = perform_svd(vectors)
    for k in [1, 2, 5, 10, 20, 50, 100, 200]:
        for latent in [True, False]:
            rho, p, n = eval_space(reduce_space(vectors, U, S, V, max_dim=k, latent_only=latent), comp_values)
            print "  SVD Space (k = %d, latent = %s):" % (k, latent)
            print "    Number of pairs compared:", n
            print "    rho =", rho, "   p =", p


spaces = {basename(sp) : read_space(sp) for sp in sys.argv[1:]}
all_keepwords = reduce(set.intersection, (set(spaces[c].keys()) for c in spaces.keys()))

for combination in combinations(spaces.keys()):
    print "+".join(list(combination)) + ":"
    keepwords = sorted(reduce(set.intersection, (set(spaces[c].keys()) for c in combination)))
    vectors = pd.DataFrame([{'word': w, 'vector':  join_vectors([spaces[c][w] for c in combination]) } for w in keepwords])
    if keepwords != all_keepwords:
        print "(Only comparable pairs)"
        mask = vectors.word.map(lambda x: x in all_keepwords)
        all_params(vectors[mask])
    else:
        print "(All Pairs)"
        all_params(vectors)

    continue
    lmi_vectors = [lmi(pd.DataFrame([{'word': w, 'vector':  spaces[c][w]}  for w in keepwords]))
                    for c in combination]
    zipped_vectors = zip(*[sp.vector for sp in lmi_vectors])
    joined_lmi = pd.DataFrame([{'word': w, 'vector': join_vectors(zv)} for w, zv in zip(keepwords, zipped_vectors)])
    print "LMI:"
    all_params(joined_lmi)

    if False and len(combination) == 2:
        # sweet, we can do CCA:
        sp1name, sp2name = combination
        X = np.array([spaces[sp1name][w] for w in keepwords])
        Y = np.array([spaces[sp2name][w] for w in keepwords])
        tau = 0.0
        kernel = LinearKernel()
        cca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lambdas = cca.learnModel(X, Y)
        Xp, Yp = cca.project(X, Y)
        cca_joined = pd.DataFrame([{'word': w, 'vector': join_vectors((v1, v2), norm=False)}
                                    for w, v1, v2 in zip(keepwords, Xp, Yp)])
        print alpha
        print
        print beta
        print
        print lambdas
        print
        print "CCA Analysis:", lambdas
        try:
            all_params(cca_joined)
        except:
            pass






    print


