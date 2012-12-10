#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
#import scipy as sp
from scipy.sparse.linalg import svds as svd
#from numpy.linalg import svd

def whiten_with_filled_zeros(data, k):
    judgement_columns = data.columns[2:]
    raw_data = data[judgement_columns]
    # need to interpolate missing judgements:
    # we'll fill them in w/ the avg of all other ratings
    matrix = raw_data.fillna(0).as_matrix()
    #matrix = raw_data.as_matrix()

    U, S, V = svd(matrix, k)
    # actual whitening
    S[k:] = 0
    # multiply stuff back out
    W = np.dot(U, np.dot(np.diag(S), V))

    # we should renull the values?

    whitened_data = data.copy()
    for i, j in enumerate(judgement_columns):
        whitened_data[j] = W[:,i]
        #nulls = data[j].isnull()
        #whitened_data[j][nulls] = float("nan")

    return whitened_data


def netflix_svd(data_only, max_k, epochs, learning_rate):
    # following http://sifter.org/~simon/journal/20070815.html
    R = data_only.as_matrix()
    NULLS = pd.isnull(data_only).as_matrix()
    m, n = R.shape

    # U, S, V = np.linalg.svd(np.random.rand(*data.shape), full_matrices=False)
    # we don't actually need S
    U = np.ones((m, max_k)) / 10.0
    V = np.ones((max_k, n)) / 10.0

    ck = 0
    while True:
        E = (R - np.dot(U, V))
        E[np.isnan(E)] = 0
        last_mag_E = (E ** 2).mean()
        for t in xrange(epochs):
            P = np.dot(U, V)
            E = R - P
            E[NULLS] = 0

            Udelta = np.dot(E, V[ck,:])
            Vdelta = np.dot(U[:,ck], E)

            U[:,ck] += learning_rate * Udelta
            V[ck,:] += learning_rate * Vdelta

            ud2 = np.linalg.norm(Udelta, 2)
            vd2 = np.linalg.norm(Vdelta, 2)
            E = (R - np.dot(U, V))
            E[np.isnan(E)] = 0
            mag_E = (E ** 2).mean()
            #if t % 1000 == 0:
            #    sys.stderr.write("K = %d, t = %d\t |E| = %.5f\t dE = %.5f\n" % (ck, t, mag_E, mag_E - last_mag_E))
            if abs(last_mag_E - mag_E) < 1e-5:
                break
            last_mag_E = mag_E

        ck += 1
        if ck >= max_k:
            break

    return U, V

def whiten(df, k, epochs=10000, learning_rate=.001):
    judgement_columns = df.columns[2:]
    data = df[judgement_columns]
    U, V = netflix_svd(data, k, epochs, learning_rate)

    W = np.dot(U[:,:k], V[:k,:])
    whitened_data = df.copy()
    for i, j in enumerate(judgement_columns):
        whitened_data[j] = W[:,i]

    return whitened_data



