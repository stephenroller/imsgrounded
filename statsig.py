#!/usr/bin/env python

"""
A module for doing approximate randomization statistical significance
testing for NLP applications.

An implementation of p.13 of http://masanjin.net/sigtest.pdf.
"""

# default number of trials to use.
NUM_TRIALS = 1000

import random

def siglevel(output1, output2, evaluator, trials=NUM_TRIALS):
    assert len(output1) == len(output2)

    score1 = evaluator(output1)
    score2 = evaluator(output2)
    t_0 = abs(score1 - score2)

    X = list(output1)
    Y = list(output2)

    r = 0
    for n in xrange(trials):
        for i in xrange(len(X)):
            if random.random() < 0.5:
                X[i], Y[i] = Y[i], X[i]
        t_n = abs(evaluator(X) - evaluator(Y))
        if t_n >= t_0:
            r += 1
    return float(r + 1) / (trials + 1)


def _test_sig():
    out1 = [random.randint(0, 19) > 0 for i in xrange(1000)]
    out2 = [random.randint(0, 4) > 0 for i in xrange(1000)]
    return siglevel(out1, out2, sum)

def _test_nonsig():
    out1 = [random.randint(0, 9) > 0 for i in xrange(1000)]
    out2 = [random.randint(0, 9) > 0 for i in xrange(1000)]
    return siglevel(out1, out2, sum)


if __name__ == '__main__':
    print _test_sig()
    print _test_nonsig()


