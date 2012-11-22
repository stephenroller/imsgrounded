#!/usr/bin/env python

from random import choice, randint, sample, random

def random_judgement():
    return randint(1, 7)

def randomize_values(data, p, rfunc=random_judgement):
    assert 0 <= p <= 1
    data = data.copy()
    judgement_columns = data.columns[2:]
    mask = data[judgement_columns].applymap(lambda x: False)
    isstr = lambda x: isinstance(x, str)
    replf = lambda x: (isstr(x) or random() > p) and x or rfunc()
    return data.applymap(replf)

def blank_values(data, p):
    assert 0 <= p <= 1
    judgement_columns = data.columns[2:]
    isstr = lambda x: isinstance(x, str)
    r = lambda x: (isstr(x) or random() > p) and x or float('nan')
    retval = data.applymap(r)
    return retval


def add_subjects(data, n, rfunc=random_judgement):
    data = data.copy()
    for i in xrange(n):
        data['fakecol_%d' % i] = data.index.map(lambda x: rfunc())
    return data

def replace_subjects(data, n, rfunc=random_judgement):
    judgement_columns = data.columns[2:]
    assert n <= len(judgement_columns)
    removed_columns = sample(judgement_columns, n)
    data = data.copy()
    for rc in removed_columns:
        data[rc] = data[rc].map(lambda x: rfunc())
    return data

def blank_subjects(data, n):
    judgement_columns = data.columns[2:]
    removed_columns = sample(judgement_columns, n)
    data = data.copy()
    for rc in removed_columns:
        data[rc] = float('nan')
    return data



