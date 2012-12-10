#!/usr/bin/env python
import math
from random import choice, randint, sample, random

def random_judgement(old_judgement):
    return randint(1, 7)

def random_judgement_offset(old_judgement, d=3):
    if random() <= 0.5:
        d = -d
    return (old_judgement + d - 1) % 7 + 1

def randomize_values(data, p, rfunc=random_judgement):
    assert 0 <= p <= 1
    data = data.copy()
    judgement_columns = data.columns[2:]
    isstr = lambda x: isinstance(x, str)
    replf = lambda x: (isstr(x) or random() > p) and x or rfunc(x)
    return data.applymap(replf)

def randomize_offsets(data, p):
    return randomize_values(data, p, random_judgement_offset)

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
        data['fakecol_%d' % i] = data.index.map(lambda x: rfunc(x))
    return data

def replace_subjects(data, n, rfunc=random_judgement):
    judgement_columns = data.columns[2:]
    assert n <= len(judgement_columns)
    removed_columns = sample(judgement_columns, n)
    data = data.copy()
    for rc in removed_columns:
        data[rc] = data[rc].map(lambda x: rfunc(x))
    return data

def replace_percent_subjects(data, p, rfunc=random_judgement):
    judgement_columns = data.columns[2]
    n = int(math.ceil(p * len(judgement_columns)))
    return replace_subjects(data, n, rfunc)

def blank_subjects(data, n):
    judgement_columns = data.columns[2:]
    removed_columns = sample(judgement_columns, n)
    data = data.copy()
    for rc in removed_columns:
        data[rc] = float('nan')
    return data



