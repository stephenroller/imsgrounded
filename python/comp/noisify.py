#!/usr/bin/env python

from random import choice, randint, sample

def random_judgement():
    return randint(1, 7)

def randomize_values(data, p, rfunc=random_judgement):
    assert 0 <= p <= 1
    data = data.copy()
    judgement_columns = data.columns[2:]
    mask = data[judgement_columns].applymap(lambda x: False)
    n = round(mask.values.size * p)
    while n > 0:
        col = choice(judgement_columns)
        i = choice(data[col].index)
        if mask[col][i]:
            continue
        n -= 1
        mask[col][i] = True
        data[col][i] = rfunc()
    return data

def blank_values(data, p):
    assert 0 <= p <= 1
    data = data.copy()
    judgement_columns = data.columns[2:]
    mask = data[judgement_columns].applymap(lambda x: False)
    n = round(mask.values.size * p)
    while n > 0:
        col = choice(judgement_columns)
        i = choice(data[col].index)
        if mask[col][i]:
            continue
        n -= 1
        mask[col][i] = True
        data[col][i] = float('nan')
    return data


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



