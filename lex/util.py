#!/usr/bin/env python

import sys
import logging
import pandas as pd
import gzip
import bz2

from math import sqrt

def readfile(filename):
    return [z.strip() for z in openfile(filename).readlines()]

def openfile(filename):
    if filename.endswith('.bz2'):
        #logger.info("Loading %s as bzip2 file." % filename)
        return bz2.BZ2File(filename)
    elif filename.endswith('.gz'):
        #logger.info("Loading %s as gzip file." % filename)
        return gzip.GzipFile(filename)
    elif filename == "-":
        return sys.stdin
    else:
        #logger.info("Loading %s as plain file." % filename)
        return open(filename)

def read_vector_file(file_or_filename):
    rawdf = pd.read_table(file_or_filename, names=("target", "context", "value"))
    piv = rawdf.pivot("context", "target", "value")
    sp = piv.fillna(0).to_sparse(0)
    return sp

def df_remove_pos(dataframe):
    newindex = map(remove_pos, dataframe.columns)
    return dataframe.rename(columns=dict(zip(dataframe.columns, newindex)))

def remove_pos(word):
    try:
        return word[:word.rindex('/')]
    except ValueError:
        return word

def extract_word_pos(word):
    try:
        splt = word.rindex('/')
        return word[:splt], word[splt+1:]
    except ValueError:
        return word, ''


def normalize(word):
    return word

def norm2(vec):
    return vec / sqrt(vec.dot(vec))

