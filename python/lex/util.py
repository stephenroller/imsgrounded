#!/usr/bin/env python

import sys
import logging

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

def remove_pos(word):
    try:
        return word[:word.rindex('/')]
    except ValueError:
        return word

def normalize(word):
    return word

def tsv_to_dict(corpus, keep_pos=True, leftind='target', rightind='context', valcol='value'):
    corpus_mem = {}
    for row in corpus:
        target = normalize(row[leftind])
        target = keep_pos and target or remove_pos(target)
        if target not in corpus_mem:
            corpus_mem[target] = dict()
        context = normalize(row[rightind])
        assert context not in corpus_mem[target], "Uh oh, found context '%s' twice for target '%s'?" % (context, target)
        corpus_mem[target][context] = row[valcol]
    return corpus_mem


