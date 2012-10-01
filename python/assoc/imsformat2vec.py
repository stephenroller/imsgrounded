#!/usr/bin/env python

import sys

words = {}

for line in sys.stdin:
    try:
        word, flag, assocs = line.strip().split("\t")
    except:
        continue
    if word not in words:
        words[word] = {}

    assocs = assocs.split(",")
    for assoc in assocs:
        words[word][assoc] = words[word].get(assoc, 0) + 1

for word, assocs in words.iteritems():
    for assoc, c in assocs.iteritems():
        print word + "\t" + assoc + "\t" + str(c)


