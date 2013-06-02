#!/usr/bin/env python

import codecs
import sys
from collections import defaultdict
from itertools import combinations

PLURALS = [
    ('ist', 'es ist', 'sind'),
    ('hat', 'es hat', 'haben'),
    ('gibt', 'es gibt'),
    ('besteht', 'es besteht'),
]

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
feature_counters = defaultdict(set)
cues = defaultdict(set)

with codecs.getreader('utf-8')(sys.stdin) as f:
    for line in f:
        cue, resp, cnt = line.strip().split('\t')
        for plural in PLURALS:
            for p in plural:
                if resp.startswith(p + ' '):
                    stem = resp[len(p)+1:]
                    feature_counters[p].add(stem)
                    cues[(stem,p)].add(cue)
                    break

needs_remap = set()
for plural in PLURALS:
    pplural = '/'.join(plural)
    intersection = set()
    for a, b in combinations((feature_counters[w] for w in plural), 2):
        intersection.update(a.intersection(b))
    for i in intersection:
        #print '\t'.join([i, pplural, ', '.join(sorted(cues[i]))])
        print '\t'.join([pplural, i])
        for p in list(plural)[1:]:
            needs_remap.update(cues[(i,p)])

print
print
print '\n'.join(sorted(needs_remap))





