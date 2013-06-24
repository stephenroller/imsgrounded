#!/usr/bin/env python

import argparse
import numpy as np
import logging
from collections import Counter
from rankedsim import utfopen
from nicemodel import load_labels
from aesir import itersplit

def tocdf(nonnormalized_dist):
    normalized = np.array(nonnormalized_dist) / float(np.sum(nonnormalized_dist))
    ids = normalized.argsort()[::-1]
    revsorted = normalized[ids]
    return zip(ids + 1, revsorted.cumsum())

def stochastic_choice(cdf):
    x = np.random.rand()
    for i, p in cdf:
        if x <= p:
            break
    return i

def load_features(file):
    with utfopen(file) as f:
        feature_vectors = [l.rstrip().split("\t") for l in f.readlines()]
        feature_vectors = [(w, map(float, v.split())) for w, v in feature_vectors]
        feature_cdfs = { w : tocdf(v) for w, v in feature_vectors }
        return feature_cdfs

def word_ids_to_features(vocab_dist, feature_dists):
    output = {}
    for wid, w in vocab_dist.iteritems():
        if not w.endswith('/NN'):
            continue
        wn = w[:w.rindex('/')]
        if wn in feature_dists:
            output[str(wid)] = feature_dists[wn]
    logging.info("%d words have features." % len(output))
    return output

def main():
    parser = argparse.ArgumentParser(description='Stochastically adds features to a corpus.')
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    parser.add_argument('--input', '-i', metavar='FILE',
                        help='The input corpus (in Andrews format).')
    parser.add_argument('--output', '-o', metavar='FILE',
                        help='The output corpus (in Andrews format).')
    parser.add_argument('--features', '-f', metavar='FILE',
                        help='The (dense) vector space of features.')

    args = parser.parse_args()
    vocab_labels = load_labels(args.vocab)
    features = load_features(args.features)
    feature_map = word_ids_to_features(vocab_labels, features)

    logging.info("First pass; gathering statistics.")
    inpt = utfopen(args.input)
    numlines = len(inpt.readlines())
    inpt.close()

    logging.info("Starting second pass; actually writing output.")
    output = open(args.output, 'w', 1024*1024)
    inpt = utfopen(args.input)
    for lno, line in enumerate(inpt.readlines(), 1):
        if lno % 1000 == 0:
            logging.info("Processing doc# %d/%d (%4.1f%%)" % (lno, numlines, 100*float(lno)/numlines))
        for chunk in itersplit(line, ' '):
            chunk = chunk.rstrip()
            if not chunk: continue
            idx = chunk.rindex(":")
            wid, cnt = chunk[:idx], chunk[idx+1:]
            if wid not in feature_map:
                output.write(chunk + ' ')
            else:
                cnt = int(cnt)
                dist = feature_map[wid]
                cnts = Counter(stochastic_choice(dist) for i in xrange(cnt))
                for fid, cnt in cnts.iteritems():
                    output.write('%s,%d:%d ' % (wid, fid, cnt))
        output.write('\n')

    inpt.close()
    output.close()


if __name__ == '__main__':
    main()

