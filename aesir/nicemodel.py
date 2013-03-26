#!/usr/bin/env python

import argparse
import numpy as np

TOPIC_WORDS_SHOW = 15
TOPIC_FEATS_SHOW = 5

def load_labels(filename):
    if not filename:
        return {}
    else:
        with open(filename) as f:
            lines = (l.strip().split() for l in f.readlines() if l.strip())
            return {int(i):w for i,w in lines}

def ranked_list(probabilities, n):
    return sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:n]

def main():
    parser = argparse.ArgumentParser(description='Outputs a human readable model.')
    parser.add_argument('--model', '-m', metavar='FILE',
                        help='The saved model.')
    parser.add_argument('--topics', '-t', action='store_true',
                        help='Output topics in human readable form.')
    parser.add_argument('--words', '-w', metavar='FILE',
                        help='Output the distributions of these words.')
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    parser.add_argument('--features', '-f', metavar='FILE',
                        help='The feature labels.')
    parser.add_argument('--docs', '-d', action='store_true',
                        help='Output the document distributions.')
    args = parser.parse_args()

    model = np.load(args.model)

    label_vocab = load_labels(args.vocab)
    label_features = load_labels(args.features)

    # phi is vocab
    # psi is features
    # pi is documents
    if args.topics:
        for k in xrange(model['k']):
            bestphi = ranked_list(model['phi'][k], TOPIC_WORDS_SHOW)
            bestpsi = ranked_list(model['psi'][k], TOPIC_FEATS_SHOW)

            print "Topic %d:" % k
            print "  Phi (vocab):"
            for i, p in bestphi:
                print "    %.5f  %s" % (p, label_vocab.get(i, "word_%d" % i))
            print "  Psi (features):"
            for i, p in bestpsi:
                print "    %.5f  %s" % (p, label_features.get(i, "feat_%d" % i))

            print


if __name__ == '__main__':
    main()


