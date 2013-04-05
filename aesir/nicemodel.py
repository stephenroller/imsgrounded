#!/usr/bin/env python
# -*- coding: utf-8 -*

import argparse
import numpy as np
import sys
import codecs

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

TOPIC_WORDS_SHOW = 15
TOPIC_FEATS_SHOW = 5

def load_labels(filename):
    if not filename:
        return {}
    else:
        with codecs.getreader('utf-8')(open(filename)) as f:
            lines = (l.strip().split() for l in f.readlines() if l.strip())
            return {int(i):w for i,w in lines}

def ranked_list(probabilities, n):
    return sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:n]

def pad_same(column, extra=2):
    width = max(len(l) for l in column) + extra
    return [l + (width - len(l)) * " " for l in column]

def column_join(list_of_columns, connector=' | '):
    return '\n'.join(connector.join(line) for line in zip(*list_of_columns))

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
    parser.add_argument('--docs', '-D', metavar='FILE',
                        help='Output the document distributions for these documents.')
    parser.add_argument('--docids', '-d', metavar='FILE',
                        help='The document labels.')
    parser.add_argument('--detailedtopics', '-p', action='store_true',
                        help='Output nice, human readable information about documents.')
    args = parser.parse_args()

    model = np.load(args.model)

    label_vocab = load_labels(args.vocab)
    label_features = load_labels(args.features)

    #print "Loglikelihood: %.5f" % model["loglikelihoods"][-1]

    # phi is vocab
    # psi is features
    # pi is documents

    topic_strings = {}

    if args.topics or args.detailedtopics:
        for k in xrange(model['k']):
            bestphi = ranked_list(model['phi'][k], TOPIC_WORDS_SHOW)
            bestpsi = ranked_list(model['psi'][k], TOPIC_FEATS_SHOW)

            topic_str = []
            topic_str.append("Topic %d:" % k)
            topic_str.append("  Phi (vocab):")
            for i, p in bestphi:
                topic_str.append("    %.5f  %s" % (p, label_vocab.get(i, "word_%d" % i)))
            topic_str.append("  Psi (features):")
            for i, p in bestpsi:
                topic_str.append("    %.5f  %s" % (p, label_features.get(i, "feat_%d" % i)))

            if args.topics:
                print '\n'.join(topic_str)
            if args.detailedtopics:
                topic_strings[k] = pad_same(topic_str)



    if args.docs:
        docids = codecs.getreader('utf-8')(open(args.docids)).readlines()
        docids = (d[:d.rindex('/')] for d in docids)
        docids = {dname: dnum for dnum, dname in enumerate(docids)}
        whitedocs = list(codecs.getreader('utf-8')(open(args.docs)).read().split())
        for docname in whitedocs:
            try:
                docid = docids[docname]
            except KeyError:
                pass
            docdist = model['pi'][docid]
            if not args.detailedtopics:
                docdist_s = ' '.join(map(repr, docdist))
                #print "%s\t%s" % (docname, docdist_s)
                for i, p in enumerate(docdist):
                    print "%s,%d,%f" % (docname, i, p)
            else:
                # output nice stuff.
                sorted_dist = sorted(list(enumerate(docdist)), key=lambda x: x[1], reverse=True)
                sorted_dist = [(i, p) for i, p in sorted_dist if p > 1e-6]
                print "Document: %s" % docname
                for i, p in sorted_dist:
                    print "  Topic %d: %f" % (i, p)
                print column_join([topic_strings[k] for k, p in sorted_dist])
                print



    if args.words:
        whitewords = codecs.getreader('utf-8')(open(args.words)).read().split()
        mappings = {}
        for k,v in label_vocab.iteritems():
            nicev = v[:v.rindex("/")]
            if nicev in whitewords and '/NN' in v:
                mappings[nicev] = k

        for ww in whitewords:
            if ww not in mappings:
                continue
            m = mappings[ww]
            word = label_vocab[m]
            probs = model['phi'][:,m]
            probs = probs / np.sum(probs)
            niceword = word[:word.rindex("/")]
            if args.detailedtopics:
                niceprobs = [(i, p) for i, p in enumerate(probs) if p > 1e-4]
                print "Word: %s" % niceword
                for i, p in niceprobs:
                    print "  Topic %d: %f" % (i, p)
                print column_join([topic_strings[k] for k, p in niceprobs])
                print
            else:
                #print word[:word.rindex("/")] + "\t" + " ".join(str(p) for p in probs)
                for i, p in enumerate(probs):
                    print "%s,%d,%f" % (niceword, i, p)


if __name__ == '__main__':
    main()



