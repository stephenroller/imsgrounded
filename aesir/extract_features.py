#!/usr/bin/env python

import argparse
import numpy as np
import logging
from collections import Counter
from rankedsim import utfopen, utfopenwrite
from nicemodel import load_labels
from aesir import itersplit

def main():
    parser = argparse.ArgumentParser(description='Stochastically adds features to a corpus.')
    parser.add_argument('--vocab', '-v', metavar='FILE',
                        help='The vocab labels.')
    parser.add_argument('--input', '-i', metavar='FILE',
                        help='The input corpus (in Andrews format). Must be multimodal.')
    parser.add_argument('--output', '-o', metavar='FILE',
                        help='The output corpus (in Andrews format).')
    parser.add_argument('--outvocab', '-V', metavar='FILE',
                        help='The output vocab labels; necessary for OOV processing later.')

    args = parser.parse_args()
    vocab_labels = load_labels(args.vocab)

    logging.info("First pass; gathering statistics.")
    inpt = open(args.input)
    numlines = len(inpt.readlines())
    inpt.close()

    output_labels = {}
    output_labels_file = utfopenwrite(args.outvocab)

    logging.info("Starting second pass; actually writing output.")
    output = open(args.output, 'w', 1024*1024)
    inpt = open(args.input)
    for lno, line in enumerate(inpt.readlines(), 1):
        if lno % 1000 == 0:
            logging.info("Processing doc# %d/%d (%4.1f%%)" % (lno, numlines, 100*float(lno)/numlines))

        outline = []
        for chunk in itersplit(line, ' '):
            chunk = chunk.rstrip()
            if not chunk: continue
            if ',' not in chunk: continue # strip just words
            idx = chunk.index(',')
            wid = int(chunk[:idx])
            rest = chunk[idx:]

            if wid not in output_labels:
                output_labels[wid] = len(output_labels) + 1
                output_labels_file.write("%d\t" % output_labels[wid])
                output_labels_file.write(vocab_labels[wid])
                output_labels_file.write("\n")
            outline.append(str(output_labels[wid]) + rest)

        if outline:
            output.write(' '.join(outline))
            output.write('\n')

    inpt.close()
    output.close()


if __name__ == '__main__':
    main()

