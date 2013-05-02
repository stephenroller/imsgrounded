#!/usr/bin/env python

import argparse
import logging
from aesir import itersplit
import runner

def main():
    parser = argparse.ArgumentParser(description='Outputs a human readable model.')
    parser.add_argument('--input', '-i', metavar='FILE',
                        help='The input corpus (in Andrews format).')
    parser.add_argument('--output', '-o', metavar='FILE',
                        help='The output corpus (in Andrews format).')

    args = parser.parse_args()

    input = open(args.input)
    output = open(args.output, 'w')
    for i, line in enumerate(input.readlines()):
        line = line.rstrip()
        output_lst = []
        for w in itersplit(line, ' '):
            if not w.strip(): continue
            if ',' not in w: continue
            output_lst.append(w[w.index(',')+1:])
        output.write(" ".join(output_lst) + "\n")
        logging.info("Document %d: %d words written." % (i, len(output_lst)))

    input.close()
    output.close()

if __name__ == '__main__':
    main()

