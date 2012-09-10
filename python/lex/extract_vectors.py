#!/usr/bin/env python

import sys
import argparse

import util
import tsv

def read_corpus(files):
    for file in files:
        reader = tsv.read_tsv(file, ('target', 'context', 'value'), parsers={'value': unicode})
        for row in reader:
            yield row

def remove_pos(word):
    return word[:word.rindex('/')]

def filter_corpus(corpus, whitewords, has_pos=False):
    for row in corpus:
        # remove pos if necessary
        target = has_pos and row['target'] or remove_pos(row['target'])
        # normalize
        target = normalize(target)
        if target in whitewords:
            yield row

def normalize(word):
    return word.lower()

def read_whitelist(file):
    words = file.read().split()
    words = [normalize(w) for w in words]
    return set(words)

def main():
    parser = argparse.ArgumentParser(
                description='Extracts vectors from a TSV file of many vectors.')
    parser.add_argument("--input", "-i", action="append", type=util.openfile,
                        metavar="FILE", help='The input vector space.')
    parser.add_argument('--whitelist', '-w', metavar='FILE', type=util.openfile,
                        help='The list of target vectors to search for.')
    parser.add_argument('word', nargs='*', metavar='WORD',
                        help='Command line specified additional words.')
    parser.add_argument('--pos', '-p', action='store_true',
                        help='Marks that the whitelist has POS tags specified.')
    args = parser.parse_args()

    whitewords = set()
    if args.whitelist:
        whitewords.update(read_whitelist(args.whitelist))

    whitewords.update([normalize(w) for w in args.word])

    corpus = read_corpus(args.input)
    filtered = filter_corpus(corpus, whitewords, args.pos)

    tsv.print_tsv(filtered, ('target', 'context', 'value'), False)


if __name__ == '__main__':
    main()


