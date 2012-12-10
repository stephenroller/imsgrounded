#!/usr/bin/env python

import sys
import argparse

from util import remove_pos, normalize, openfile

def filter_vecspace(vecspace, whitewords, has_pos=False, is_sorted=False):
    last_element = is_sorted and max(whitewords) or None
    for line in vecspace:
        line = line.rstrip("\n")
        row = line.split("\t")
        # remove pos if necessary
        target = has_pos and row[0] or remove_pos(row[0])
        # normalize
        target_norm = normalize(target)
        if target_norm in whitewords:
            yield line
        if last_element and not target.startswith(last_element) and target > last_element:
            # the incoming vector space is sorted, so we
            # know we've already seen the last word we're looking for.
            # no need to seek through the whole thing.
            break

def read_whitelist(file):
    words = file.read().split()
    words = [normalize(w) for w in words]
    return set(words)

def main():
    parser = argparse.ArgumentParser(
                description='Extracts vectors from a TSV file of many vectors.')
    parser.add_argument("--input", "-i", action="append", type=openfile,
                        metavar="FILE", help='The input vector space.')
    parser.add_argument('--whitelist', '-w', metavar='FILE', type=openfile,
                        help='The list of target vectors to search for.')
    parser.add_argument('word', nargs='*', metavar='WORD',
                        help='Command line specified additional words.')
    parser.add_argument('--sorted', '-s', action='store_true',
                        help='Indicates the incoming vector space is sorted, allowing for optimization.')
    parser.add_argument('--pos', '-p', action='store_true',
                        help='Marks that the whitelist has POS tags specified.')
    args = parser.parse_args()

    whitewords = set()
    if args.whitelist:
        whitewords.update(read_whitelist(args.whitelist))

    whitewords.update([normalize(w) for w in args.word])

    for input in args.input:
        for line in filter_vecspace(input, whitewords, args.pos, args.sorted):
            print line

if __name__ == '__main__':
    main()


