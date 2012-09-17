#!/usr/bin/env python

"""
Joins two TSV files by a given set of key(s)

This program loads the whole TSV files into memory, and *requires* headers.
"""

import sys
import argparse
from itertools import izip
from collections import OrderedDict
# hack to get things to work in sets
OrderedDict.__hash__ = lambda x: id(x)


import tsv
from util import remove_pos, normalize, openfile


def union_records(*records):
    """
    Creates a new record which is all the given records concatenated.
    """
    if isinstance(records[0], (list, tuple)):
        return reduce(lambda x, y: x + y, records)
    else:
        new_record = OrderedDict()
        for r in records:
            for k, v in r.iteritems():
                new_record[k] = v
        return new_record

def cartesian(left, right):
    for row in left:
        for row2 in right:
            yield union_records(row, row2)

def join_on(keys, left, right):
    # let's index all the data
    right_indexes = []
    # we really only need to index the right side
    for left_keyname, right_keyname in keys:
        key_index = {}
        for row in right:
            key_value = row[right_keyname]
            if key_value not in key_index:
                key_index[key_value] = set()
            key_index[key_value].add(row)
        right_indexes.append(key_index)

    # okay now let's actually do this
    for row in left:
        intersections = []
        for (left_keyname, right_keyname), index in izip(keys, right_indexes):
            join_value = row[left_keyname]
            try:
                intersections.append(index[join_value])
            except KeyError:
                pass
        final = intersections and reduce(set.intersection, intersections) or None
        if final:
            for rightside in final:
                yield union_records(row, rightside)



def main():
    parser = argparse.ArgumentParser(
                description='Extracts vectors from a TSV file of many vectors.')
    parser.add_argument("--left", "-l", type=openfile,
                        metavar="FILE", help='The left TSV file.')
    parser.add_argument("--right", "-r", type=openfile,
                        metavar="FILE", help='The right TSV file.')
    parser.add_argument('keys', nargs='*', metavar='KEYS',
                        help=('Keys to join on. If none are given, then it performs cartesian '
                              'product. For keys of different names, you can use a key like '
                              '"left=right" to join the left file to the right file with key '
                              '"left" on the left file and "right" on the right file.'))

    # todo: support other kinds of joins, as cli options
    args = parser.parse_args()

    keys = []
    for key in args.keys:
        if '=' in key:
            left, right = key.split("=")
            keys.append((left, right))
        else:
            keys.append((key, key))

    left = list(tsv.read_tsv(args.left, headers=True))
    right = list(tsv.read_tsv(args.right, headers=True))

    if len(keys) == 0:
        tsv.print_tsv(cartesian(left, right), headers=True, write_header=True)
    else:
        joined = join_on(keys, left, right)
        tsv.print_tsv(joined, headers=True, write_header=True)

if __name__ == '__main__':
    main()





