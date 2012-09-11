#!/usr/bin/env python

"""
A general library for handling Tab Seperated Value files.
"""

import csv

from itertools import izip

def parse_tsv(line_iter, delim="\t", strip=True):
    for line in line_iter:
        if strip:
            line = line.strip()
        # always use unicode
        line = unicode(line.decode("utf-8")).encode("utf-8")
        yield tuple(line.split("\t"))

def read_tsv(file_or_filename, headers=False, delim="\t", parsers=None):
    if not isinstance(file_or_filename, file):
        file_or_filename = open(file_or_filename)

    tsv_reader = parse_tsv(file_or_filename, delim, True)

    if headers is True:
        # first line is the header. Read it in. Emit dicts.
        headers = tsv_reader.next()

    if isinstance(parsers, (tuple, list)):
        # parsers is a list of type converters. go ahead and apply them.
        tsv_reader = (f(x) for f, x in zip(parsers, row) for row in tsv_reader)

    if headers:
        tsv_reader = (dict(zip(headers, row)) for row in tsv_reader)

    if isinstance(parsers, dict):
        assert headers, "We have to have headers to convert to a dictionary!"
        tsv_reader = ( {k : (k in parsers and parsers[k](v) or v)
                       for k, v in row.iteritems() }
                       for row in tsv_reader )

    return tsv_reader


def safe_encode(row):
    if isinstance(row, (tuple, list)):
        return [unicode(v).encode('utf-8') for v in row]
    elif isinstance(row, dict):
        return { k: unicode(v.decode("utf-8")).encode('utf-8') for k,v in row.iteritems() }
    else:
        raise ArgumentError, "I don't know how to handle data of type '%s'." % typeof(row)

def write_tsv(file_or_filename, data, headers=None, write_header=True, delim="\t"):
    if not isinstance(file_or_filename, file):
        file_or_filename = open(file_or_filename, 'w')

    kwargs = dict(delimiter=delim)

    first_row = data.next()
    if isinstance(first_row, dict):
        if not headers:
            fieldnames = sorted(first_row.keys())
        else:
            fieldnames = headers
        csv_writer = csv.DictWriter(file_or_filename, fieldnames, **kwargs)
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerow(safe_encode(first_row))
    else:
        csv_writer = csv.writer(file_or_filename, **kwargs)
        if write_header:
            assert isinstance(headers, (list, tuple))
            csv_writer.writerow(safe_encode(headers))
        csv_writer.writerow(safe_encode(first_row))

    for row in data:
        csv_writer.writerow(safe_encode(row))

def print_tsv(data, headers=None, write_header=True, delim="\t"):
    import sys
    write_tsv(sys.stdout, data, headers, write_header, delim)

