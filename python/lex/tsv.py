#!/usr/bin/env python

"""
A general library for handling Tab Seperated Value files.
"""

import csv
from itertools import izip

def read_tsv(file_or_filename, headers=False, delim="\t", parsers=None):
    if not isinstance(file_or_filename, file):
        file_or_filename = open(file_or_filename)

    kwargs = dict(delimiter=delim)

    if headers is False:
        # no header names. Emit tuples.
        assert parsers is None or isinstance(parsers, (list, tuple))
        csv_reader = csv.csvreader(file_or_filename, **kwargs)
    elif headers is True:
        # first line is the header. Read it in. Emit dicts.
        assert parsers is None or isinstance(parsers, dict)
        csv_reader = csv.DictReader(file_or_filename, **kwargs)
    else:
        # header names actually specified.
        assert parsers is None or isinstance(parsers, dict)
        csv_reader = csv.DictReader(file_or_filename, fieldnames=headers, **kwargs)


    for row in csv_reader:
        if headers is True:
            row = dict(zip(headers, row))
        if parsers:
            if headers is False:
                row = tuple(f(x) for f, x in izip(parsers, row))
            else:
                for k in row.keys():
                    row[k] = unicode(row[k], 'utf-8')
                for k,f in parsers.iteritems():
                    row[k] = f(row[k])
        yield row


def safe_encode(row):
    if isinstance(row, (tuple, list)):
        return [unicode(v).encode('utf-8') for v in row]
    elif isinstance(row, dict):
        return { k: unicode(v).encode('utf-8') for k,v in row.iteritems() }
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

