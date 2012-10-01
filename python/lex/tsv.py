#!/usr/bin/env python

"""
A general library for handling Tab Seperated Value files.
"""

import csv
from collections import OrderedDict

from itertools import izip, chain

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
        tsv_reader = (OrderedDict(zip(headers, row)) for row in tsv_reader)

    if isinstance(parsers, dict):
        assert headers, "We have to have headers to convert to a dictionary!"
        tsv_reader = ( {k : (k in parsers and parsers[k](v) or v)
                       for k, v in row.iteritems() }
                       for row in tsv_reader )

    return tsv_reader


def read_many_tsv(files_or_filenames, *args, **kwargs):
    return chain(*[read_tsv(f, *args, **kwargs) for f in files_or_filenames])

def safe_encode(row):
    if isinstance(row, (str, unicode)):
        return unicode(row.decode("utf-8")) #.encode("utf-8")
    elif isinstance(row, (int, float)):
        return unicode(row).decode("utf-8")
    elif isinstance(row, (tuple, list)):
        return [safe_encode(v) for v in row]
    elif isinstance(row, dict):
        return { k: safe_encode(v) for k,v in row.iteritems() }
    else:
        raise ValueError, "I don't know how to handle data of type '%s'." % typeof(row)

def write_tsv(file_or_filename, data, headers=None, write_header=True, delim=u"\t"):
    if not isinstance(file_or_filename, file):
        file_or_filename = open(file_or_filename, 'w')

    try:
        first_row = iter(data).next()
    except StopIteration:
        first_row = []
    fieldnames = headers
    if headers is True:
        if isinstance(first_row, dict):
            fieldnames = first_row.keys()
        else:
            fieldnames = first_row

    if fieldnames and write_header:
        outstr = safe_encode(delim.join(fieldnames) + "\n")
        file_or_filename.write(outstr)

    if isinstance(first_row, dict) or fieldnames is headers:
        data = chain([first_row], data)

    for row in data:
        row = safe_encode(row)
        if isinstance(row, dict):
            write_order = [row[field] for field in fieldnames]
            outstr = delim.join(write_order) + u"\n"
        else:
            outstr = delim.join(row) + u"\n"
        try:
            file_or_filename.write(outstr.encode("utf-8"))
        except IOError:
            break


def print_tsv(data, headers=None, write_header=True, delim="\t"):
    import sys
    write_tsv(sys.stdout, data, headers, write_header, delim)

