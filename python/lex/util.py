#!/usr/bin/env python

import sys
import logging

def readfile(filename):
    return [z.strip() for z in openfile(filename).readlines()]

def openfile(filename):
    if filename.endswith('.bz2'):
        #logger.info("Loading %s as bzip2 file." % filename)
        return bz2.BZ2File(filename)
    elif filename.endswith('.gz'):
        #logger.info("Loading %s as gzip file." % filename)
        return gzip.GzipFile(filename)
    elif filename == "-":
        return sys.stdin
    else:
        #logger.info("Loading %s as plain file." % filename)
        return open(filename)


