#!/usr/bin/env python

import sys
import math
import collections
import heapq
import bz2
import gzip
import argparse
import logging
from functools import partial

import util

logger = logging.getLogger("transform")
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)-8s  %(message)s"))
logger.addHandler(log_handler)

# general utilities
# ------------------------------------------------------------------------------

def log2(x):
    return math.log(x, 2)

# vectorspace stuff
# ------------------------------------------------------------------------------

class VectorSpace(object):
    def __init__(self, files):
        self.files = files

    def __true_iter(self):
        logger.info("Resetting all files.")
        for infile in self.files:
            infile.seek(0)
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                target, context, value = line.split("\t")
                value = float(value)
                if value:
                    yield target, context, value

    transformations = []
    def add_transformation(self, func):
        self.transformations.append(func)

    def __iter__(self):
        iterator = self.__true_iter()
        for func in self.transformations:
            iterator = func(iterator)
        return iterator


def count_masses(vectorspace):
    logger.info("Counting mass.")
    targets = collections.defaultdict(int)
    contexts = collections.defaultdict(int)

    total_mass = 0
    for target, context, value in vectorspace:
        targets[target] += value
        contexts[context] += value
        total_mass += value

    logger.info("Total mass: %f" % total_mass)
    return total_mass, targets, contexts

def mutual_information(mode, counted_masses, vectorspace):
    logger.info("Computing mutual information (%s)" % mode)
    # needs two passes, so you need to pass it the "same" vectorspace
    # twice.
    # PMI = log [ p(x,y)/(p(x)*p(y)) ]
    #     = log p(x,y) - (log p(x) + log p(y))
    #     = log [ c(x,y) / c(*,*) ] - { log [ c(x,*)/c(*,*) ] + log [ c(*,y)/c(*,*) ]
    #     = log c(x,y) - log c(*,*) - { log c(x,*) - log c(*,*) + log c(*,y) - log c(*,*) }
    #     = log c(x,y) - log c(*,*) - log c(x,*) + log c(*,*) - log c(*,y) + log c(*,*)
    #     = log c(x,y) + log(*,*) - log c(x,*) - log c(*,y)
    total_mass, targets, contexts = counted_masses
    tm_log = log2(total_mass)
    processed_targets = set()
    for target, context, value in vectorspace:
        pmi = log2(value) + tm_log - log2(targets[target]) - log2(contexts[context])
        if mode == 'lmi':
            freq = value / total_mass
            transformed_value = freq * pmi
        elif mode == 'pmi':
            transformed_value = pmi
        else:
            raise ValueError("'%s' is not a valid mode for mutual_information." % mode)
        if target not in processed_targets:
            processed_targets.add(target)
            logger.info("Transforming '%s' (%d/%d)" % (target, len(processed_targets), len(targets)))
        yield target, context, transformed_value

def positive(vectorspace):
    logger.info("Removing nonpositive values.")
    return ((t,c,v) for t,c,v in vectorspace if v > 0)

def find_top(n, vectorspace):
    logger.info("Keeping only the top %d dimensions." % n)
    total_mass, targets, contexts = count_masses(vectorspace)
    top_contexts = heapq.nlargest(n, contexts)
    return top_contexts

def keep_contexts(words, vectorspace):
    words = set(words)
    logger.info("Keeping dimensions: %s" % words)
    return ((t,c,v) for t,c,v in vectorspace if c in words)

def remove_contexts(words, vectorspace):
    words = set(words)
    logger.info("Removing dimensions: %s" % words)
    return ((t,c,v) for t,c,v in vectorspace if c not in words)

def output_pairs(outfile, vectorspace):
    logger.info("Outputting as pairs.")
    for t,c,v in vectorspace:
        outfile.write("%s\t%s\t%.25f\n" % (t, c, v))

def norm1(counted_masses, vectorspace):
    total_mass, targets, contexts = counted_masses
    return ((t, c, v / targets[t]) for t,c,v in vectorspace)

def prob(vectorspace, total_mass):
    return ((t,c, v / total_mass) for t,c,v in vectorspace)

def neglogprob(vectorspace, total_mass):
    ltm = log2(total_mass)
    return ((t,c, - log2(v) + ltm) for t,c,v in vectorspace)

def parse_args():
    # this is a complicated system for argument parsing, let's go.
    parser = argparse.ArgumentParser(description="Transform a tab separated, pairs vectorspace")
    # allow input from stdin or a file (compressed or otherwise)
    parser.add_argument("--stopwords", "-s", type=util.readfile, metavar="STOPFILE",
                        help="Removes context which appear in the list of stopwords.")
    parser.add_argument("--keepn", "-n", type=int, metavar="N",
                        help="Keeps only the top N contexts.")
    parser.add_argument("--output", "-o", type=argparse.FileType("w"), default=sys.stdout,
                        help="Output to the given filename." )
    parser.add_argument("--outformat", "-O", metavar="OUTPUT FORMAT", 
                        default="pairs", choices=["pairs"], #, "stripes", "dense", "contexts"],
                        help="Output format. Currently only 'pairs' is supported.")
    parser.add_argument("transformation", metavar="METHOD", nargs="+",
                        choices=["nop", "pmi", "lmi", "tfidf", "prob", "neglogprob", "norm1", "positive"],
                        help="Transformation method. Possible values are: %(choices)s.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show logger information.")
    parser.add_argument("--input", "-i", action="append", type=util.openfile, metavar="FILE",
                        help=("The input vector space. Multiple files may be specified with "
                              "multiple -i's, but target-contexts are assumed to be unique."))

    return parser.parse_args()


def main(args):
    # load up the vectorspace
    vectorspace = VectorSpace(args.input)

    # filter stopwords
    if args.stopwords:
        vectorspace.add_transformation(partial(remove_contexts, args.stopwords))

    if args.keepn:
        top_dimensions = find_top(args.keepn, vectorspace)
        vectorspace.add_transformation(partial(keep_contexts, top_dimensions))

    for transformation in args.transformation:
        if transformation == "nop":
            continue
        # otherwise, we require context and target counts. compute them.
        counted_masses = count_masses(vectorspace)
        if transformation == "lmi" or transformation == "pmi":
            vectorspace.add_transformation(partial(mutual_information, transformation, counted_masses))
        elif transformation == "norm1":
            vectorspace.add_transformation(partial(norm1, counted_masses))
        elif transformation == "prob":
            vectorspace.add_transformation(partial(prob, total_mass=counted_masses[0]))
        elif transformation == "neglogprob":
            vectorspace.add_transformation(partial(neglogprob, total_mass=counted_masses[0]))
        elif transformation == "positive":
            vectorspace.add_transformation(positive)
        else:
            raise NotImplementedError("Transformation '%s' not supported yet." % transformation)

    if args.outformat == 'pairs':
        output_pairs(args.output, vectorspace)
    else:
        raise NotImplementedError("Can't output as %s" % args.outformat)

if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled..")
        logger.info("Command line options: %s" % args)
    main(args)

