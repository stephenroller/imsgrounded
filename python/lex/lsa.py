#!/usr/bin/env python

import argparse
from math import sqrt

import pandas as pd
import numpy as np
from numpy.linalg import svd

from util import read_vector_file
from matrix import norm2_matrix


def main():
    parser = argparse.ArgumentParser(
                description='Computes an LSA model correlating associations and lexsem model.')
    parser.add_argument('--lexspace', '-l', metavar='FILE',
                        help='The input lexical space.')
    parser.add_argument('--assoc', '-a', metavar='FILE',
                        help='The input association space.')
    args = parser.parse_args()

    lex = norm2_matrix(read_vector_file(args.lexspace))
    assoc = norm2_matrix(read_vector_file(args.assoc))
    together = pd.concat(lex, assoc, keys=("lex", "assoc"))

    org_matrix = together.as_matrix()
    U, S, V = svd(org_matrix)

    np.savez("svd.npz", U, S, V)


if __name__ == '__main__':
    main()


