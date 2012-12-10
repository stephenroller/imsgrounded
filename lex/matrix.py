#!/usr/bin/env python

import pandas as pd
import numpy as np

from util import norm2

def norm2_matrix(mat):
    for k in mat:
        mat[k] = norm2(mat[k])
    return mat.to_sparse(0)



