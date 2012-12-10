#!/usr/bin/env python

import pandas as pd

def rebin(data, mapping):
    data = data.copy()
    judgement_columns = data.columns[2:]
    for c in judgement_columns:
        data[c] = data[c].map(lambda x: x in mapping and mapping[x] or x)
    return data


