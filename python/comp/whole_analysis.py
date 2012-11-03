#!/usr/bin/env python

import sys
import pandas as pd

def parse_whole_judgements(file):
    data = pd.read_csv(file, names=('workerId', 'stuff', 'stuff2', 'compound', 'judgement'))
    data = data.fillna(8)
    # don't need these columns
    del data['stuff']
    del data['stuff2']
    # filter out nonanswers
    data = data[data.judgement != 8]
    # group into users
    output = data.pivot(index='compound', columns='workerId', values='judgement')
    workers = list(output.columns)
    output['compound'] = output.index
    output['const'] = output['compound']
    output = output.reindex(columns=['compound', 'const'] + workers)
    return output

parse_whole_judgements(sys.argv[1]).to_csv(sys.stdout, index=False)

