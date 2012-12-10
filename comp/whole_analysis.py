#!/usr/bin/env python

import sys
import pandas as pd
from collections import Counter

fake_words = set("""
    Analigzerbruch Armmoder Brattlider Bulkerzagen Engschogen Fennhoder Harmweg
    Luderschwege Malligwohmer Pillinrugen Quetpfluge Tropebuhle Wierzverkuhr
    Zogschucht
""".split())

def remove_fakes(data):
    fake_judgements = data[data.compound.map(fake_words.__contains__)]
    failed_workers = set(data[data.judgement != 8].workerId)

    without_fakes = data[data.workerId.map(failed_workers.__contains__)]
    sys.stderr.write("Before removing fakes: %d\tafter: %d\n" % (len(data), len(without_fakes)))
    return without_fakes


def parse_whole_judgements(file):
    data = pd.read_csv(file, names=('workerId', 'worktime', 'approval', 'compound', 'judgement'))
    # remove everyone who failed our fake word tests
    #data = remove_fakes(data)
    ## only keep people with an approval of at least 90.
    #N = len(data)
    #data.approval = data.approval.map(lambda x: int(x.split()[0]))
    #data = data[data.approval >= 90]
    #sys.stderr.write("Before approval rating: %d\tafter: %d\n" % (N, len(data)))
    ## Now we only want to keep people who made at least 10 judgements
    # filter out nonanswers
    user_counts = Counter(data.workerId)
    good_users = set(k for k, v in user_counts.iteritems() if v >= 10)
    #N = len(data)
    data = data[data.workerId.map(good_users.__contains__)]
    #sys.stderr.write("Before min count: %d\tafter: %d\n" % (N, len(data)))
    data = data[data.judgement != 8]



    # group into users
    output = data.pivot(index='compound', columns='workerId', values='judgement')
    workers = list(output.columns)
    output['compound'] = output.index
    output['const'] = output['compound']
    output = output.reindex(columns=['compound', 'const'] + workers)
    return output

parse_whole_judgements(sys.argv[1]).to_csv(sys.stdout, index=False)

