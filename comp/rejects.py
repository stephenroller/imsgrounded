import pandas as pd
from standard_cleanup import calculate_subject_agreements
import math
import sys


df1 = pd.read_csv("../../../data/comp/rejected-processed.txt")
df2 = pd.read_csv("../../../data/comp/amt_reshaped.csv")
left = ['compound', 'const']
badsubj = set(df1.columns) - set(df2.columns)
aggr = calculate_subject_agreements(df1)
for u, a in aggr.iteritems():
    if not isinstance(a, float) or math.isnan(a):
        aggr[u] = -1

aggr_sorted = sorted(aggr.iteritems(), key=lambda x: x[1])

goods = sum(u in df2.columns for u, a in aggr_sorted)
bads = sum(u not in df2.columns for u, a in aggr_sorted)

#print sum(u in badsubj for u, a in aggr_sorted[:bads])/float(bads)
# 0.8152173913043478
#print sum(u in badsubj for u, a in aggr_sorted[-goods:])/float(goods)
# 0.11333333333333333

# out = pd.DataFrame(aggr.items(), columns=('subject', 'agreement'))
# out['good'] = out.subject.map(lambda x: x in df2.columns and "Accepted" or "Rejected")
# out.to_csv(sys.stdout, index=False)

out = []
for i in xrange(1, len(aggr_sorted)):
    found_goods = sum(u in df2.columns for u, a in aggr_sorted[-i:])
    # found_bads = sum(u not in df2.columns for u, a in aggr_sorted[-i:])

    precision = float(found_goods) / i
    recall = float(found_goods) / goods
    fscore = 2 * precision * recall / (precision + recall)
    bound = aggr_sorted[-i][1]

    out.append(dict(i=i, precision=precision, recall=recall, f=fscore, bound=bound))

pd.DataFrame(out).to_csv(sys.stdout, index=False)

