#!/usr/bin/env python

import sys
import aesir
import datetime
now = datetime.datetime.now

data_f, labels_f, features_f, k = sys.argv[1:]
k = int(k)

print "[%s] Loading Data..." % now()
data = aesir.dataread(data_f)
print "[%s] Initing Model..." % now()
model = aesir.freyr(data, K=k)
print "[%s] Starting MCMC..." % now()
model.mcmc()
print "[%s] Done with MCMC!" % now()
model.getvocablabels(labels_f)
model.getfeaturelabels(features_f)
print "[%s] Loaded labels..." % now()
print
print
model.printlatentlabels(50)
print "[%s] Done!" % now()


