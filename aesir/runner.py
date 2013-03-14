#!/usr/bin/env python

import sys
import aesir
import datetime
now = datetime.datetime.now

data_f, labels_f, features_f, k = sys.argv[1:]
k = int(k)

print "[%s] Loading Data..." % now()
data = aesir.dataread(data_f)
init_time = now()
print "[%s] Initing Model..." % init_time
model = aesir.freyr(data, K=k)
mcmc_time = now()
print "Initialization time: %s" % (mcmc_time - init_time)
print "[%s] Starting MCMC..." % mcmc_time
model.mcmc()
mcmc_stop_time = now()
print "[%s] Done with MCMC!" % mcmc_stop_time
print "Time for MCMC: %s" % (mcmc_stop_time - mcmc_time)
model.getvocablabels(labels_f)
model.getfeaturelabels(features_f)
print "[%s] Loaded labels..." % now()
print
print
model.printlatentlabels(50)
print "[%s] Done!" % now()


