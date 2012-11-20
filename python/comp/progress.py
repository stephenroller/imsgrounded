#!/usr/bin/env python

import sys
import time
from datetime import datetime, timedelta
from math import floor

class ProgressBar(object):
    def __init__(self, max_value):
        self.max_value = max_value
        self.value = 0
        self.width = 40
        self.start_time = datetime.now()

    def incr(self, v=1):
        self.value += v

    def __str__(self):
        p = float(self.value) / self.max_value
        w = int(floor(self.width * p))
        if w == self.width:
            left = "=" * w
        else:
            left = "=" * (w - 1) + ">"

        right = "." * (self.width - len(left))

        elapsed = datetime.now() - self.start_time
        if self.value == 0:
            elapsed_s = "0:00:00"
            eta_s = "-:--:--"
        else:
            estimated_whole = elapsed.total_seconds() * (1.0 / p)
            eta = timedelta(seconds=estimated_whole) - elapsed
            elapsed_s = str(elapsed)[:7]
            eta_s = str(eta)[:7]


        return "[%s%s] (%5.1f%%)   ETA: %s   Elapsed: %s " % (left, right, p * 100, eta_s, elapsed_s)

    def errput(self):
        sys.stderr.write("\r%s" % str(self))
        if self.value == self.max_value:
            sys.stderr.write("\n")

    def incr_and_errput(self):
        self.incr()
        self.errput()


if __name__ == '__main__':
    N = 1000
    p = ProgressBar(N)
    for i in xrange(N):
        p.incr()
        p.errput()
        time.sleep(0.05)

