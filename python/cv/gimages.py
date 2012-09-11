#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
try:
    import json as simplejson
except ImportError:
    import simplejson
import urllib2
import urllib
import time

IP_ADDRESS = '141.58.160.225'
BASE_URL = 'https://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=%s&userip=%s&start=%d&hl=de'
REFERER = 'http://ims.uni-stuttgart.de/~roller/wordmeaning/%s'
SLEEP_TIME = 5000

results = []

def gimages(query, ip=IP_ADDRESS):
    escaped_query = urllib.quote('"%s"' % query)
    results = []
    for start in range(0, 100, 4):
        url = BASE_URL % (escaped_query, IP_ADDRESS, start)
        request = urllib2.Request(url, None, {'Referer': REFERER % escaped_query})
        response = urllib2.urlopen(request)
        json = simplejson.load(response)
        try:
            for result in json['responseData']['results']:
                unicode(results.append(result['unescapedUrl']))
        except TypeError:
            break
    return results


if __name__ == '__main__':
    for i, query in enumerate(sys.stdin):
        if i > 0:
            time.sleep(SLEEP_TIME / 1000.0)
        query = query.strip()
        if not query:
            continue
        for result in gimages(query):
            query_nice  = query.decode("utf-8").replace(" ", "_")
            print u"%s\t%s" % (query_nice, result)
        sys.stderr.write("%d complete...\n" % (i+1))

