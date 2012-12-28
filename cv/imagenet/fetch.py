#!/usr/bin/env python

import urllib2

# b/c fuck the police
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.97 Safari/537.11"

def make_request(url):
    request = urllib2.Request(url)
    request.add_header('User-Agent', USER_AGENT)
    return request

def fetch(url):
    req = make_request(url)
    opener = urllib2.build_opener()
    resp = opener.open(req)
    return resp.getcode(), resp.headers.dict, resp.read()

def fetch_data(url):
    rc, h, d = fetch(url)
    if rc != 200:
        raise ValueError, "Non-200 response code (%d)" % rc
    return d


def download(url, outpath):
    respcode, headers, data = fetch(url)
    if respcode != 200:
        raise ValueError, "Didn't get a 200 response code (%d)." % rc
    outf.write(data)
    outf.close()
    resp.close()
    return headers



