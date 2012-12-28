#!/usr/bin/env python

import sys
import pandas as pd

import fetch
from progress import ProgressBar

USERNAME = 'stephenroller'
ACCESS_KEY = '5b22bd3303fe4c21a4463d37cca0353813a56109'

MAPPING_URL = 'http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=%s'
HYPO_URL = 'http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=%s&full=1'
TARBALL_URL = 'http://www.image-net.org/download/synset?username=%s&accesskey=%s&release=latest&wnid=%%s' % (USERNAME, ACCESS_KEY)

mappings = pd.read_csv(sys.argv[1], sep="\t")

synsets = mappings.Synset[mappings.Synset.notnull()]
synsets = [y for x in synsets.map(lambda z: z.split()) for y in x]

def fetch_image_urls(synset):
    data = fetch.fetch_data(MAPPING_URL % synset)
    image_mappings = [y.split() for y in data.split("\r\n") if y]
    return image_mappings

def fetch_hypos(synset):
    data = fetch.fetch_data(HYPO_URL % synset)
    return data.replace("-", "").split("\r\n")


pb = ProgressBar(len(synsets))
pb.errput()
for synset in synsets:
    image_urls = fetch_image_urls(synset)
    if len(image_urls) == 0:
        children_synsets = fetch_hypos(synset)
        children_urls = [fetch_image_urls(cs) for cs in children_synsets]
        image_urls = [y for x in children_urls for y in x]

    for imgid, url in image_urls:
        print "%s\t%s\t%s" % (synset, imgid, url)

    pb.incr_and_errput()







