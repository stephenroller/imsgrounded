#!/usr/bin/env python

import sys
import bz2file
import tarfile
import SimpleCV as scv
import PIL.ImageFile
import argparse
import os.path
import numpy as np
from itertools import chain
from lxml import etree
from random import random
from cStringIO import StringIO
import math

#from progress import ProgressBar

def forgiving_taropen(**tar_kw):
    try:
        tf = tarfile.open(debug=1, **tar_kw)
        tarinfos = list(tf)
        #sys.stderr.write("Tarfile has %d file in it...\n" %  len(tarinfos))
        return ((m.name, tf.extractfile(m)) for m in tarinfos if m.isfile())
    except Exception, e:
        sys.stderr.write("ERROR: Couldn't read '%s' due to '''%s'''\n" % (repr(tar_kw), repr(e)))
        return []

def yield_imagefiles(tarfilename):
    if tarfilename == "-":
        buffered_file = StringIO(sys.stdin.read())
        #sys.stderr.write("Reading tarfile from stdin...(%d bytes)\n" % len(buffered_file.getvalue()))
        return forgiving_taropen(fileobj=buffered_file)
    else:
        return forgiving_taropen(name=tarfilename)

def image_from_imagefile(imgfile):
    parser = PIL.ImageFile.Parser()
    s = imgfile.read()
    #sys.stderr.write("Read in image: %d bytes\n" % len(s))
    imgfile.seek(0)
    parser.feed(imgfile.read())
    pilimg = parser.close()
    imgfile.close()
    return scv.Image(pilimg)

def dict2obj(dict):
    class Obj:
        def __init__(self, entries):
            self.__dict__.update(entries)
    return Obj(dict)

def xml_to_objects(node):
    if len(node.getchildren()) == 0:
        try:
            return int(node.text)
        except ValueError:
            return node.text
    else:
        return dict2obj({c.tag : xml_to_objects(c) for c in node.getchildren()})

def relcrop(img, owidth, oheight, x1, y1, x2, y2):
    if img.width == owidth and img.height == oheight:
        return img.crop(x1, y1, x2 - x1, y2 - y1)
    else:
        ax = round(img.width * float(x1) / owidth)
        ay = round(img.height * float(y1) / oheight)
        aw = round(img.width * float(x2 - x1) / owidth)
        ah = round(img.height * float(y2 - y1) / oheight)
        return img.crop(ax, ay, aw, ah)

def with_boundingboxes(imgid, img, bbox_dir):
    wnid, imgnum = imgid.split("_")
    try:
        if bbox_dir is None:
            raise IOError, "No bounding box dir."
        f = open(os.path.join(bbox_dir, wnid, imgid + ".xml"))
        bbox_root = etree.parse(f)
        osizes = xml_to_objects(bbox_root.xpath("/annotation/size")[0])
        objects = map(xml_to_objects, bbox_root.xpath("/annotation/object"))
        for i, o in enumerate(objects):
            cropped = relcrop(img, osizes.width, osizes.height,
                                   o.bndbox.xmin, o.bndbox.ymin,
                                   o.bndbox.xmax, o.bndbox.ymax)
            if cropped is None:
                continue
            yield o.name, imgid + "_" + str(i), cropped

    except IOError:
        yield wnid, imgid + "_0", img

def extract_hue(scv_img, nbins=128):
    hhfe = scv.HueHistogramFeatureExtractor(nbins)
    yield hhfe.extract(scv_img)

def extract_intensity(scv_img, nbins=128):
    arr = np.array(scv_img.histogram(numbins=nbins))
    yield arr / float(arr.sum())

def extract_surf(scv_img):
    iw, ih = float(scv_img.width), float(scv_img.height)
    try:
        keypoints = scv_img.findKeypoints(highQuality=True, flavor='SURF')
    except TypeError:
        return []
    if keypoints:
        return (((kp.x / iw, kp.y / ih), kp.descriptor()) for kp in keypoints if kp)
    else:
        return []

def extract_xyz(scv_img, nbins=128):
    xyz = scv_img.toXYZ()
    mat = xyz.getNumpy()
    x = np.histogram(mat[:,:,0], normed=True, bins=nbins, range=(0, 255))[0]
    y = np.histogram(mat[:,:,1], normed=True, bins=nbins, range=(0, 255))[0]
    z = np.histogram(mat[:,:,2], normed=True, bins=nbins, range=(0, 255))[0]
    return [np.array([x, y, z]).flatten()]


def extract_sift(scv_img):
    iw, ih = float(scv_img.width), float(scv_img.height)
    try:
        keypoints = scv_img.findKeypoints(flavor='SIFT')
    except TypeError:
        return []
    if keypoints:
        return (((kp.x / iw, kp.y / ih), kp.descriptor()) for kp in keypoints if kp)
    else:
        return []

COORD_FEATS = ('surf', 'sift')
SPATIAL = 4

def main():
    parser = argparse.ArgumentParser(
                description='Extracts CV features from a tarball of images.')
    parser.add_argument('--tarball', '-t', metavar="FILE", 
                        help='The input tarball of images.', default="-")
    parser.add_argument('--feature', '-f', choices=('surf', 'hue', 'intensity', 'sift', 'xyz'),
                        help='The type of features to extract.')
    parser.add_argument('--whitelist', '-w', metavar='FILE',
                        help='A whitelist of ImageNet IDs to accept.')
    parser.add_argument('--bbox', '-b', metavar='DIR',
                        help='The directory containing the bounding boxes.')
    parser.add_argument('--out', '-o', metavar='FILE', default="-",
                        help='The output file of features.')
    parser.add_argument('--no-bzip2', '-J', action='store_true',
                        help="Don't bzip2 the output.")
    parser.add_argument('--prob', '-p', default=1.0, type=float, metavar='FLOAT',
                        help="Sample features with given probability.")
    parser.add_argument('--clusters', '-c', metavar='FILE',
                        help="Output features in terms of clusters (BoVW).")
    args = parser.parse_args()

    if args.feature == 'surf':
        extractor = extract_surf
    elif args.feature == 'hue':
        extractor = extract_hue
    elif args.feature == 'xyz':
        extractor = extract_xyz
    elif args.feature == 'intensity':
        extractor = extract_intensity
    elif args.feature == 'sift':
        extractor = extract_sift
    else:
        raise NotImplementedError, "Can't extract feature %s yet." % args.feature

    if args.out == "-":
        ofile = sys.stdout
    else:
        ofile = open(args.out, 'w')

    if not args.no_bzip2:
        ofile = bz2file.BZ2File(ofile, 'w')

    if args.whitelist:
        whitelist = set(open(args.whitelist).read().split('\n'))
    else:
        whitelist = None

    if args.clusters:
        with open(args.clusters) as clusters_f:
            clusters_t = [l for l in clusters_f.read().split("\n") if l]
            clusters = np.array([np.array(map(float, l.split())) for l in clusters_t])
    else:
        clusters = None


    image_files = yield_imagefiles(args.tarball)
    for filename, image_file in image_files:
        imgid = filename[:filename.index('.')]
        #sys.stderr.write("Processing '%s' (%s).\n" % (filename, imgid))
        try:
            img = image_from_imagefile(image_file)
        except IOError, ioerr:
            #sys.stderr.write("Shit, forced to skip %s (%s).\n" % (filename, ioerr))
            continue
        cropped_images = with_boundingboxes(imgid, img, args.bbox)
        for wnid, cropid, cimg in cropped_images:
            if whitelist and wnid not in whitelist:
                continue
            if clusters is not None:
                current_bovw = np.array([0] * len(clusters) * SPATIAL * SPATIAL)
            for feature in extractor(cimg):
                if args.feature in COORD_FEATS:
                    ((xp, yp), feature) = feature
                else:
                    xp, yp = 0.0, 0.0

                if random() > args.prob:
                    continue

                if clusters is None:
                    fstr = " ".join(map(repr, feature))
                    if args.feature in COORD_FEATS:
                        ofile.write("%s\t%s\t%f,%f\t%s\n" % (wnid, cropid, xp, yp, fstr))
                    else:
                        ofile.write("%s\t%s\t%s\n" % (wnid, cropid, fstr))
                else:
                    offset = len(clusters) * (math.floor(xp * SPATIAL) * SPATIAL + math.floor(yp * SPATIAL))
                    cluster_num = ((clusters - feature) ** 2).sum(axis=1).argmin()
                    current_bovw[cluster_num + offset] += 1

            if clusters is not None:
                fstr = " ".join(map(repr, current_bovw))
                ofile.write("%s\t%s\t%s\n" % (wnid, cropid, fstr))
        #sys.stderr.write("finished with '%s' (%s).\n" % (filename, imgid))

    ofile.flush()
    ofile.close()



if __name__ == '__main__':
    main()


