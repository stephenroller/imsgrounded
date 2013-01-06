#!/usr/bin/env python

import sys
import bz2file
import tarfile
import SimpleCV as scv
import PIL.ImageFile
import argparse
import os.path
import numpy as np
from lxml import etree
from random import random

#from progress import ProgressBar


def yield_imagefiles(tarfilename):
    tf = tarfile.open(tarfilename)
    for fn in tf.getnames():
        yield fn, tf.extractfile(fn)

def image_from_imagefile(imgfile):
    parser = PIL.ImageFile.Parser()
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

def extract_hue(scv_img, nbins=179):
    bins = [0] * nbins
    for hue, fraction in scv_img.huePeaks(bins=nbins):
        bins[int(hue)] = fraction
    yield bins

def extract_intensity(scv_img, nbins=50):
    arr = np.array(scv_img.histogram(numbins=nbins))
    yield arr / float(arr.sum())

def extract_surf(scv_img):
    keypoints = scv_img.findKeypoints(highQuality=True)
    if keypoints:
        return (kp.descriptor() for kp in keypoints if kp)
    else:
        return []


def main():
    parser = argparse.ArgumentParser(
                description='Extracts CV features from a tarball of images.')
    parser.add_argument('--tarball', '-t', metavar="FILE", help='The input tarball of images.')
    parser.add_argument('--feature', '-f', choices=('surf', 'hue', 'intensity'),
                        help='The type of features to extract.')
    parser.add_argument('--whitelist', '-w', metavar='FILE',
                        help='A whitelist of ImageNet IDs to accept.')
    parser.add_argument('--bbox', '-b', metavar='DIR',
                        help='The directory containing the bounding boxes.')
    parser.add_argument('--out', '-o', metavar='FILE',
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
    elif args.feature == 'intensity':
        extractor = extract_intensity
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
        img = image_from_imagefile(image_file)
        imgid = filename[:filename.index('.')-1]
        cropped_images = with_boundingboxes(imgid, img, args.bbox)
        for wnid, cropid, cimg in cropped_images:
            if whitelist and wnid not in whitelist:
                continue
            if clusters is not None:
                current_bovw = np.array([0] * len(clusters))
            for feature in extractor(cimg):
                if random() > args.prob:
                    continue
                if clusters is None:
                    fstr = " ".join(map(repr, feature))
                    ofile.write("%s\t%s\t%s\n" % (wnid, cropid, fstr))
                else:
                    cluster_num = ((clusters - feature) ** 2).sum(axis=1).argmin()
                    current_bovw[cluster_num] += 1

            if clusters is not None:
                fstr = " ".join(map(repr, current_bovw))
                ofile.write("%s\t%s\n" % (cropid, fstr))

    ofile.close()



if __name__ == '__main__':
    main()


