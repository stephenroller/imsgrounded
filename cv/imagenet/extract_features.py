#!/usr/bin/env

import sys
import bz2file
import tarfile
import SimpleCV as scv
import PIL
import argparse
import os.path
from lxml import etree

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
        return img.crop(x, y, w, h)
    else:
        ax = round(img.width * float(x1) / owidth)
        ay = round(img.height * float(y1) / oheight)
        aw = round(img.width * float(x2 - x1) / owidth)
        ah = round(img.height * float(y2 - y1) / oheight)
        return img.crop(ax, ay, aw, ah)

def with_boundingboxes(imgid, img, bbox_dir):
    wnid, imgnum = imgid.split("_")
    try:
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


def extract_surf(scv_img):
    return (kp.descriptor() for kp in scv_img.findKeypoints(highQuality=True))


def main():
    parser = argparse.ArgumentParser(
                description='Extracts CV features from a tarball of images.')
    parser.add_argument('--tarball', '-t', metavar="FILE", help='The input tarball of images.')
    parser.add_argument('--feature', '-f', choices=('surf', 'color'),
                        help='The type of features to extract.')
    parser.add_argument('--whitelist', '-w', metavar='FILE',
                        help='A whitelist of ImageNet IDs to accept.')
    parser.add_argument('--bbox', '-b', metavar='DIR',
                        help='The directory containing the bounding boxes.')
    parser.add_argument('--out', '-o', metavar='FILE',
                        help='The output file of features.')
    parser.add_argument('--no-bzip2', '-J', action='store_true',
                        help="Don't bzip2 the output.")
    args = parser.parse_args()

    if args.feature == 'surf':
        extractor = extract_surf
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


    image_files = yield_imagefiles(args.tarball)
    for filename, image_file in image_files:
        img = image_from_imagefile(image_file)
        imgid = filename[:filename.index('.')-1]
        cropped_images = with_boundingboxes(imgid, img, args.bbox)
        for wnid, cropid, cimg in cropped_images:
            if whitelist and wnid not in whitelist:
                continue
            for feature in extractor(cimg):
                fstr = " ".join(map(repr, feature))
                ofile.write("%s\t%s\t%s\n" % (wnid, cropid, fstr))
        #ofile.flush()

    ofile.close()



if __name__ == '__main__':
    main()


