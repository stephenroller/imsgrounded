#!/usr/bin/env python

import sys
import urllib
import itertools
import time
import pygame
import urllib2
import math

from SimpleCV import Image, DrawingLayer, Display, Color
from translate import quick_translate

class Skip(Exception):
    pass

class BadImage(Exception):
    pass


def read_urls(filename):
    f = open(filename)
    for line in f:
        line = line.strip()
        keyword, url, filename = line.split("\t")
        yield (keyword, url, filename)

def scale(ratio, point):
    return (point[0] * ratio, point[1] * ratio)

def get_bounding_box(keyword, url, filename):
    # get the image
    img = Image(url)

    # resize the image so things aren't so slow, if necessary
    w, h = img.size()
    if w > 1200 or h > 1200:
        maxdim = max(w, h)
        ratio = math.ceil(maxdim/800.0)
        print "   resizing..."
        img = img.resize(w=int(w/ratio), h=int(h/ratio))
    else:
        ratio = 1

    # get the canvas
    disp = Display((800, 800))
    # text overlay
    textlayer = DrawingLayer(img.size())
    textlayer.setFontSize(30)
    cx, cy = 10, 10
    for xoff in range(-2, 3):
        for yoff in range(-2, 3):
            textlayer.text(keyword, (cx + xoff, cy + yoff), color=Color.BLACK)
    textlayer.text(keyword, (cx, cy), color=Color.WHITE)

    # two points to declare a bounding box
    point1 = None
    point2 = None
    while disp.isNotDone():
        cursor = (disp.mouseX, disp.mouseY)
        if disp.leftButtonUp:
            if point1 and point2:
                point1 = None
                point2 = None
            if point1:
                point2 = disp.leftButtonUpPosition()
            else:
                point1 = disp.leftButtonUpPosition()
        bb = None
        if point1 and point2:
            bb = disp.pointsToBoundingBox(point1, point2)
        elif point1 and not point2:
            bb = disp.pointsToBoundingBox(point1, cursor)

        img.clearLayers()
        drawlayer = DrawingLayer(img.size())
        if bb:
            drawlayer.rectangle((bb[0], bb[1]), (bb[2], bb[3]), color=Color.RED)

        # keyboard commands
        if pygame.key.get_pressed()[pygame.K_s]:
            # skip for now
            raise Skip()
        elif pygame.key.get_pressed()[pygame.K_b]:
            # mark it as an invalid picture
            raise BadImage()
        elif pygame.key.get_pressed()[pygame.K_RETURN]:
            if point1 and point2:
                bb = disp.pointsToBoundingBox(scale(ratio, point1), scale(ratio, point2))
                return bb
            elif not point1 and not point2:
                bb = disp.pointsToBoundingBox((0, 0), (w, h))
                return bb


        drawlayer.line((cursor[0], 0), (cursor[0], img.height), color=Color.BLUE)
        drawlayer.line((0, cursor[1]), (img.width, cursor[1]), color=Color.BLUE)
        #drawlayer.circle(cursor, 2, color=Color.BLUE, filled=True)
        img.addDrawingLayer(textlayer)
        img.addDrawingLayer(drawlayer)
        img.save(disp)

def read_bounding_box_file(filename):
    bounding_boxes = {}
    f = open(filename)
    for line in f:
        fields = line.strip().split('\t')
        keyword, url, filename, bb = fields
        bounding_boxes['\t'.join([keyword, url, filename])] = bb
    f.close()
    return bounding_boxes

def save_bounding_boxes(bounding_boxes, bounding_box_file):
    wf = open(bounding_box_file, 'w')
    for key, bb in bounding_boxes.iteritems():
        wf.write(key + '\t' + bb + '\n')

cached_translations = {}
def cached_quick_translate(keyword):
    if keyword in cached_translations:
        return cached_translations[keyword]
    else:
        translation = quick_translate(keyword, 'en', 'de')
        cached_translations[keyword] = translation
        return translation


def main(filename, bounding_box_file):
    bounding_boxes = read_bounding_box_file(bounding_box_file)
    urls = list(read_urls(filename))
    for keyword, url, filename in urls:
        translation = cached_quick_translate(keyword)
        key = '\t'.join([keyword, url, filename])
        if key in bounding_boxes:
            continue
        try:
            nice_keyword = "%s - %s" % (keyword.decode("utf-8"), translation)
            print u"keyword: %s" % nice_keyword
            print "   Loading image... (%s)" % filename
            bounding_box = get_bounding_box(nice_keyword, url, filename)
            bounding_boxes[key] = str(bounding_box)
            save_bounding_boxes(bounding_boxes, bounding_box_file)
        except BadImage:
            print "   Marking as bad."
            bounding_boxes[key] = 'bad'
            save_bounding_boxes(bounding_boxes, bounding_box_file)
        except (Skip, urllib2.URLError, IOError):
            print "   Skipping..."
            bounding_boxes[key] = 'skip'
            save_bounding_boxes(bounding_boxes, bounding_box_file)
        except KeyboardInterrupt:
            break
        #except Exception, e:
        #    # just ignore most exceptions. bad practice but easiest.
        #    print "   caught exception (%s)" % e
        #    continue


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

