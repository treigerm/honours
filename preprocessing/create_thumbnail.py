#!/usr/bin/env python

import os
import openslide
import argparse

def main(infile):
    slide = openslide.OpenSlide(infile)
    thumbnail_dims = [int(x * 0.1) for x in slide.dimensions]
    thumbnail = slide.get_thumbnail(thumbnail_dims)

    fileprefix = os.path.splitext(infile)[0]
    outfile = "{}.jpg".format(fileprefix)
    thumbnail.save(outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    args = parser.parse_args()
    main(**vars(args))