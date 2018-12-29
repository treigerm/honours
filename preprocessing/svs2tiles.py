#!/usr/bin/env python

import openslide
import argparse
import sys
import os

SERIES = 0
OUTTYPE = "tiff"

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def main(infile, tilesize):
    output_prefix = os.path.splitext(infile)[0]
    donefile = "{}.done".format(output_prefix)
    if os.path.isfile(donefile):
        sys.exit(0)
    
    slide = openslide.OpenSlide(infile)
    for line in sys.stdin:
        location = list(map(int, line.strip().split("\t")))
        tile = slide.read_region(location, SERIES, (tilesize, tilesize))
        outfile = "{prefix}_{x_loc}_{y_loc}_{tsize}x{tsize}.{type}".format(
            prefix=output_prefix, x_loc=location[0], y_loc=location[1], 
            tsize=tilesize, type=OUTTYPE
        )
        tile.save(outfile)
    
    slide.close()
    touch(donefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="Name of the input .svs file.")
    parser.add_argument("--tilesize", help="Size of the tiles. All tiles are square.",
                        type=int)
    args = parser.parse_args()
    main(**vars(args))