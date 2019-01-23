#!/usr/bin/env python

import openslide
import numpy as np
import argparse
import sys
import os
import itertools

SERIES = 0
OUTTYPE = "tiff"

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def is_dense_tile(image, location, tiles_info):
    """
    Args:
        image: PIL.image
        location: tuple (x_location, y_location)
        tiles_info: dict
    """
    if location in tiles_info["dense"]:
        return True
    elif location in tiles_info["non-dense"]:
        return False

    if image is not None:
        gray_dens = np.mean(image.convert("L"))
    if gray_dens < 180:
        tiles_info["dense"].add(location)
        return True
    else:
        tiles_info["non-dense"].add(location)
        return False
    
def all_neighbours_dense(neighbours, slide, tilesize, tiles_info):
    any_neighbour_not_dense = any([loc in tiles_info["non-dense"] 
                                   for loc in neighbours])
    if any_neighbour_not_dense:
        return False

    for loc in neighbours:
        if loc in tiles_info["dense"]:
            continue

        t = slide.read_region(loc, SERIES, (tilesize, tilesize))
        if not is_dense_tile(t, loc, tiles_info):
            return False
    
    return True

def get_neighbours(location, tilesize, ixs):
    x_loc, y_loc = location
    neighbours = [
        (x_loc + tilesize, y_loc), # Right neighbour
        (x_loc - tilesize, y_loc), # Left neighbour
        (x_loc, y_loc + tilesize), # Upper neighbour
        (x_loc, y_loc - tilesize)  # Lower neighbour
    ]
    # Finally, check that all neighbours are valid locations
    return filter(lambda x: x in ixs, neighbours)

def make_indixes(tilesize, length):
    ixs = []
    current_ix = 0
    while True:
        if current_ix + tilesize < length:
            ixs.append(current_ix)
            current_ix += tilesize
        else:
            break
    return ixs

def main(infile, tilesize, identifier):
    output_prefix = os.path.splitext(infile)[0]
    donefile = "{}.done".format(output_prefix)
    if os.path.isfile(donefile):
        sys.exit(0)
    
    slide = openslide.OpenSlide(infile)
    width, height = slide.dimensions
    ixs = set(itertools.product(
        make_indixes(tilesize, width), make_indixes(tilesize, height)
    ))

    tiles_info = {
        "dense": set(),
        "non-dense": set()
    }
    for x_loc, y_loc in ixs:
        location = (x_loc, y_loc)
        if location in tiles_info["non-dense"]:
            continue
        tile = slide.read_region(location, SERIES, (tilesize, tilesize))
        if not is_dense_tile(tile, location, tiles_info):
            continue

        neighbours = get_neighbours(location, tilesize, ixs)
        if not all_neighbours_dense(neighbours, slide, tilesize, tiles_info):
            continue
        
        if identifier is None:
            outfile = "{prefix}_{x_loc}_{y_loc}_{tsize}x{tsize}.{type}".format(
                prefix=output_prefix, x_loc=location[0], y_loc=location[1], 
                tsize=tilesize, type=OUTTYPE
            )
        else:
            outfile = "{prefix}_{x_loc}_{y_loc}_{tsize}x{tsize}.{id}.{type}".format(
                prefix=output_prefix, x_loc=location[0], y_loc=location[1], 
                tsize=tilesize, id=identifier, type=OUTTYPE
            )
        tile.save(outfile)
    
    slide.close()
    touch(donefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="Name of the input .svs file.")
    parser.add_argument("--tilesize", help="Size of the tiles. All tiles are square.",
                        type=int)
    parser.add_argument("--identifier")
    args = parser.parse_args()
    main(**vars(args))