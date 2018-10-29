#!/usr/bin/env python

import openslide
import numpy as np
import argparse

SERIES = 0

def non_white_density(image):
    """
    Args:
        image: np.array (tile_size, tile_size, channels)
    """
    c1_dens = image[:,:,0] < 200
    c2_dens = image[:,:,1] < 200
    c3_dens = image[:,:,2] < 200
    c1_c2_dens = np.logical_and(c1_dens, c2_dens)
    pix_dens = np.logical_and(c1_c2_dens, c3_dens)
    return np.sum(pix_dens) / float(pix_dens.size)

def main(infile, tilesize, top_n=10):
    slide = openslide.OpenSlide(infile)
    (max_x, max_y) = slide.dimensions
    cur_x, cur_y = 0, 0
    
    tile_densities = []
    while True:
        if cur_y + tilesize > max_y:
            break

        if cur_x + tilesize > max_x:
            cur_x = 0
            cur_y += tilesize
            continue
        
        tile = slide.read_region((cur_x, cur_y), SERIES, (tilesize, tilesize))
        density = non_white_density(np.array(tile))
        tile_densities.append(((cur_x, cur_y), density))
        #print("{}\t{}\t{}".format(cur_x, cur_y, density))
        cur_x += tilesize

    top_n_tiles = sorted(tile_densities, key=lambda x: x[1], reverse=True)[:top_n]
    for (x_loc, y_loc), _ in top_n_tiles:
        print("{}\t{}".format(x_loc, y_loc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="Name of the input file.")
    parser.add_argument("--tilesize", help="Size of the tiles. All tiles are square.",
                        type=int)
    args = parser.parse_args()
    main(**vars(args))
