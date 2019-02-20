#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -pe sharedmem 16
#$ -cwd

# Script to submit a tiling job to the Eddie job scheduler.
# The script takes a directory which contains directories of .svs slides and tiles 
# the .svs slides in parallel
# Usage (on eddie):
# $ qsub tile_job.sh /dir/to/data

set -e

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load environment
module load anaconda
source activate honours

SCRIPTSDIR="/home/s1547426/dev/honours/preprocessing"

DATADIR="$1"
NUM_CORES=16
TILESIZE=512

find $DATADIR -mindepth 1 -maxdepth 1 -type d | \
    parallel -j $NUM_CORES ${SCRIPTSDIR}/tile_batch.sh {} $TILESIZE