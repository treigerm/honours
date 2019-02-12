#!/bin/bash

set -e

SCRIPTSDIR="$HOME/dev/cw/honours/preprocessing"

# Directory that .svs file is located.
IMAGEDIR=$1
TILESIZE=$2

DONEFILE="${IMAGEDIR}/tiles.done"

if [ -f ${DONEFILE} ]; then
    echo "Skipping ${IMAGEDIR}"
    exit 0
fi

for FILE in ${IMAGEDIR}/*.svs; do
    ${SCRIPTSDIR}/svs2tiles.py --tilesize ${TILESIZE} --infile ${FILE} \
    --identifier "non_black_and_non_white_0.9_neighbours" --check-neighbours
done

touch ${DONEFILE}