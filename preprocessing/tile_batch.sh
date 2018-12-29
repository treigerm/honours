#!/bin/bash

set -e

SCRIPTSDIR="$HOME/dev/cw/honours/data"

# Directory that .svs file is located.
IMAGEDIR=$1
TILESIZE=$2

DONEFILE="${IMAGEDIR}/tiles.done"

if [ -f ${DONEFILE} ]; then
    echo "Skipping ${IMAGEDIR}"
    exit 0
fi

for FILE in ${IMAGEDIR}/*.svs; do
    OUTPUT_PREFIX="${FILE%.*}"
    TILESFILE="${OUTPUT_PREFIX}.tiles.txt"
    TILESDONE="${OUTPUT_PREFIX}.tiles.done"
    if [ ! -f ${TILESDONE} ]; then
        echo "Finding densest tiles for ${FILE}"
        ${SCRIPTSDIR}/find_dense_tiles.py --infile $FILE --tilesize ${TILESIZE} > ${TILESFILE}
        touch ${TILESDONE}
    fi
    ${SCRIPTSDIR}/svs2tiles.py --tilesize ${TILESIZE} --infile ${FILE} < ${TILESFILE}
done

touch ${DONEFILE}