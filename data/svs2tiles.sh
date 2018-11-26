#!/bin/bash

set -e # Exit directly if any command fails.

BFCONVERT="/Users/Tim/dev/other/bftools/bfconvert"

TILESIZE=$1
INPUTFILE=$2
LOCATIONSFILE=$3

OUTPUT_PREFIX="${INPUTFILE%.*}"
OUTTYPE="tiff"
DONEFILE="${OUTPUT_PREFIX}.done"

SERIES=0

if [ -f ${DONEFILE} ]; then
    exit 0
fi

while IFS='' read -r LINE || [[ -n "$LINE" ]]; do
    X_LOC=$(echo $LINE | cut -d " " -f 1)
    Y_LOC=$(echo $LINE | cut -d " " -f 2)
    OUTPUTFILE="${OUTPUT_PREFIX}_${X_LOC}_${Y_LOC}_${TILESIZE}x${TILESIZE}.${OUTTYPE}"
    ${BFCONVERT} -series ${SERIES} -crop ${X_LOC},${Y_LOC},${TILESIZE},${TILESIZE} \
    ${INPUTFILE} ${OUTPUTFILE} 
done < ${LOCATIONSFILE}

touch ${DONEFILE}