#!/bin/bash

BFCONVERT="/Users/Tim/dev/other/bftools/bfconvert"

TILESIZE=$1
INPUTFILE=$2

OUTPUT_PREFIX="${INPUTFILE%.*}"
DONEFILE="${OUTPUT_PREFIX}.done"

SERIES=2

if [ -f ${DONEFILE} ]; then
    exit 0
fi

X_LOC=0
Y_LOC=0

PREVIOUS_EXIT_CODE=0
while true; do
    OUTPUTFILE="${OUTPUT_PREFIX}_${X_LOC}_${Y_LOC}_${TILESIZE}x${TILESIZE}.tiff"
    ${BFCONVERT} -series ${SERIES} -crop ${X_LOC},${Y_LOC},${TILESIZE},${TILESIZE} \
    ${INPUTFILE} ${OUTPUTFILE}
    if [ $? -ne 0 ]; then
        if [ ${X_LOC} -eq 0 ]; then
            # We reached the end of the y-axis.
            break
        elif [ ${PREVIOUS_EXIT_CODE} -ne 0 ]; then
            >&2 echo "Exit because of two failed attempts in a row"
            exit 1
        fi
        X_LOC=0
        Y_LOC=$((Y_LOC + TILESIZE))
        PREVIOUS_EXIT_CODE=1
    else
        X_LOC=$((X_LOC + TILESIZE))
    fi
done

touch ${DONEFILE}