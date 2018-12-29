#!/bin/bash

set -e

MATLAB="/Applications/MATLAB_R2016b.app/bin/matlab"
SCRIPTDIR="/Users/Tim/dev/cw/honours/data" # Location of stain_normalization.m
TOOLBOXDIR="/Users/Tim/Documents/MATLAB/stain_normalisation_toolbox" # Location of stain normalisation toolbox

SOURCEFILE=$1
REFERENCEFILE=$2
OUTFILE=$3

CURDIR=$PWD
# Why MATLAB? Why?
${MATLAB} -nodisplay -nosplash -nodesktop -r "addpath ${TOOLBOXDIR}; addpath ${SCRIPTDIR}; cd ${TOOLBOXDIR}; install; cd ${CURDIR}; stain_normalization('${SOURCEFILE}', '${REFERENCEFILE}', '${OUTFILE}')"