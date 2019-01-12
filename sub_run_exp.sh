#!/bin/sh
# Grid engine options (lines prefixed with #$)
#$ -cwd
#$ -pe gpu 1
#$ -l h_vmem=16G

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load environment
module load anaconda
source activate honours

./run_exp.py --config config.yaml