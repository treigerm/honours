#!/bin/sh
# Grid engine options (lines prefixed with #$)
#$ -cwd
#$ -l h_vmem=16GB

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load environment
module load anaconda
source activate honours

./run_exp.py --config config.yaml