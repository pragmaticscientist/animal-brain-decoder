#!/bin/bash

#OAR -l walltime=1:00:00
#OAR -p cluster=musa
#OAR -O experiments_%jobid%.out
#OAR -E experiments_%jobid%.err
#OAR -n diurnality_pn

# Display resource info
hostname
nvidia-smi

# Activate environment
module load conda
conda activate pt3d

python -m src.main --config config/diurnality_binary/pointnet.yaml

echo "Experiment completed!"