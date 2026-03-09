#!/bin/bash

#OAR -l walltime=4:00:00
#OAR -O experiments_%jobid%.out
#OAR -E experiments_%jobid%.err
#OAR -n diurnality_mlp

# Display resource info
hostname
nvidia-smi

# Activate environment
module load conda
conda activate pt3d

python -m src.main --config config/diurnality/mlp.yaml

echo "Experiment completed!"