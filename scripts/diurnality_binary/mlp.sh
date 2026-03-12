#!/bin/bash

#OAR -l walltime=00:30:00
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

python -m src.main --config-name diurnality_binary data=point_cloud_flat model=mlp training=mlp
python -m src.main --config-name diurnality_binary data=point_cloud_flat model=mlp training=mlp split=random

echo "Experiment completed!"