#!/bin/bash

#OAR -l walltime=01:00:00
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

python -m src.main --config-name diurnality_binary data=point_cloud_front model=pointnet_full
python -m src.main --config-name diurnality_binary split=random data=point_cloud_front model=pointnet_full

echo "Experiment completed!"