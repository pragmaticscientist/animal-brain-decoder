#!/bin/bash

#OAR -l walltime=05:00:00
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

for n in 10 20 40 60 80 100 125 150 175 200; do
    python -m src.main --config-name diurnality_binary data.npoint=$n model=pointnet_full
    python -m src.main --config-name diurnality_binary split=random data.npoint=$n model=pointnet_full
done


echo "Experiment completed!"