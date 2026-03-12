#!/bin/bash

#OAR -l walltime=04:00:00
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

for n in 10 20 30 40 50 60; do
    for loc in front middle back; do
        python -m src.main --config-name diurnality_binary data=point_cloud_hub_$loc data.npoint=$n model=pointnet_full
        python -m src.main --config-name diurnality_binary split=random data=point_cloud_hub_$loc data.npoint=$n model=pointnet_full
    done
done

echo "Experiment completed!"