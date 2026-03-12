#!/bin/bash

#OAR -l walltime=01:00:00
#OAR -O experiments_%jobid%.out
#OAR -E experiments_%jobid%.err
#OAR -n diurnality_pn

# Display resource info
hostname
nvidia-smi

# Activate environment
module load conda
conda activate pt3d

python -m src.main --config-name diurnality_binary data=point_cloud_spectral data.nfeatures=9 model=mlp_small training=mlp task.input=singular_vectors
python -m src.main --config-name diurnality_binary data=point_cloud_spectral data.nfeatures=9 model=mlp_small training=mlp split=random task.input=singular_vectors

echo "Experiment completed!"