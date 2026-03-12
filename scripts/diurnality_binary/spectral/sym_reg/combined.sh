#!/bin/bash

#OAR -l walltime=00:30:00
#OAR -O experiments_%jobid%.out
#OAR -E experiments_%jobid%.err
#OAR -n diurnality_pn

# Display resource info
hostname
nvidia-smi

# Activate environment
module load conda
conda activate pt3d

python -m src.main --config-name diurnality_binary data=point_cloud_spectral model=sym_reg data.nfeatures=12 task.input=singular_values_vectors
python -m src.main --config-name diurnality_binary data=point_cloud_spectral model=sym_reg data.nfeatures=12 split=random task.input=singular_values_vectors

echo "Experiment completed!"