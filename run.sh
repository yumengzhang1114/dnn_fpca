#!/bin/bash



#BSUB -W 90

#BSUB -n 4

#BSUB -o out.%J

#BSUB -e err.%J

#BSUB -q gpu

#BSUB -gpu "num=1:mode=shared:mps=no"

#BSUB -R "select[h100]"

source ~/.bashrc

conda activate /usr/local/usrapps/st758f20/cpeng22/conda_envs/pytorch

python main.py



conda deactivate
