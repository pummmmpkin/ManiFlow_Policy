#!/bin/bash

#SBATCH -N 1 # Number of nodes
#SBATCH -n 20
#SBATCH -A dheld
#SBATCH -p preempt 
#SBATCH --qos=preempt_qos
#SBATCH -J maniflow_1024
#SBATCH --gpus=4
#SBATCH -t 48:00:00 # Estimated time
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=chenyuah@andrew.cmu.edu

source "/home/chenyuah/miniconda3/etc/profile.d/conda.sh"
conda activate maniflow
cd /project/flame/chenyuah/ManiFlow_Policy
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export OMP_NUM_THREADS=2    # often helps dataloaders
# export MKL_NUM_THREADS=2

bash /project/flame/chenyuah/RoboGen-sim2real/orchard_scripts/download.sh 165-obj
bash scripts/train_robogen.sh maniflow_pointcloud_policy robogen_open_door open_only 0922_open_1024 0 0_1_2_3
