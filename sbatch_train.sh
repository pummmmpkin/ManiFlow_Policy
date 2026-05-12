#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=60G

# use the bash shell
set -x 
# echo each command to standard out before running it


# echo "Starting job $SLURM_JOB_ID"
# singularity exec --bind /data/chenyuah/RoboGen-sim2real:/mnt/RoboGen_sim2real/ --nv /data/ziyuw2/robogen-dp3-act3d.sif /mnt/RoboGen_sim2real/scripts/eval.sh ${1} ${2}

singularity exec \
  --bind /project_data/held/chenyuah/RoboGen-sim2real:/mnt/RoboGen_sim2real/ \
  --bind /project_data/held/chenyuah/ManiFlow_Policy:/mnt/ManiFlow/ \
  --nv /project_data/held/chenyuah/maniflow_sandbox_0113.sif \
  /mnt/ManiFlow/orchard_train.sh