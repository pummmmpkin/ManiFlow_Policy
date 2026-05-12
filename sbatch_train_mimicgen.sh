#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=60G

set -x

dataset_name=${1:-three_piece_assembly_d2}
seed=${2:-0}
gpu_id=${3:-1}

echo "Starting job ${SLURM_JOB_ID:-local}"
singularity exec \
  --bind /project_data/held/chenyuah/RoboGen-sim2real:/mnt/RoboGen_sim2real/ \
  --bind /project_data/held/chenyuah/ManiFlow_Policy:/mnt/ManiFlow/ \
  --nv /project_data/held/chenyuah/maniflow_sandbox_0113.sif \
  /mnt/ManiFlow/scripts/train_mimicgen.sh "${dataset_name}" "${seed}" "${gpu_id}"
