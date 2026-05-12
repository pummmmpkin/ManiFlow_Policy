#!/bin/bash

set -euo pipefail

dataset_name=${1:-three_piece_assembly_d2}
seed=${2:-0}
gpu_id=${3:-4}

DEBUG=${DEBUG:-False}
save_ckpt=${SAVE_CKPT:-True}
checkpoint_every=${CHECKPOINT_EVERY:-10}
training_epochs=${TRAINING_EPOCHS:-100}

base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${base_path}"
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate unisim
export PYTHONPATH=${PWD}:${PYTHONPATH:-}
export PROJECT_DIR=${PWD}
export WANDB_API_KEY=c9187c7dfcc339af75f2f47c3b80c95743057b42

cuda_ids=${gpu_id//_/,}
num_gpus=$(echo "${cuda_ids}" | tr ',' ' ' | wc -w)
export CUDA_VISIBLE_DEVICES="${cuda_ids}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

use_point_cloud_rgb=false
pc_channels=3
batch_size=128
val_batch_size=128
dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'goal_gripper_pcd']"
point_cloud_shape="[4500,3]"
if [[ "${dataset_name}" == *"_color" ]]; then
    use_point_cloud_rgb=true
    pc_channels=6
    batch_size=64
    val_batch_size=64
    dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'goal_gripper_pcd', 'rgb_values']"
    point_cloud_shape="[4500,6]"
fi

pretrain_ckpt="/project_data/held/chenyuah/ManiFlow_Policy/ckpt/baseline_maniflow_full_new/checkpoints/epoch-96.ckpt"
fixed_cat_idx="${FIXED_CAT_IDX:-0}"
exp_name="mimicgen_${dataset_name}_maniflow_finetune_fixedcat${fixed_cat_idx}"
run_dir="/mnt/ManiFlow/ckpt/${exp_name}_seed${seed}"

echo "dataset_name=${dataset_name}"
echo "pretrain_ckpt=${pretrain_ckpt}"
echo "fixed_cat_idx=${fixed_cat_idx}"
echo "use_point_cloud_rgb=${use_point_cloud_rgb}"
echo "run_dir=${run_dir}"

cd ManiFlow/maniflow/workspace
torchrun --standalone --nproc_per_node="${num_gpus}" \
    train_maniflow_robogen_workspace.py \
    --config-name=maniflow_pointcloud_policy_dex.yaml \
    load_policy_path="${pretrain_ckpt}" \
    policy.language_conditioned=True \
    +policy.fixed_cat_idx="${fixed_cat_idx}" \
    training.resume=False \
    training.use_dataset_normalization=0 \
    training.checkpoint_every="${checkpoint_every}" \
    training.num_epochs="${training_epochs}" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${val_batch_size}" \
    task=robogen_open_door \
    task.dataset.zarr_path="${dataset_name}" \
    task.dataset.observation_mode="act3d_goal_mlp" \
    task.dataset.enumerate=True \
    task.dataset.train_ratio=0.9 \
    task.dataset.num_load_episodes=1000 \
    task.dataset.kept_in_disk=true \
    task.dataset.load_per_step=true \
    task.dataset.augmentation_rot=false \
    task.dataset.augmentation_pcd=true \
    task.dataset.use_absolute_waypoint=false \
    task.dataset.is_pickle=false \
    task.dataset.use_point_cloud_rgb="${use_point_cloud_rgb}" \
    task.dataset.dataset_keys="${dataset_keys}" \
    task.env_runner.max_steps=35 \
    task.shape_meta.obs.agent_pos.shape="[10]" \
    task.shape_meta.obs.point_cloud.shape="${point_cloud_shape}" \
    task.shape_meta.action.shape="[10]" \
    task.env_runner.num_point_in_pc=4500 \
    task.env_runner.use_absolute_waypoint=false \
    horizon=8 \
    n_obs_steps=2 \
    policy.use_pc_color="${use_point_cloud_rgb}" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channels}" \
    hydra.run.dir="${run_dir}" \
    training.debug="${DEBUG}" \
    training.seed="${seed}" \
    exp_name="${exp_name}" \
    logging.name="${exp_name}" \
    logging.mode=online \
    checkpoint.save_ckpt="${save_ckpt}"
