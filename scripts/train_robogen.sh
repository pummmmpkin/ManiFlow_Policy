
# Usage examples:
# bash scripts/train_eval_dex.sh maniflow_pointcloud_policy_dex adroit_door_pointcloud pointcloud_debug 0 1
# bash scripts/train_eval_dex.sh maniflow_image_timm_policy_dex adroit_door_image image_debug 0 1
# bash scripts/train_eval_dex.sh maniflow_pointcloud_policy_dex dexart_laptop_pointcloud pointcloud_debug 0 1
# bash scripts/train_eval_dex.sh maniflow_image_timm_policy_dex dexart_laptop_image image_debug 0 1

DEBUG=False
save_ckpt=True
checkpoint_every=10
batch_size=256
train=True
eval=False # set to false since mostly we do online eval for adroit/dexart tasks

alg_name=${1}
task_name=${2}
zarr_path=${3}
addition_info=${4}
seed=${5}
gpu_id=${6}

# Process task name (remove _image or _pointcloud suffix)
processed_task_name=${task_name}
if [[ $task_name == *"_image"* ]]; then
    processed_task_name=${task_name//_image/}
elif [[ $task_name == *"_pointcloud"* ]]; then
    processed_task_name=${task_name//_pointcloud/}
fi

# Example dataset paths:
# dataset_path=data/adroit_hammer_expert.zarr
# dataset_path=data/adroit_door_expert.zarr
# dataset_path=data/adroit_pen_expert.zarr

# dataset_path=data/dexart_faucet_expert.zarr
# dataset_path=data/dexart_laptop_expert.zarr
# dataset_path=data/dexart_bucket_expert.zarr
# dataset_path=data/dexart_toilet_expert.zarr

# dataset_path=data/metaworld_basketball_expert.zarr
# dataset_path=data/metaworld_assembly_expert.zarr


# Setup paths and configuration
base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# dataset_path=data/${processed_task_name}_expert.zarr # e.g., data/adroit_door_expert.zarr, data/dexart_faucet_expert.zarr, data/metaworld_basketball_expert.zarr
exp_name=${task_name}-${alg_name}-${addition_info}
gcs_path="gs://cmu-gpucloud-chenyuah/ManiFlow/${exp_name}_seed${seed}"
run_dir="/tmp/ManiFlow/data/${exp_name}_seed${seed}"
config_name=${alg_name}

if gcloud storage ls "${gcs_path}" >/dev/null 2>&1; then
    echo "[Found] ${gcs_path}, start syncing..."
    mkdir -p "${run_dir}"
    gcloud storage rsync -r "${gcs_path}" "${run_dir}"
else
    echo "[Not Found] ${gcs_path}, skip."
fi

# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}


# Set wandb mode based on debug flag
if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33m=== DEBUG MODE ===\033[0m"
else
    wandb_mode=online
    echo -e "\033[33m=== TRAINING MODE ===\033[0m"
fi


# Print configuration
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
echo -e "\033[33mTrain: ${train}, Eval: ${eval}\033[0m"


# Change to workspace directory
cd ManiFlow/maniflow/workspace
observation_mode="act3d_goal_mlp"
encoding_mode="keep_position_feature_in_attention_feature"
use_absolute_waypoint=false
pointcloud_num=4500
horizon=8
n_obs_steps=2 # 2 or 4
agent_pos_dim=10
action_dim=10
augmentation_rot=false
augmentation_pcd=true
is_pickle=true
num_load_episodes=1000    
train_ratio=0.9 
# Training phase
if [ $train = True ]; then
    echo -e "\033[32m=== Starting Training ===\033[0m"
    python train_maniflow_robogen_workspace.py \
        --config-name=${config_name}.yaml \
        policy.language_conditioned=True \
        training.checkpoint_every=${checkpoint_every} \
        dataloader.batch_size="${batch_size}" \
        task=${task_name} \
        task.dataset.zarr_path=${zarr_path} \
        task.env_runner.demo_experiment_path="[]" \
        task.env_runner.experiment_name="[]" \
        task.env_runner.experiment_folder="[]" \
        task.env_runner.num_point_in_pc="${pointcloud_num}" \
        task.env_runner.use_absolute_waypoint="${use_absolute_waypoint}" \
        horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
        task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
        task.shape_meta.action.shape="[${action_dim}]" \
        task.dataset.observation_mode="${observation_mode}" \
        task.dataset.enumerate=True \
        task.env_runner.max_steps=35 \
        task.dataset.train_ratio="${train_ratio}" \
        task.dataset.num_load_episodes=${num_load_episodes} \
        task.dataset.kept_in_disk=true \
        task.dataset.load_per_step=true \
        task.dataset.augmentation_rot="${augmentation_rot}" \
        task.dataset.augmentation_pcd="${augmentation_pcd}" \
        task.dataset.use_absolute_waypoint="${use_absolute_waypoint}" \
        task.dataset.is_pickle="${is_pickle}" \
        task.dataset.dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']" \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt}
    
    if [ $? -eq 0 ]; then
        echo -e "\033[32m=== Training completed successfully ===\033[0m"
    else
        echo -e "\033[31m=== Training failed ===\033[0m"
        exit 1
    fi
else
    echo -e "\033[33m=== Skipping Training ===\033[0m"
fi

# # Evaluation phase
# if [ $eval = True ]; then
#     echo -e "\033[32m=== Starting Evaluation ===\033[0m"
#     python eval_maniflow_dex_workspace.py \
#         --config-name=${config_name}.yaml \
#         task=${task_name} \
#         task.dataset.zarr_path=${zarr_path} \
#         hydra.run.dir=${run_dir} \
#         training.debug=$DEBUG \
#         training.seed=${seed} \
#         training.device="cuda:0" \
#         exp_name=${exp_name} \
#         logging.mode=${wandb_mode} \
#         checkpoint.save_ckpt=${save_ckpt}

    
#     if [ $? -eq 0 ]; then
#         echo -e "\033[32m=== Evaluation completed successfully ===\033[0m"
#     else
#         echo -e "\033[31m=== Evaluation failed ===\033[0m"
#         exit 1
#     fi
# else
#     echo -e "\033[33m=== Skipping Evaluation ===\033[0m"
# fi

# echo -e "\033[32m=== Script completed ===\033[0m"
# # Usage examples:
# # bash scripts/train_eval_robotwin.sh maniflow_pointcloud_policy_robotwin pick_apple_messy_pointcloud 0825 0 1
# # bash scripts/train_eval_robotwin.sh maniflow_pointcloud_policy_robotwin diverse_bottles_pick_pointcloud 0825 1 1
# # bash scripts/train_eval_robotwin.sh maniflow_image_timm_policy_robotwin pick_apple_messy_image image_debug 0 1
# # bash scripts/train_eval_robotwin.sh maniflow_image_transformer_policy_robotwin pick_apple_messy_image image_debug 0 1


# DEBUG=False
# save_ckpt=True
# train=True
# eval=True

# alg_name=${1}
# task_name=${2}
# addition_info=${3}
# seed=${4}
# gpu_id=${5}

# # Training/Evaluation parameters
# eval_episode=100
# eval_mode="latest"  # "best" or "latest"
# num_inference_steps=10
# n_obs_steps=2
# horizon=16
# n_action_steps=16\


# # Validate required arguments
# if [[ -z "$alg_name" || -z "$task_name" || -z "$addition_info" || -z "$seed" || -z "$gpu_id" ]]; then
#     echo "Usage: $0 <alg_name> <task_name> <addition_info> <seed> <gpu_id>"
#     echo "Example: $0 maniflow_pointcloud_policy_robotwin pick_apple_messy_pointcloud pointcloud_debug 0 1"
#     exit 1
# fi

# # Process task name (remove _image or _pointcloud suffix)
# processed_task_name=${task_name}
# if [[ $task_name == *"_image"* ]]; then
#     processed_task_name=${task_name//_image/}
# elif [[ $task_name == *"_pointcloud"* ]]; then
#     processed_task_name=${task_name//_pointcloud/}
# fi

# # Setup paths and configuration
# base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# task_zarr_name=${processed_task_name}
# zarr_path="${base_path}/ManiFlow/data/${task_zarr_name}.zarr"
# exp_name=${processed_task_name}-${alg_name}-${addition_info}
# # run_dir="/gscratch/scrubbed/geyan/projects/ManiFlow_Policy/ManiFlow/data/outputs/${exp_name}_seed${seed}"
# run_dir="${base_path}/ManiFlow/data/outputs/${exp_name}_seed${seed}"
# config_name=${alg_name}


# # Environment setup
# # Environment setup
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
# export TOKENIZERS_PARALLELISM=false
# export HYDRA_FULL_ERROR=1
# export CUDA_VISIBLE_DEVICES=${gpu_id}


# # Set wandb mode based on debug flag
# if [ $DEBUG = True ]; then
#     wandb_mode=offline
#     echo -e "\033[33m=== DEBUG MODE ===\033[0m"
# else
#     wandb_mode=online
#     echo -e "\033[33m=== TRAINING MODE ===\033[0m"
# fi


# # Print configuration
# echo -e "\033[33mTask: ${processed_task_name}\033[0m"
# echo -e "\033[33mGPU ID: ${gpu_id}\033[0m"
# echo -e "\033[33mTrain: ${train}, Eval: ${eval}\033[0m"


# # Change to workspace directory
# cd ManiFlow/maniflow/workspace

# observation_mode="act3d_goal_mlp"
# encoding_mode="keep_position_feature_in_attention_feature"
# use_absolute_waypoint=false
# pointcloud_num=4500
# horizon=8
# n_obs_steps=2 # 2 or 4
# agent_pos_dim=10
# action_dim=10
# augmentation_rot=false
# augmentation_pcd=true
# is_pickle=true
# num_load_episodes=1000    
# train_ratio=0.9 
# # Training phase
# if [ $train = True ]; then
#     echo -e "\033[32m=== Starting Training ===\033[0m"
#     python train_maniflow_robotwin_workspace.py \
#         --config-name=${config_name}.yaml \
#         task=${task_name} \
#         hydra.run.dir=${run_dir} \
#         training.debug=$DEBUG \
#         training.seed=${seed} \
#         training.device="cuda:0" \
#         exp_name=${exp_name} \
#         logging.mode=${wandb_mode} \
#         checkpoint.save_ckpt=${save_ckpt} \
#         task.dataset.zarr_path=test \
#         task.env_runner.demo_experiment_path="[]" \
#         task.env_runner.experiment_name="[]" \
#         task.env_runner.experiment_folder="[]" \
#         task.env_runner.num_point_in_pc="${pointcloud_num}" \
#         task.env_runner.use_absolute_waypoint="${use_absolute_waypoint}" \
#         horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
#         task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
#         task.shape_meta.action.shape="[${action_dim}]" \
#         task.dataset.observation_mode="${observation_mode}" \
#         task.dataset.enumerate=True \
#         task.env_runner.max_steps=35 \
#         task.dataset.train_ratio="${train_ratio}" \
#         task.dataset.num_load_episodes=${num_load_episodes} \
#         task.dataset.kept_in_disk=true \
#         task.dataset.load_per_step=true \
#         task.dataset.augmentation_rot="${augmentation_rot}" \
#         task.dataset.augmentation_pcd="${augmentation_pcd}" \
#         task.dataset.use_absolute_waypoint="${use_absolute_waypoint}" \
#         task.dataset.is_pickle="${is_pickle}" \
#         task.dataset.dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']" \
    
#     if [ $? -eq 0 ]; then
#         echo -e "\033[32m=== Training completed successfully ===\033[0m"
#     else
#         echo -e "\033[31m=== Training failed ===\033[0m"
#         exit 1
#     fi
# else
#     echo -e "\033[33m=== Skipping Training ===\033[0m"
# fi

# # # Evaluation phase
# # if [ $eval = True ]; then
# #     echo -e "\033[32m=== Starting Evaluation ===\033[0m"
# #     python eval_maniflow_robotwin_workspace.py \
# #         --config-name=${config_name}.yaml \
# #         +eval_mode=${eval_mode} \
# #         robotwin_task=${task_name} \
# #         robotwin_task.env_runner.eval_episodes=${eval_episode} \
# #         hydra.run.dir=${run_dir} \
# #         training.debug=$DEBUG \
# #         training.seed=${seed} \
# #         training.device="cuda:0" \
# #         policy.num_inference_steps=${num_inference_steps} \
# #         policy.n_obs_steps=${n_obs_steps} \
# #         policy.horizon=${horizon} \
# #         policy.n_action_steps=${n_action_steps} \
# #         exp_name=${exp_name} \
# #         logging.mode=${wandb_mode} \
# #         checkpoint.save_ckpt=${save_ckpt}
    
# #     if [ $? -eq 0 ]; then
# #         echo -e "\033[32m=== Evaluation completed successfully ===\033[0m"
# #     else
# #         echo -e "\033[31m=== Evaluation failed ===\033[0m"
# #         exit 1
# #     fi
# # else
# #     echo -e "\033[33m=== Skipping Evaluation ===\033[0m"
# # fi

# # echo -e "\033[32m=== Script completed ===\033[0m"