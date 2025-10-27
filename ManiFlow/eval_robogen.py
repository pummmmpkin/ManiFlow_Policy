import os
import hydra
import torch
from omegaconf import OmegaConf
from maniflow.workspace.train_maniflow_dex_workspace import TrainManiFlowDexWorkspace
from maniflow.common.pytorch_util import dict_apply
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from manipulation.utils import build_up_env, save_numpy_as_gif
import numpy as np
from copy import deepcopy
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from maniflow.gym_util.multistep_wrapper import MultiStepWrapper
import json
import yaml
import argparse
from typing import Optional
from collections import deque
import pybullet as p
from tqdm import tqdm

all_task_names = [
    "open the storage furniture",
    "bucket",
    "faucet",
    "open the folding chair",
    "open the laptop",
    "open the stapler",
    "open the toilet",
    "close the storage furniture",
    "close the folding chair",
    "close the laptop",
    "close the stapler",
    "close the toilet",
    "grasp the object",
    "place the object on top of the storage furniture"
    "place the object inside the storage furniture"
]

def construct_env(cfg, config_file, env_name, init_state_file, obj_translation=None, real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False, object_name = None):
    if object_name is None:
        config = yaml.safe_load(open(config_file, "r"))
        for config_dict in config:
            if 'name' in config_dict:
                object_name = config_dict['name'].lower()
    
    env, _ = build_up_env(
                    task_config=config_file,
                    env_name=env_name,
                    restore_state_file=init_state_file,
                    # render=False, 
                    render=False, 
                    randomize=False,
                    obj_id=0,
                    horizon=600,
                    random_object_translation=obj_translation,
            )
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                num_points=cfg.task.env_runner.num_point_in_pc,
                                                use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                use_segmask=cfg.task.env_runner.use_segmask,
                                                only_handle_points=cfg.task.env_runner.only_handle_points,
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
                                                )
        
    if randomize_camera:
        pointcloud_env.reset_random_cameras()
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    return env

def run_eval_non_parallel(cfg, env_name, policy: ManiFlowTransformerPointcloudPolicy, num_worker, save_path, cat_idx, exp_beg_idx=0,
                          exp_end_idx=1000, horizon=150, exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None, obj_translation: Optional[list]= None,
                         real_world_camera=False, noise_real_world_pcd=False,
                          randomize_camera=False, pos_ori_imp=False, invert=False):
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        if dataset_index is not None:
            dataset_idx = dataset_index
        
        init_state_files = []
        config_files = []
        object_names = []
        # experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)

        if demo_experiment_path is not None:
            # demo_experiment_path = demo_experiment_path[demo_experiment_path.find("RoboGen_sim2real/") + len("RoboGen_sim2real/"):]
            all_subfolder = os.listdir(demo_experiment_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            all_experiments = all_subfolder

        expert_opened_angles = []
        for experiment in all_experiments:
            if "meta" in experiment:
                continue

            exp_folder = os.path.join(experiment_path, experiment)
            if os.path.exists(os.path.join(exp_folder, "label.json")):
                with open(os.path.join(exp_folder, "label.json"), 'r') as f:
                    label = json.load(f)
                if not label['good_traj']: continue
                
            states_path = os.path.join(exp_folder, "states")
            if not os.path.exists(states_path):
                continue
            if len(os.listdir(states_path)) <= 1 or not os.path.exists(os.path.join(exp_folder, "all.gif")):
                continue
            expert_states = [f for f in os.listdir(states_path) if f.startswith("state")]
            if len(expert_states) == 0:
                continue

            if env_name == 'grasp':
                object_name_path = os.path.join(exp_folder, "object_name.txt")
                if os.path.exists(object_name_path):
                    with open(object_name_path, "r") as f:
                        object_name = f.readlines()[0].lstrip().rstrip().lower()
            else:
                object_name = None 
            
            if env_name == 'articulated':
                expert_opened_angle_file = os.path.join(experiment_path, experiment, "opened_angle.txt")
                if os.path.exists(expert_opened_angle_file):
                    with open(expert_opened_angle_file, "r") as f:
                        angles = f.readlines()
                        expert_opened_angle = float(angles[0].lstrip().rstrip())
                    # max_angle = float(angles[-1].lstrip().rstrip())
                    # ratio = expert_opened_angle / max_angle+0.001)
                # if ratio < 0.65:
                #     continue
                expert_opened_angles.append(expert_opened_angle)
            else:
                expert_opened_angles.append(None)

            init_state_file = os.path.join(states_path, "state_0.pkl")
            init_state_files.append(init_state_file)

            config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
            config_files.append(config_file)
            object_names.append(object_name)

        opened_joint_angles = {}
        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))
        if env_name == "articulated":
            angle_threshold = np.quantile(expert_opened_angles, 0.5)
            if invert:
                selected_idx = [i for i, angle in enumerate(expert_opened_angles) if angle < angle_threshold]
            else:
                selected_idx = [i for i, angle in enumerate(expert_opened_angles) if angle > angle_threshold]

            config_files = [config_files[i] for i in selected_idx]
            object_names = [object_names[i] for i in selected_idx]
            init_state_files = [init_state_files[i] for i in selected_idx]
            expert_opened_angles = [expert_opened_angles[i] for i in selected_idx]

        config_files = config_files[exp_beg_idx:exp_end_idx]
        object_names = object_names[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        print(f"config_files: {config_files}")

        all_distances = []
        all_grasp_distances = []

        for exp_idx, (config_file, object_name, init_state_file) in enumerate(zip(config_files, object_names, init_state_files)):

            env = construct_env(cfg, config_file, env_name, init_state_file, obj_translation, real_world_camera, noise_real_world_pcd, 
                                randomize_camera, object_name=object_name)

            obs = env.reset(open_gripper_at_reset=True)
            rgb = env.env.render()
            info = env.env._env._get_info() 
            all_rgbs = [rgb]

            with tqdm(total=horizon) as pbar:
                for t in range(1, horizon):
                    parallel_input_dict = obs
                    parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                    for key in obs:
                        parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                    print(cat_idx)
                    lang_cond = all_task_names[cat_idx]
                    print(lang_cond)

                    with torch.no_grad():
                        batched_action = policy.predict_action(parallel_input_dict, lang_cond=lang_cond, cat_idx=cat_idx)
                    
                    np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                    np_batched_action = np_batched_action['action']

                    obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                    rgb = env.env.render()
                    all_rgbs.append(rgb)
                    pbar.update(1)
            
            env.env._env.close()

            if env_name == "articulated":
                opened_joint_angles[config_file] = \
                {
                    "final_door_joint_angle": float(info['opened_joint_angle'][-1]), 
                    "expert_door_joint_angle": expert_opened_angles[exp_idx], 
                    "initial_joint_angle": float(info['initial_joint_angle'][-1]),
                    "ik_failure": float(info['ik_failure'][-1]),
                    'oversized_joint_distance': float(info['oversized_joint_distance'][-1]),
                    'grasped_handle': float(info['grasped_handle'][-1]),
                    "exp_idx": exp_idx, 
                }
            elif env_name == "pick_and_place":
                opened_joint_angles[config_file] = \
                {
                    "final_grasped": int(info['current_grasp'][-1]),
                    "final_placed": int(info['current_in_container'][-1]),
                    "grasp_success": int(info['grasped'][-1]),
                    "place_success": int(info['placed'][-1]),
                }
            elif env_name == "grasp":
                opened_joint_angles[config_file] = \
                {
                    "final_grasped": int(info['current_grasp'][-1]),
                    "grasp_success": int(info['grasped'][-1]),
                }
            # gif_save_exp_name = experiment_folder.split("/")[-1] 
            gif_save_exp_name = experiment_folder.split("/")[-2] + "/" + experiment_folder.split("/")[-1]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)                 
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)

            with open("{}/opened_joint_angles.json".format(gif_save_folder), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)

            if env_name == "articulated":
                gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                        float(info["improved_joint_angle"][-1]))
            elif env_name == "pick_and_place":
                gif_save_path = "{}/{}_grasp_{}_place_{}.gif".format(gif_save_folder, exp_idx, 
                        int(info["grasped"][-1]), int(info["placed"][-1]))
            elif env_name == "grasp":
                gif_save_path = "{}/{}_grasp_{}.gif".format(gif_save_folder, exp_idx, 
                        int(info["grasped"][-1]))
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default=None)
    parser.add_argument("--env_name", type=str, default="articulated")
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument("--use_predicted_goal", type=bool, default=True)
    parser.add_argument("--test_cross_category", type=bool, default=False)
    parser.add_argument("--noise_real_world_pcd", type=int, default=0)
    parser.add_argument("--randomize_camera", type=int, default=0)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument('-n', '--noise', type=float, default=None, nargs=2, help='bounds for noise. e.g. `--noise -0.1 0.1')
    parser.add_argument('--pos_ori_imp', action='store_true', help='Set the flag for 10D representation Training')
    parser.add_argument('--exp_dir', type=str, help='Experiment directory')
    parser.add_argument('--invert', action='store_true', help='Set the flag for invert training')
    args = parser.parse_args()
    
    num_worker = 30

    env_name = args.env_name

    categories = ['bucket', 'faucet', 'foldingchair', 'laptop', 'stapler', 'toilet']
    cat_idx = 0
    for i, cat in enumerate(categories):
        if cat in args.exp_dir:
            cat_idx = i + 1
            break
    if args.invert:
        if cat_idx == 0:
            cat_idx = 7
        else:
            cat_idx += 5

    if args.low_level_exp_dir is None:
        # best 50 objects
        exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"
        checkpoint_name = 'latest.ckpt'
    else:
        exp_dir = args.low_level_exp_dir
        checkpoint_name = args.low_level_ckpt_name

    with hydra.initialize(config_path='maniflow/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dit_pointcloud_policy_act3d.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    cfg = recomposed_config
    
    workspace = TrainManiFlowDexWorkspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir)

    #Low level policy loading 
    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')
   
    if 'diverse_objects' in args.exp_dir:
        cfg.task.env_runner.experiment_name = ['0705' for _ in range(1)]
        if args.invert:
            if '45132' in args.exp_dir:
                cfg.task.env_runner.experiment_name = ['invert_0725' for _ in range(1)]
            else:
                cfg.task.env_runner.experiment_name = ['invert' for _ in range(1)]
    else: 
        cfg.task.env_runner.experiment_name = ['165-obj' for _ in range(1)]
        if args.invert:
            if 'toilet' in args.exp_dir or 'stapler' in args.exp_dir:
                cfg.task.env_runner.experiment_name = ['invert_0725' for _ in range(1)]
            else:
                cfg.task.env_runner.experiment_name = ['invert' for _ in range(1)]
    if env_name == 'grasp':
        cfg.task.env_runner.experiment_name = ['gen_grasp' for _ in range(1)]    
        
    cfg.task.env_runner.experiment_folder = [
        args.exp_dir,
    ]
    cfg.task.env_runner.demo_experiment_path = [None for _ in range(1)]
    
    save_path = "data/{}".format(args.eval_exp_name)
    if args.noise is not None:
        save_path = "data/{}_{}_{}".format(args.eval_exp_name, args.noise[0], args.noise[1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_info = {
        "low_level_policy": checkpoint_dir,
        "low_level_policy_checkpoint": checkpoint_name,
    }
    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)
    cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
    cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"
    
    run_eval_non_parallel(
            cfg, env_name, policy,
            num_worker, save_path, cat_idx,
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=25,
            obj_translation=args.noise,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera,
            pos_ori_imp=args.pos_ori_imp,
            invert=args.invert
    )


# python eval_robogen_with_goal_PointNet.py --high_level_ckpt_name /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-09-30_use_75_episodes_200-obj/model_39.pth --eval_exp_name eval_yufei_weighted_displacement_pointnet_large_200_invariant_reproduce --pointnet_class PointNet2_super --model_invariant True