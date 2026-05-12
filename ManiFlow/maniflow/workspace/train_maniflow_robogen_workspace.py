if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
# ddp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import subprocess
from hydra.core.hydra_config import HydraConfig
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler

# ddp tools
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0
    # return True  # for debug

def all_reduce_mean(t: torch.Tensor):
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t

OmegaConf.register_new_resolver("eval", eval, replace=True)

def upload_file(local_folder):
    base = "gs://cmu-gpucloud-chenyuah/ManiFlow"
    folder_name = os.path.basename(local_folder.rstrip("/"))
    destination = f"{base}/{folder_name}"
    
    try:
        cmd = ["gcloud", "storage", "rsync", "-r", local_folder, destination]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[Success] Uploaded: {local_folder} -> {destination}")
    except subprocess.CalledProcessError as e:
        print(f"[Failure] Failed to upload {local_folder}: {e.stderr.strip()}")


def _normalizer_flat_key(field_key: str, stat_name: str):
    if stat_name in ('scale', 'offset'):
        return f'params_dict.{field_key}.{stat_name}'
    return f'params_dict.{field_key}.input_stats.{stat_name}'


def _get_identity_normalizer_field_state(field_key: str, dim: int, dtype=torch.float32):
    return {
        _normalizer_flat_key(field_key, 'scale'): torch.ones(dim, dtype=dtype),
        _normalizer_flat_key(field_key, 'offset'): torch.zeros(dim, dtype=dtype),
        _normalizer_flat_key(field_key, 'min'): -torch.ones(dim, dtype=dtype),
        _normalizer_flat_key(field_key, 'max'): torch.ones(dim, dtype=dtype),
        _normalizer_flat_key(field_key, 'mean'): torch.zeros(dim, dtype=dtype),
        _normalizer_flat_key(field_key, 'std'): torch.ones(dim, dtype=dtype),
    }


def maybe_expand_normalizer_channels(normalizer_state_dict, key: str, target_dim: int):
    scale_key = _normalizer_flat_key(key, 'scale')
    offset_key = _normalizer_flat_key(key, 'offset')
    if scale_key not in normalizer_state_dict:
        return normalizer_state_dict
    field_scale = normalizer_state_dict[scale_key]
    current_dim = int(field_scale.numel())
    if current_dim >= target_dim:
        return normalizer_state_dict

    extra_dim = target_dim - current_dim
    dtype = field_scale.dtype
    device = field_scale.device

    def _cat_extra(base_tensor, extra_tensor):
        return torch.cat([base_tensor.detach().to(device=device), extra_tensor.to(device=device)], dim=0)

    min_key = _normalizer_flat_key(key, 'min')
    max_key = _normalizer_flat_key(key, 'max')
    mean_key = _normalizer_flat_key(key, 'mean')
    std_key = _normalizer_flat_key(key, 'std')

    base_min = normalizer_state_dict.get(min_key, -torch.ones(current_dim, dtype=dtype, device=device))
    base_max = normalizer_state_dict.get(max_key, torch.ones(current_dim, dtype=dtype, device=device))
    base_mean = normalizer_state_dict.get(mean_key, torch.zeros(current_dim, dtype=dtype, device=device))
    base_std = normalizer_state_dict.get(std_key, torch.ones(current_dim, dtype=dtype, device=device))

    normalizer_state_dict[scale_key] = _cat_extra(field_scale, torch.ones(extra_dim, dtype=dtype))
    normalizer_state_dict[offset_key] = _cat_extra(
        normalizer_state_dict[offset_key], torch.zeros(extra_dim, dtype=dtype)
    )
    normalizer_state_dict[min_key] = _cat_extra(base_min, -torch.ones(extra_dim, dtype=dtype))
    normalizer_state_dict[max_key] = _cat_extra(base_max, torch.ones(extra_dim, dtype=dtype))
    normalizer_state_dict[mean_key] = _cat_extra(base_mean, torch.zeros(extra_dim, dtype=dtype))
    normalizer_state_dict[std_key] = _cat_extra(base_std, torch.ones(extra_dim, dtype=dtype))
    return normalizer_state_dict


def ensure_identity_normalizer_key(normalizer_state_dict, key: str, dim: int, dtype=torch.float32):
    scale_key = _normalizer_flat_key(key, 'scale')
    if scale_key in normalizer_state_dict:
        return normalizer_state_dict
    normalizer_state_dict.update(_get_identity_normalizer_field_state(key, dim, dtype=dtype))
    return normalizer_state_dict

class TrainManiFlowDexWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None, local_rank=0):
        self.cfg = cfg
        self._output_dir = output_dir
        self.local_rank = local_rank
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ManiFlowTransformerPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ManiFlowTransformerPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        if getattr(cfg, 'load_policy_path', None) is not None:
            self.load_policy(cfg.load_policy_path)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        # train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
        # # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # train_dataloader = DataLoader(
        #     dataset,
        #     **{k:v for k,v in cfg.dataloader.items() if k != "shuffle"},  # 不要再传 shuffle
        #     shuffle=False,
        #     sampler=train_sampler,

        # )
        train_dataloader = DataLoader(dataset, 
                                      shuffle=False,
                                      sampler=DistributedSampler(dataset),
                                      batch_size=cfg.dataloader.batch_size,
                                      num_workers=cfg.dataloader.num_workers,
                                      pin_memory=True,
                                      )
        normalizer = dataset.get_normalizer()

        # print dataset info
        cprint(f"Dataset: {dataset.__class__.__name__}", 'red')
        # cprint(f"Dataset Path: {dataset.zarr_path}", 'red')
        # cprint(f"Number of training episodes: {dataset.train_episodes_num}", 'red')
        # cprint(f"Number of validation episodes: {dataset.val_episodes_num}", 'red')


        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        val_dataloader = DataLoader(val_dataset, 
                                      shuffle=False,
                                      sampler=DistributedSampler(val_dataset),
                                      batch_size=cfg.dataloader.batch_size,
                                      num_workers=cfg.dataloader.num_workers,
                                      pin_memory=True,
                                      )

        if getattr(cfg.training, 'use_dataset_normalization', True):
            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        env_runner = None

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        if is_main_process():
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update({"output_dir": self.output_dir})
        else:
            # 其它进程禁用 wandb，避免多进程重复写
            wandb_run = None
            os.environ["WANDB_MODE"] = "offline"
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(f"cuda:{self.local_rank}")
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # 用 DDP 包装主模型
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False  # 若模型存在条件分支用不到的参数，可设 True
        )

         # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            dist.barrier()
            train_dataloader.sampler.set_epoch(self.epoch)
            val_dataloader.sampler.set_epoch(self.epoch)
            print(f"[rank{get_rank()}] START EPOCH {self.epoch}", flush=True)
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            use_tqdm = is_main_process()
            iterable = train_dataloader if not use_tqdm else tqdm.tqdm(
                train_dataloader, desc=f"Training epoch {self.epoch}",
                leave=False, mininterval=cfg.training.tqdm_interval_sec
            )
            for batch_idx, batch in enumerate(iterable):
                t1 = time.time()
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = batch
            
                # compute loss
                t1_1 = time.time()
                # Forward pass
                raw_loss, loss_dict = self.model.module.compute_loss(batch, self.ema_model)

                loss = raw_loss / cfg.training.gradient_accumulate_every
                # Only synchronize gradients on the last accumulation step
                will_sync = True

                if will_sync:
                    loss.backward()
                else:
                    # ✅ prevent unnecessary allreduce calls
                    with self.model.no_sync():
                        loss.backward()

                # Update parameters only on sync step
                if will_sync:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                
                t1_2 = time.time()

                # step optimizer
                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                t1_3 = time.time()
                # update ema
                if cfg.training.use_ema:
                    ema.step(self.model.module)
                t1_4 = time.time()
                # logging
                raw_loss_cpu = raw_loss.item()
                if use_tqdm:
                    iterable.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }
                t1_5 = time.time()
                step_log.update(loss_dict)
                t2 = time.time()
                
                if verbose:
                    print(f"total one step time: {t2-t1:.3f}")
                    print(f" compute loss time: {t1_2-t1_1:.3f}")
                    print(f" step optimizer time: {t1_3-t1_2:.3f}")
                    print(f" update ema time: {t1_4-t1_3:.3f}")
                    print(f" logging time: {t1_5-t1_4:.3f}")

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    # log of last step is combined with validation and rollout
                    if is_main_process():
                        wandb_run.log(step_log, step=self.global_step)
                    self.global_step += 1

                if (cfg.training.max_train_steps is not None) \
                    and batch_idx >= (cfg.training.max_train_steps-1):
                    break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()
            
            # # run rollout
            # if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
            #     t3 = time.time()
            #     # runner_log = env_runner.run(policy, dataset=dataset)
            #     runner_log = env_runner.run(policy)
            #     t4 = time.time()
            #     # print(f"rollout time: {t4-t3:.3f}")
            #     # log all
            #     step_log.update(runner_log)

            rollout_multiple_inference_step = True
            if rollout_multiple_inference_step:
                # run rollout
                cur_inference_step = cfg.policy.num_inference_steps
                # all_rollout_steps = [1, 4, 10]
                all_rollout_steps = [10]
                # add cur_inference_step to all_rollout_steps if not in all_rollout_steps
                all_rollout_steps = list(set(all_rollout_steps + [cur_inference_step]))

                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    for inference_step in all_rollout_steps:
                        cprint(f"Running rollout with inference step {inference_step}", 'green')
                        policy.num_inference_steps = inference_step
                        # runner_log = env_runner.run(policy, dataset=dataset)
                        runner_log = env_runner.run(policy)
                        new_runner_log = dict()
                        # add inference step to all keys in runner_log
                        for key in runner_log:
                            new_key = f"{key}_infer{inference_step}"
                            new_runner_log[new_key] = runner_log[key]
                            # runner_log[key] = {f"{key}/{inference_step}": runner_log[key]}
                        runner_log = new_runner_log
                        t4 = time.time()
                        # print(f"rollout time: {t4-t3:.3f}")
                        # log all
                        step_log.update(runner_log)
            else:
                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    runner_log = env_runner.run(policy)
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                            # Forward pass
                            loss, loss_dict = self.model.module.compute_loss(batch, self.ema_model)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    cat_idx = batch['cat_idx'].squeeze(-1)
                    result = policy.predict_action(obs_dict, cat_idx=cat_idx)
                    pred_action = result['action_pred']
                    pred_action = pred_action.to(device)
                    gt_action = gt_action.to(device)
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint
            if is_main_process() and (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                self.save_checkpoint(tag=f'epoch-{self.epoch}')
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()
                # def _async_upload():
                #     try:
                #         upload_file(self.output_dir)
                #         print(f"Epoch {self.epoch} uploaded.")
                #     except Exception as e:
                #         print(f"[Upload Warning] {e}")

                # threading.Thread(target=_async_upload, daemon=True).start()
                # print(f"Epoch {self.epoch} uploaded.")

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                # if topk_ckpt_path is not None:
                #     self.save_checkpoint(path=topk_ckpt_path)
                # if topk_ckpt_path is not None:
                #     self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            # dist.barrier()
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if is_main_process():
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
    def load_policy(self, path):
        path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')

        def _strip_module_prefix(state_dict):
            if not any(key.startswith('module.') for key in state_dict.keys()):
                return state_dict
            return {
                key[len('module.'):] if key.startswith('module.') else key: value
                for key, value in state_dict.items()
            }

        def _filter_state_dict_for_module(module, loaded_state_dict, module_name):
            loaded_state_dict = _strip_module_prefix(loaded_state_dict)
            normalizer_keys = set()
            normalizer_state_dict = {}
            if hasattr(module, 'normalizer'):
                normalizer_prefix = 'normalizer.'
                normalizer_state_dict = {
                    key[len(normalizer_prefix):]: value
                    for key, value in loaded_state_dict.items()
                    if key.startswith(normalizer_prefix)
                }
                normalizer_keys = {
                    key for key in loaded_state_dict.keys()
                    if key.startswith(normalizer_prefix)
                }
                target_pc_dim = int(getattr(self.cfg.policy.pointcloud_encoder_cfg, 'in_channels', 3))
                agent_pos_dim = int(self.cfg.task.shape_meta.obs.agent_pos.shape[0])
                action_dim = int(np.prod(self.cfg.task.shape_meta.action.shape))
                if target_pc_dim > 3:
                    normalizer_state_dict = maybe_expand_normalizer_channels(
                        normalizer_state_dict, 'point_cloud', target_pc_dim
                    )
                normalizer_state_dict = ensure_identity_normalizer_key(
                    normalizer_state_dict, 'point_cloud', target_pc_dim
                )
                normalizer_state_dict = ensure_identity_normalizer_key(
                    normalizer_state_dict, 'agent_pos', agent_pos_dim
                )
                normalizer_state_dict = ensure_identity_normalizer_key(
                    normalizer_state_dict, 'action', action_dim
                )

            current_state_dict = module.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for key, value in loaded_state_dict.items():
                if key in normalizer_keys:
                    continue
                if key not in current_state_dict:
                    skipped_keys.append((key, "missing_in_current_model"))
                    continue
                if current_state_dict[key].shape != value.shape:
                    skipped_keys.append(
                        (key, f"shape_mismatch ckpt={tuple(value.shape)} current={tuple(current_state_dict[key].shape)}")
                    )
                    continue
                filtered_state_dict[key] = value

            if skipped_keys:
                print(f"Skipping {len(skipped_keys)} {module_name} keys during partial load:")
                for key, reason in skipped_keys:
                    print(f"  {key}: {reason}")
            missing_keys, unexpected_keys = module.load_state_dict(filtered_state_dict, strict=False)
            if hasattr(module, 'normalizer') and normalizer_state_dict:
                module.normalizer.load_state_dict(normalizer_state_dict, strict=False)
            if missing_keys:
                print(f"Missing {module_name} keys after partial load: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected {module_name} keys after partial load: {unexpected_keys}")

        _filter_state_dict_for_module(self.model, payload['state_dicts']['model'], "model")
        if self.ema_model is not None and 'ema_model' in payload['state_dicts']:
            _filter_state_dict_for_module(self.ema_model, payload['state_dicts']['ema_model'], "ema_model")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config'))
)
def main(cfg):
    # -------- DDP init begin --------
    # torchrun 会自动设置 LOCAL_RANK / RANK / WORLD_SIZE 等环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    # 为了不同进程的数据打乱不一致
    seed = int(cfg.training.seed)
    seed = seed + get_rank()
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    # -------- DDP init end --------

    try:
        workspace = TrainManiFlowDexWorkspace(cfg, local_rank=local_rank)
        print(f"Workspace created. Local rank: {local_rank}. Global rank: {get_rank()}.")
        workspace.run()
    finally:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
