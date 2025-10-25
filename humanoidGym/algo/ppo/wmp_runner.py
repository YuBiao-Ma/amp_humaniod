import time
import os
from collections import deque
import statistics
from typing import Union

from torch.utils.tensorboard import SummaryWriter
import torch

from humanoidGym.algo.dataset.motion_loader import AMPLoader
from humanoidGym.algo.dataset.lite_motion_loader import LongAMPLoader
from humanoidGym.algo.ppo.discriminator import Discriminator
from humanoidGym.algo.dreamer.models import *
from .modules import DepthPredictor
from .amp_ppo import AMPPPO
from .wmp_ppo import WMPPPO
from .actor_critic import ActorCritic, InferenceActor,ActorCriticWMP,InferenceActorWMP
from .actor_crtic_recurrent import ActorCriticRecurrent, InferenceActorLSTM, ActorCriticRecurrentWMP
from .normalizer import EmpiricalDiscountedVariationNormalization, EmpiricalHistoryNormalization,EmpiricalNormalization
from humanoidGym.algo import VecEnv

import ruamel.yaml as yaml
import argparse
import pathlib
import sys
import collections
from humanoidGym.algo.dreamer import tools
import datetime
import uuid
import numpy as np

class WMPRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: Union[str, None] = None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.depth_predictor_cfg = train_cfg["depth_predictor"]
        self.device = device
        self.env = env
  

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        # build world model
        self._build_world_model()
        
        # build depth predictor
        self.depth_predictor = DepthPredictor().to(self._world_model.device)
        self.depth_predictor_opt = torch.optim.Adam(self.depth_predictor.parameters(), lr=self.depth_predictor_cfg["lr"],
                                              weight_decay=self.depth_predictor_cfg["weight_decay"])


        actor_critic_class = eval(self.cfg.pop("policy_class_name"))  # ActorCritic
        actor_critic = actor_critic_class(self.env.cfg.env.num_single_observations,
                                                        self.env.cfg.env.num_privileged_obs,
                                                        self.env.cfg.env.num_obs_lens,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        # amp related
        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        
        amp_state_normalizer = EmpiricalNormalization(shape=amp_data.observation_dim).to(self.device)
        
        if self.cfg["normalize_style_reward"]:
            style_reward_normalizer = EmpiricalNormalization(shape=1).to(self.device)
        else:
            style_reward_normalizer = None
            
        discriminator = Discriminator(
            observation_dim=amp_data.observation_dim,
            observation_horizon=2, # set 2 for now
            device=self.device,
            **self.discriminator_cfg).to(self.device)
        
        self.env.discriminator = discriminator
        self.env.amp_state_normalizer = amp_state_normalizer
        self.env.style_reward_normalizer = style_reward_normalizer

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for they key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # init algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: AMPPPO = alg_class(actor_critic, discriminator,amp_data,amp_state_normalizer, device=self.device, **self.alg_cfg)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        
        if self.empirical_normalization:
            # self.obs_normalizer = EmpiricalHistoryNormalization(shape=[self.env.cfg.env.num_single_observations], until=1.0e8).to(self.device)
            self.obs_normalizer = EmpiricalNormalization(shape=[self.env.cfg.env.num_observations], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0


    def _build_world_model(self):
        # world model
        print('Begin construct world model')
        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent.parent / "humanoidGym/algo/dreamer/configs.yaml").read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config = parser.parse_args()
        # allow world model and rl env on different device
        if (self.wm_config.wm_device != 'None'):
            self.wm_config.device = self.wm_config.wm_device
        self.wm_config.num_actions = self.wm_config.num_actions * self.env.cfg.depth.update_interval
        prop_dim = self.env.cfg.env.num_single_observations
        image_shape = self.env.cfg.depth.resized + (1,)
        obs_shape = {'prop': (prop_dim,), 'image': image_shape,}

        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera)
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        self.wm_feature_dim = self.wm_config.dyn_deter #+ self.wm_config.dyn_stoch * self.wm_config.dyn_discrete


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        amp_obs = self.env.get_amp_observations()
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        
        # init normalize
        n,d = obs.size()
        obs = self.obs_normalizer(obs.reshape(-1,self.env.cfg.env.num_observations)).reshape(n,d)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        


        # init world model input
        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, -self.env.cfg.env.num_single_observations:].to(self._world_model.device),
            "is_first": wm_is_first,
        }

        if(self.env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)

        wm_metrics = None
        self.wm_update_interval = self.env.cfg.depth.update_interval
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions),
                                        device=self._world_model.device)
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim))

        self.init_wm_dataset()

        for it in range(start_iter, tot_iter):
            if (self.env.cfg.rewards.reward_curriculum):
                self.env.update_reward_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions from policy
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        # world model obs step
                        wm_embed = self._world_model.encoder(wm_obs)
                        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                                                                           wm_obs["is_first"])
                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                        wm_is_first[:] = 0

                   
                    actions = self.alg.act(obs, critic_obs, amp_obs, wm_feature.to(self.env.device))
                    
                    # save amp state for style reward prediction 
                    self.env.update_current_amp_state(amp_obs)
                    
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    
                    # 获取额外信息
                    amp_next_obs = self.env.get_amp_observations()
                    critic_obs = infos["observations"]["critic"]
                    # 获取状态转换obs为了训练esitimator
                    termination_critic_obs = infos["termination_privileged_obs"]
                    termination_ids = infos["termination_id"]
                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_critic_obs.clone().detach()
                    

                    # Move to the agent device
                    critic_obs,termination_critic_obs,termination_ids,next_critic_obs = critic_obs.to(self.device),termination_critic_obs.to(self.device),termination_ids.to(self.device),next_critic_obs.to(self.device)
                    obs, rewards, dones, amp_next_obs = obs.to(self.device), rewards.to(self.device), dones.to(self.device), amp_next_obs.to(self.device)

                    # Normalize observations
                    n,d = obs.size()
                    obs = self.obs_normalizer(obs.reshape(-1,self.env.cfg.env.num_observations)).reshape(n,d)
                    
                    # Extract critic observations and normalize
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(critic_obs)
                    else:
                        critic_obs = obs


                    # update world model input
                    wm_action_history = torch.concat(
                        (wm_action_history[:, 1:], actions.unsqueeze(1).to(self._world_model.device)), dim=1)
                    wm_obs = {
                    "prop": obs[:, -self.env.cfg.env.num_single_observations:].to(self._world_model.device),
                    "is_first": wm_is_first,
                            }
                    
                    # store the data in buffer into the dataset before reset
                    reset_env_ids = termination_ids.cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                            if(k == "image"):
                                for id in reset_env_ids:
                                    idx_in_buffer = np.where(self.env.depth_index == id)[0]
                                    if(len(idx_in_buffer) > 0):
                                        v[idx_in_buffer, :] = self.wm_buffer[k][idx_in_buffer].to(self._world_model.device)
                            else:
                                v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self._world_model.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_action_history[reset_env_ids, :] = 0
                        wm_is_first[reset_env_ids] = 1

                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self._world_model.device)
                    
                    # store current step into buffer
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        if (self.env.cfg.depth.use_camera):
                            forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
                            pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                            wm_obs["image"] = pred_depth_image
                            self.wm_buffer["forward_height_map"][range(self.env.num_envs), self.wm_buffer_index,:] = forward_heightmap[:].to('cpu')
                            wm_obs["image"][self.env.depth_index] = infos["depth"].unsqueeze(-1).to(self._world_model.device)
                            self.wm_buffer["image"][range(self.env.cfg.depth.camera_num_envs),
                            self.wm_buffer_index[self.env.depth_index], :] = wm_obs["image"][self.env.depth_index].to(
                                'cpu')
                        # not_reset_env_ids = (~dones).nonzero(as_tuple=False).flatten().cpu().numpy()
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1

                        wm_reward[:] = 0

                    # Process env step and store in buffer
                    self.alg.process_env_step(rewards, dones, infos, amp_next_obs, next_critic_obs)
                    
                   

                    # Intrinsic rewards (extracted here only for logging)!
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, wm_feature.to(self.env.device))

            # Update policy
            # reset boost data
            # self.alg.storage.refresh_bootstrapping_data()
            # Note: we keep arguments here since locals() loads them
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss,mean_subtask_loss,mean_smooth_loss,mean_amp_loss,mean_grad_pen_loss,mean_policy_pred,mean_expert_pred = self.alg.update(it)
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            start_time = time.time()
            if (sum_wm_dataset_size > self.wm_config.train_start_steps):

                if(it % self.depth_predictor_cfg["training_interval"] == 0):
                # Train Depth Predictor
                    depth_mse_loss = self.train_depth_predictor()
                    self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)

                # Train World Model
                wm_metrics = self.train_world_model()
                for name, values in wm_metrics.items():
                    self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
            print('training world model time:', time.time() - start_time)

   

                        

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def init_wm_dataset(self):
        self.wm_dataset = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.num_single_observations),
                                device=self._world_model.device),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device=self._world_model.device),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device=self._world_model.device),
        }
        if(self.env.cfg.depth.use_camera):
            self.wm_dataset["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)
            self.wm_dataset["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device=self._world_model.device)

        self.wm_dataset_size = np.zeros(self.env.num_envs)

        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.num_single_observations),
                                device='cpu'),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device='cpu'),
        }
        if(self.env.cfg.depth.use_camera):
            self.wm_buffer["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device='cpu')
            self.wm_buffer["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device='cpu')

        self.wm_buffer_index = np.zeros(self.env.num_envs)

    def train_depth_predictor(self):
        total_mse_loss = 0
        for _ in range(self.depth_predictor_cfg["training_iters"]):
            batch_idx = np.random.choice(self.env.depth_index_without_crawl_tilt, self.depth_predictor_cfg["batch_size"],
                                         replace=True)
            time_index = [np.random.randint(0, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            forward_heightmap = self.wm_dataset["forward_height_map"][batch_idx, time_index]
            prop = self.wm_dataset["prop"][batch_idx, time_index]
            depth_image = self.wm_dataset["image"][self.env.depth_index_inverse[batch_idx], time_index]

            predict_depth_image = self.depth_predictor(forward_heightmap, prop)
            depth_predict_loss = (depth_image - predict_depth_image).pow(2).mean() * self.depth_predictor_cfg[
                "loss_scale"]
            # Gradient step
            self.depth_predictor_opt.zero_grad()
            depth_predict_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
            self.depth_predictor_opt.step()
            total_mse_loss += depth_predict_loss.detach() / self.depth_predictor_cfg["loss_scale"]
        return float(total_mse_loss / self.depth_predictor_cfg["training_iters"])

    def train_world_model(self):
        wm_metrics = {}
        mets = {}
        for i in range(self.wm_config.train_steps_per_iter):
            p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
            batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                         p=p)
            batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
            if (batch_length <= 1):
                continue  # an error occur about the predict loss if batch_length < 1
            batch_end_idx = [np.random.randint(batch_length, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            batch_data = {}
            for k, v in self.wm_dataset.items():
                if (k == "forward_height_map"):
                    continue
                value = []
                for idx, end_idx in zip(batch_idx, batch_end_idx):
                    if (k == "image"):
                        idx_in_buffer = np.where(self.env.depth_index == idx)[0]
                        if (len(idx_in_buffer) == 0):
                            # not in the buffer, use the predicted ones
                            tmp_forward_heightmap = self.wm_dataset["forward_height_map"][idx,
                                                    end_idx - batch_length: end_idx]
                            tmp_prop = self.wm_dataset["prop"][idx, end_idx - batch_length: end_idx]
                            pred_depth_image = self.depth_predictor(tmp_forward_heightmap, tmp_prop)
                            value.append(pred_depth_image)
                        else:
                            value.append(v[idx_in_buffer[0], end_idx - batch_length: end_idx])
                    else:
                        value.append(v[idx, end_idx - batch_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value
            is_first = torch.zeros((self.wm_config.batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            post, context, mets = self._world_model._train(batch_data)
        wm_metrics.update(mets)
        return wm_metrics

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/entropy", locs["mean_entropy"], locs["it"])
        self.writer.add_scalar("Loss/subtask", locs["mean_subtask_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Loss/smooth", locs["mean_smooth_loss"],locs["it"])
        self.writer.add_scalar("Loss/amp", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/amp_grad", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Discriminator/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Discriminator/expert_pred", locs["mean_expert_pred"], locs["it"])
        if self.alg.rnd:
            self.writer.add_scalar("Loss/rnd", locs["mean_rnd_loss"], locs["it"])
        if self.alg.symmetry:
            self.writer.add_scalar("Loss/symmetry", locs["mean_symmetry_loss"], locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )

            # -- For symmetry
            if self.alg.symmetry:
                log_string += f"""{'Symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""

            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""

            # -- For RND
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )

            log_string += f"""{'Mean total reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )
            # -- For symmetry
            if self.alg.symmetry:
                log_string += f"""{'Symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""

            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""

            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save PPO model
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "policy_optimizer_state_dict": self.alg.policy_optimizer.state_dict(),
            'world_model_dict': self._world_model.state_dict(),
            'depth_predictor': self.depth_predictor.state_dict(),
            "discriminator_optimizer_state_dict": self.alg.discriminator_optimizer.state_dict(),
            "state_normalizer": self.env.amp_state_normalizer.state_dict(),
            "style_reward_normalizer": self.env.style_reward_normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load PPO model
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        # --load wmp
        self._world_model.load_state_dict(loaded_dict['world_model_dict'], strict=False)

        # -- Load AMP related
        self.env.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.env.amp_state_normalizer.load_state_dict(loaded_dict["state_normalizer"])
        self.env.style_reward_normalizer.load_state_dict(loaded_dict["style_reward_normalizer"])
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        # -- Load optimizer if used
        if load_optimizer:
            # -- PPO
            self.alg.policy_optimizer.load_state_dict(loaded_dict["policy_optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            # -- AMP
            self.alg.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
            
        # -- Load current learning iteration
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        if self.alg.actor_critic.is_recurrent:
            policy = self.alg.actor_critic.actor
        else:
            policy = self.alg.actor_critic.act_inference#actor_teacher_backbone
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            if self.alg.actor_critic.is_recurrent:
                policy = InferenceActorLSTM(self.alg.actor_critic.actor,self.obs_normalizer)
            else:
                policy = InferenceActorWMP(self.alg.actor_critic,self.obs_normalizer)
            policy.eval()
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.actor_critic.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()
        # -- AMP
        self.env.discriminator.train()
        self.env.amp_state_normalizer.train()
        self.env.style_reward_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor_critic.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()
        # -- AMP
        self.env.discriminator.eval()
        self.env.amp_state_normalizer.eval()
        self.env.style_reward_normalizer.eval()