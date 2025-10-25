import sys
from humanoidGym import GYM_ROOT_DIR
import os

import isaacgym
from humanoidGym.envs import *
from humanoidGym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from humanoidGym.utils.helpers import export_policy_as_jit_wmp
from humanoidGym.algo.ppo.normalizer import EmpiricalNormalization

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    num_play_envs = 10
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_play_envs)
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 1
    #env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.add_action_lag = False
    env_cfg.domain_rand.randomize_rfi = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_init_joint_offset = False
    env_cfg.domain_rand.randomize_init_joint_scale = False
    env_cfg.domain_rand.randomize_inertia = False

    env_cfg.env.test = True
    env_cfg.commands.ranges.lin_vel_x = [0.5,0.5]
    env_cfg.commands.ranges.lin_vel_y = [0,0]
    env_cfg.commands.ranges.heading = [0,0]
    env_cfg.commands.ranges.ang_vel_yaw = [0,0]
    env_cfg.env.episode_length_s = 200
    
    # if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt']):
    #         print('terrain should be one of slope, stair, gap, climb, crawl, and tilt, set to climb as default')
    #         args.terrain = 'climb'
    # env_cfg.terrain.terrain_proportions = {
    #         'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
    #         'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
    #         'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
    #         'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
    #         'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
    #         'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
    #     }[args.terrain]
    
    # env_cfg.terrain.terrain_proportions =[0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.5]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,_ = env.get_observations()
    obs_normalizer = EmpiricalNormalization(shape=[env.cfg.env.num_observations], until=1.0e8).to(env.device)
    # n,d = obs.size()
    # obs = obs_normalizer(obs.reshape(-1,env.cfg.env.num_observations)).reshape(n,d)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit_wmp(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    
    

    world_model = ppo_runner._world_model.to(env.device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions),
                                    device=env.device)
    wm_obs = {
            "prop": obs[:, -env.cfg.env.num_single_observations:].to(world_model.device),
            "is_first": wm_is_first,
        }
    if (env.cfg.depth.use_camera):
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                      device=world_model.device)

    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)
    infos = {}
    infos["depth"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                      device=world_model.device).squeeze(-1)
    amp_obs = env.get_amp_observations()
    env.update_current_amp_state(amp_obs)
    for i in range(10*int(env.max_episode_length)):
        if (env.global_counter % wm_update_interval == 0):
            if (env.cfg.depth.use_camera):
                wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)

            wm_embed = world_model.encoder(wm_obs)
            wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
            wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
            wm_is_first[:] = 0

        
        actions = policy(obs.detach(), wm_feature.detach())

        obs,rews, dones, infos = env.step(actions.detach())
        
        # n,d = obs.size()
        # obs = obs_normalizer(obs.reshape(-1,env.cfg.env.num_observations)).reshape(n,d)

        # update world model input
        wm_action_history = torch.concat(
            (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        
        wm_obs = {
            "prop": obs[:, -env.cfg.env.num_single_observations:].to(world_model.device),
            "is_first": wm_is_first,
        }
        if (env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                          device=world_model.device)
        
        reset_env_ids = infos["termination_id"].cpu().numpy()
        if (len(reset_env_ids) > 0):
            wm_action_history[reset_env_ids, :] = 0
            wm_is_first[reset_env_ids] = 1

        wm_action = wm_action_history.flatten(1)
        
        
       
    
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)