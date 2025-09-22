import sys
from humanoidGym import GYM_ROOT_DIR
import os

import isaacgym
from humanoidGym.envs import *
from humanoidGym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from humanoidGym.utils.helpers import export_policy_as_rnn_jit


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    num_play_envs = 10
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_play_envs)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 1
    #env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
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
    env_cfg.commands.ranges.lin_vel_x = [0,2]
    env_cfg.commands.ranges.lin_vel_y = [0,0]
    env_cfg.commands.ranges.heading = [0,0]
    env_cfg.commands.ranges.ang_vel_yaw = [0,0]
    
    env_cfg.terrain.stair_height_range = [0.15, 0.15]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,_ = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    h0 = torch.zeros(1, num_play_envs, 256).to(env.device)
    c0 = torch.zeros(1, num_play_envs, 256).to(env.device)
    
    est_h0 = torch.zeros(1, num_play_envs, 256).to(env.device)
    est_c0 = torch.zeros(1, num_play_envs, 256).to(env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_rnn_jit(env,policy, path)
        print('Exported policy as jit script to: ', path)

    amp_obs = env.get_amp_observations()
    env.update_current_amp_state(amp_obs)
    for i in range(10*int(env.max_episode_length)):
        actions,est_h0,est_c0,h0,c0 = policy(obs.detach(),est_h0,est_c0,h0,c0)
        obs,rews, dones, infos = env.step(actions.detach())
    
if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
