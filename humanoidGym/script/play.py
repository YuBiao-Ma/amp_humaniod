import sys
from humanoidGym import GYM_ROOT_DIR
import os

import isaacgym
from humanoidGym.envs import *
from humanoidGym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib

# 无显示环境或服务器上运行时：强制使用无界面后端
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")          # ★ 在 import pyplot 之前
# 不要工具栏，避免 Tk 创建图标
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from humanoidGym.utils.helpers import export_policy_as_rnn_jit

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
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
    env_cfg.domain_rand.add_action_lag = True
    env_cfg.domain_rand.randomize_rfi = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_init_joint_offset = False
    env_cfg.domain_rand.randomize_init_joint_scale = False
    env_cfg.domain_rand.randomize_inertia = False

    env_cfg.env.test = True
    env_cfg.commands.ranges.lin_vel_x = [0.5,2]
    env_cfg.commands.ranges.lin_vel_y = [0,0]
    env_cfg.commands.ranges.heading = [0,0]
    # env_cfg.commands.ranges.ang_vel_yaw = [0,0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs,_ = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ====== rollout 配置 ======
    STEPS = 3000
    REC_ENV_ID = 0  # 记录第0个环境

    # NEW: 分别为 target_dof_pos 与 dof_pos 建两个缓冲
    buf_tdp = []   # target dof pos
    buf_pos = []   # measured dof pos

    joint_order = list(env_cfg.init_state.default_joint_angles.keys())
    left_joints = [
        "joint_left_hip_pitch",
        "joint_left_hip_roll",
        "joint_left_hip_yaw",
        "joint_left_knee",
        "joint_left_ankle_pitch",
        "joint_left_ankle_roll",
    ]
    right_joints = [
        "joint_right_hip_pitch",
        "joint_right_hip_roll",
        "joint_right_hip_yaw",
        "joint_right_knee",
        "joint_right_ankle_pitch",
        "joint_right_ankle_roll",
    ]
    name_to_idx = {name: i for i, name in enumerate(joint_order)}

    if EXPORT_POLICY:
        path = os.path.join(GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(env, policy, path)
        print('Exported policy as jit script to: ', path)

    amp_obs = env.get_amp_observations()
    env.update_current_amp_state(amp_obs)

    # NEW: 只滚动 STEPS 步，便于在第 STEPS-1 步触发绘图
    for i in range(STEPS):
        actions = policy(obs.detach())
        obs, rews, dones, infos = env.step(actions.detach())

        # 读取期望关节位置（target）与实际关节位置（measured）
        tdp = env.target_dof_pos  # [num_envs, dof]
        dpos = env.dof_pos        # [num_envs, dof]  —— 假设为弧度；若是归一化需自行反归一化

        if isinstance(tdp, torch.Tensor):
            tdp = tdp.detach().cpu().numpy()
        if isinstance(dpos, torch.Tensor):
            dpos = dpos.detach().cpu().numpy()

        # 只记录指定环境
        buf_tdp.append(tdp[REC_ENV_ID].copy())
        buf_pos.append(dpos[REC_ENV_ID].copy())

        if PLOT and i == STEPS - 1:
            data_tdp = np.asarray(buf_tdp)  # (steps, dof)
            data_pos = np.asarray(buf_pos)  # (steps, dof)
            steps, dof = data_tdp.shape

            fig, axes = plt.subplots(6, 2, figsize=(10, 12), sharex=True)
            x = np.arange(steps)

            # 左腿
            for row, name in enumerate(left_joints):
                j = name_to_idx[name]
                ax = axes[row, 0]
                ax.plot(x, data_tdp[:, j], linewidth=1.5, label="target")   # 折线
                ax.plot(x, data_pos[:, j], linewidth=1.0, linestyle="--", label="measured")
                ax.set_title(name)
                ax.set_ylabel("dof_pos [rad]")
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.legend(frameon=False)

            # 右腿
            for row, name in enumerate(right_joints):
                j = name_to_idx[name]
                ax = axes[row, 1]
                ax.plot(x, data_tdp[:, j], linewidth=1.5, label="target")
                ax.plot(x, data_pos[:, j], linewidth=1.0, linestyle="--", label="measured")
                ax.set_title(name)
                ax.set_ylabel("dof_pos [rad]")
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.legend(frameon=False)

            for col in range(2):
                axes[-1, col].set_xlabel("step")

            fig.suptitle(f"Env {REC_ENV_ID} — target vs measured DOF positions (first {steps} steps)")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()


    
if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    PLOT = False
    args = get_args()

    play(args)
