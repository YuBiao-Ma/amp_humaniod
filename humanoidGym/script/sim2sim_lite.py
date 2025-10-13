# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

import math
import time
import numpy as np
import mujoco, mujoco_viewer
import mujoco.viewer as mjv
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoidGym.envs.lite.lite_config import LiteAmpCfg
import torch
import onnxruntime as ort  # 导入 ONNX Runtime
from humanoidGym import GYM_ROOT_DIR

# default_dof_pos = [0.24,-0.0,0.12,0.3,0.155,0,0.24,-0.0,0.12,0.3,0.155,0]
default_dof_pos = [0.39,-0.0,-0.12,0.74,0.36,0,0.39,-0.0,-0.12,0.74,0.36,0]

class KeyboardCmd:
    def __init__(self):
        self.forward=False
        self.backward=False
        self.left=False
        self.right=False
        self.turn_left=False
        self.turn_right=False
        self.reset=False
    def zero(self):
        self.forward=False
        self.backward=False
        self.left=False
        self.right=False
        self.turn_left=False
        self.turn_right=False
        self.reset=False
    def __str__(self) -> str:
        s="Key: "
        if self.forward:
            s+="W "
        if self.backward:
            s+="S "
        if self.left:
            s+="A "
        if self.right:
            s+="D "
        if self.turn_left:
            s+="Q "
        if self.turn_right:
            s+="E "
        if self.reset:
            s+="R "
        return s

class KeyboardCmdHandler:
    def __init__(self,max_linear_speed=5,max_angular_speed=1.5,mode_game=False,step=0.01):
        self.linear_x = 0
        self.linear_y = 0
        self.angular_z = 0
        self.heading = 0
        self.step=step
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.game_mode=mode_game
    def update(self,key_cmd:KeyboardCmd):
        if key_cmd.forward:
            self.linear_x += self.step
        elif self.game_mode:
            if self.linear_x>0:
                self.linear_x -= self.step
        if key_cmd.backward:
            self.linear_x -= self.step
        elif self.game_mode:
            if self.linear_x<0:
                self.linear_x += self.step
        if key_cmd.right:
            self.linear_y += self.step
        elif self.game_mode:
            if self.linear_y>0:
                self.linear_y -= self.step
        if key_cmd.left:
            self.linear_y -= self.step
        elif self.game_mode:
            if self.linear_y<0:
                self.linear_y += self.step
        if key_cmd.turn_left:
            self.angular_z += self.step
        elif self.game_mode:
            if self.angular_z>0:
                self.angular_z -= self.step
        if key_cmd.turn_right:
            self.angular_z -= self.step
        elif self.game_mode: 
            if self.angular_z<0:
                self.angular_z += self.step

        # clip values
        self.linear_x = np.clip(self.linear_x,-self.max_linear_speed,self.max_linear_speed)
        self.linear_y= np.clip(self.linear_y,-self.max_linear_speed,self.max_linear_speed)
        self.angular_z = np.clip(self.angular_z,-self.max_angular_speed,self.max_angular_speed)

kb_cmd = KeyboardCmd()

def keyboard_callback(keycode):
        # 一些系统发送小写；统一转大写兼容
        try:
            ch = chr(keycode).upper()
        except Exception:
            return
        print(f"key {ch} is pressed")
        if ch=='W':
            kb_cmd.forward=True
        elif ch=='S':
            kb_cmd.backward=True
        elif ch=='A':
            kb_cmd.left=True
        elif ch=='D':
            kb_cmd.right=True
        elif ch=='Q':
            kb_cmd.turn_left=True
        elif ch=='E':
            kb_cmd.turn_right=True
        elif ch=='R':
            kb_cmd.reset=True

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def viewer_running(v):
    """
    兼容不同 viewer 实现的“仍在运行”判断：
    - mujoco.viewer.Viewer: is_running()
    - 第三方 mujoco_viewer.MujocoViewer: 有的版本提供 is_alive / is_running
    - 若无可用接口，默认 True
    """
    for name in ("is_running", "is_alive", "is_active"):
        attr = getattr(v, name, None)
        if callable(attr):
            try:
                return bool(attr())
            except Exception:
                pass
        elif isinstance(attr, bool):
            return attr
    return True

def get_gravity_orientation(quaternion):    
    qx = quaternion[0]
    qy = quaternion[1]
    qz = quaternion[2]
    qw = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    # gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    gvec = get_gravity_orientation(quat).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions,last_actions,flt):
    return actions * flt + last_actions * (1-flt)

def run_mujoco(policy, cfg, render_fps=20, rt_speed=1.0):
    global default_dof_pos
    """
    Run the Mujoco simulation using the provided policy and configuration.
    render_fps: 限制可视化帧率（例如 20 FPS）
    rt_speed:   实时倍率（1.0 实时，0.5 半速，2.0 两倍速）
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)  # 载入初始化位置由XML决定
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    # viewer = mujoco_viewer.MujocoViewer(model, data)  # 若需要第三方 Viewer
    viewer = mjv.launch_passive(model=model, data=data, key_callback=keyboard_callback)
    keyboard_cmd_handler = KeyboardCmdHandler(step=0.1)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)     # 12
    action = np.zeros((cfg.env.num_actions), dtype=np.double)       # 12
    action_flt = np.zeros((cfg.env.num_actions), dtype=np.double)   # 12
    last_actions = np.zeros((cfg.env.num_actions), dtype=np.double) # 12
    hist_obs = deque(maxlen=cfg.env.num_obs_lens)
    for _ in range(cfg.env.num_obs_lens):
        hist_obs.append(np.zeros([1, cfg.env.num_single_observations], dtype=np.double)) # 39

    count_lowlevel = 0
    init=0
    action_count = 0

    # ---- 渲染限帧：每多少个仿真步渲染一次 ----
    render_stride = max(1, int(round(1.0 / (cfg.sim_config.dt * max(1, render_fps)))))  # dt=0.001, 20 FPS -> 50步渲染一次

    # ---- 实时节流：按墙钟控制仿真速度 ----
    next_wall = time.perf_counter()
    wall_step = cfg.sim_config.dt / max(1e-6, rt_speed)  # rt_speed=1.0 -> 1×实时

    try:
        for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
            # ---- 如果窗口被关闭，优雅退出 ----
            if viewer is not None and not viewer_running(viewer):
                print("[INFO] Viewer closed. Stopping simulation.")
                break

            # 限帧渲染（与控制频率解耦）
            if viewer is not None and viewer_running(viewer):
                if (count_lowlevel % render_stride) == 0:
                    viewer.sync()

            # Obtain an observation
            q, dq, quat, v, omega, gvec = get_obs(data)  # 从 Mujoco 获取仿真数据
            q = q[-cfg.env.num_actions:]
            dq = dq[-cfg.env.num_actions:]

            # 策略频率：每 decimation 步更新一次高层动作
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_observations], dtype=np.float32) # 1x45

                keyboard_cmd_handler.update(kb_cmd)  # type: ignore
                cmd.vx = keyboard_cmd_handler.linear_x
                cmd.vy = keyboard_cmd_handler.linear_y
                cmd.dyaw = keyboard_cmd_handler.angular_z

                print(f"cmd_vx: {cmd.vx:.3f}, cmd_vy: {cmd.vy:.3f}, cmd_dyaw: {cmd.dyaw:.3f}")

                kb_cmd.zero()

                obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel

                obs[0, 3] = omega[0] * cfg.normalization.obs_scales.ang_vel
                obs[0, 4] = omega[1] * cfg.normalization.obs_scales.ang_vel
                obs[0, 5] = omega[2] * cfg.normalization.obs_scales.ang_vel

                obs[0, 6] = gvec[0]
                obs[0, 7] = gvec[1]
                obs[0, 8] = gvec[2]

                obs[0, 9:21]  = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos  # 关节角
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = last_actions  # 上次控制指令

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)  # 1x705
                if init==0:
                    print("init buf")
                    init=1
                    for i in range(cfg.env.num_obs_lens):  # 15
                        s = i * cfg.env.num_single_observations
                        e = (i + 1) * cfg.env.num_single_observations
                        policy_input[0, s:e] = obs
                else:
                    # 注意：deque 从左到右是老→新；如果你希望 0 段是最旧的，这里保持索引即可
                    for i in range(cfg.env.num_obs_lens):  # 15
                        s = i * cfg.env.num_single_observations
                        e = (i + 1) * cfg.env.num_single_observations
                        policy_input[0, s:e] = hist_obs[i][0, :]

                ort_inputs = {'input': policy_input}  # shape: (1, num_observations)
                action = policy.run(None, ort_inputs)[0].squeeze(0)  # 移除 batch 维度
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                last_actions = action
                action_count += 1

            # 低通滤波 & PD 控制（每步）
            action_flt = _low_pass_action_filter(action, action_flt, 0.05)

            target_q = 1 * action_flt * cfg.control.action_scales + default_dof_pos
            target_q[4]  = np.clip(target_q[4],  -0.8, 0.8)
            target_q[5]  = np.clip(target_q[5],  -0.4, 0.4)
            target_q[10] = np.clip(target_q[10], -0.8, 0.8)
            target_q[11] = np.clip(target_q[11], -0.4, 0.4)

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                             target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau

            mujoco.mj_step(model, data)
            count_lowlevel += 1

            # ---- 按墙钟节流，控制实时倍率 ----
            next_wall += wall_step
            now = time.perf_counter()
            sleep_s = next_wall - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # 仿真落后墙钟：追平，避免延迟积累
                next_wall = now
    finally:
        # 安全关闭（有的实现 close 是可选）
        try:
            if viewer is not None and hasattr(viewer, "close"):
                viewer.close()
        except Exception:
            pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='/home/user/ws/amp_humaniod/logs/miniloong_mlp/exported/policies/policy.onnx',
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default=False)
    parser.add_argument('--render-fps', type=int, default=50, help='Limit viewer FPS (e.g., 20)')
    parser.add_argument('--rt-speed', type=float, default=1.0, help='Real-time factor: 1.0=real-time, 0.5=half speed, 2.0=2x faster')
    args = parser.parse_args()

    class Sim2simCfg(LiteAmpCfg):

        class sim_config:
            if args.terrain:
                # mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v2_full/xml/world_terrain.xml'
                mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/xml/scene_terrain.xml'
            else:
                # mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v2_full/xml/scene_plane.xml'
                mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/xml/scene_plane.xml'
            sim_duration = 600.0
            dt = 0.001  # 1Khz 底层
            decimation = 20  # 100Hz 上层策略频率

        class robot_config:
            kps = np.array([150, 150, 150, 150,  40, 40,
                            150, 150, 150, 150,  40, 40], dtype=np.double)  # PD 和 Isaac 内部一致
            kds = np.array([5, 5, 5, 5, 2.5, 2.5,
                            5, 5, 5, 5, 2.5, 2.5], dtype=np.double)
            tau_limit = 300. * np.ones(12, dtype=np.double)  # Nm

    policy = ort.InferenceSession(args.load_model)
    run_mujoco(policy, Sim2simCfg(), render_fps=args.render_fps, rt_speed=args.rt_speed)
