# SPDX-License-Identifier: BSD-3-Clause
# LSTM-ONNX 部署（双状态：est 与 policy），固定 I/O 名称

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
# from humanoidGym.envs.lite.lite_config import LiteAmpCfg
from humanoidGym.envs.lite.lite_lstm_config import LiteAmpLSTMCfg as LiteAmpCfg
import torch
from humanoidGym import GYM_ROOT_DIR
import onnxruntime as ort
import argparse

# --------------------------- 固定 I/O 名的双 LSTM ONNX 封装器 ---------------------------

class OnnxDualLstmPolicy:
    """
    输入:  input, est_h0, est_c0, h0, c0
    输出:  output, est_hn, est_cn, hn, cn

    维护两套隐状态：(est_h, est_c) 与 (h, c)
    默认 (num_layers=1, hidden_size=256)；如不同，载入后调用 reset_states(...)。
    """
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.num_layers = 1
        self.hidden_size = 256
        self.reset_states()

        # 固定 I/O 名称（与你的导出一致）
        self._obs_name   = 'input'
        self._est_h0     = 'est_h0'
        self._est_c0     = 'est_c0'
        self._h0         = 'h0'
        self._c0         = 'c0'

        self._act_name   = 'output'
        self._est_hn     = 'est_hn'
        self._est_cn     = 'est_cn'
        self._hn         = 'hn'
        self._cn         = 'cn'

    def reset_states(self, num_layers=None, hidden_size=None):
        if num_layers is not None:
            self.num_layers = int(num_layers)
        if hidden_size is not None:
            self.hidden_size = int(hidden_size)
        self.h      = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
        self.c      = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
        self.est_h  = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
        self.est_c  = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)

    def step(self, obs_np):
        """
        obs_np: (1, obs_dim), float32
        返回: action (act_dim,)
        同时内部更新四个隐状态：est_h/est_c/h/c
        """
        if obs_np.dtype != np.float32:
            obs_np = obs_np.astype(np.float32, copy=False)

        inputs = {
            self._obs_name: obs_np,
            self._est_h0  : self.est_h,
            self._est_c0  : self.est_c,
            self._h0      : self.h,
            self._c0      : self.c,
        }
        outputs = self.sess.run(
            [self._act_name, self._est_hn, self._est_cn, self._hn, self._cn],
            inputs
        )
        action, est_hn, est_cn, hn, cn = outputs

        # 更新内部状态（保持 float32，形状 [L, 1, H]）
        self.est_h = np.array(est_hn, dtype=np.float32)
        self.est_c = np.array(est_cn, dtype=np.float32)
        self.h     = np.array(hn,     dtype=np.float32)
        self.c     = np.array(cn,     dtype=np.float32)

        return np.squeeze(action, axis=0)

# --------------------------- 其余逻辑与原脚本一致 ---------------------------

default_dof_pos = [0.39,-0.0,-0.12,0.74,0.36,0,0.39,-0.0,-0.12,0.74,0.36,0]

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def get_gravity_orientation(quaternion):
    qx, qy, qz, qw = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def quaternion_to_euler_array(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = get_gravity_orientation(quat).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions, last_actions, flt):
    return actions * flt + last_actions * (1 - flt)

def run_mujoco(policy, cfg):
    global default_dof_pos

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    tau = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    action_flt = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_actions = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_vel = np.zeros((3), dtype=np.double)

    hist_obs = deque(maxlen=cfg.env.num_obs_lens)
    for _ in range(cfg.env.num_obs_lens):
        hist_obs.append(np.zeros([1, cfg.env.num_single_observations], dtype=np.double))

    count_lowlevel = 0
    init = 0
    play_log = []
    action_count = 0
    stop_state_log = 500

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        q, dq, quat, v, omega, gvec = get_obs(data)
        acc = (v - last_vel) / cfg.sim_config.dt
        last_vel = v

        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 高层控制周期
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_observations], dtype=np.float32)

            # 简单命令时序
            if count_lowlevel < 2000:
                cmd.vx, cmd.vy, cmd.dyaw = 1.0, 0.0, 0.0
            elif count_lowlevel < 4000:
                cmd.vx, cmd.vy, cmd.dyaw = 0.0, 0.0, 0.0
            elif count_lowlevel < 6000:
                cmd.vx, cmd.vy, cmd.dyaw = -0.5, 0.0, 0.0
            elif count_lowlevel < 8000:
                cmd.vx, cmd.vy, cmd.dyaw = 0.0, 0.0, 0.0
            else:
                cmd.vx, cmd.vy, cmd.dyaw = 0.5, 0.0, 0.0

            # 单步观测（需与训练一致）
            obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel

            obs[0, 3] = omega[0] * cfg.normalization.obs_scales.ang_vel
            obs[0, 4] = omega[1] * cfg.normalization.obs_scales.ang_vel
            obs[0, 5] = omega[2] * cfg.normalization.obs_scales.ang_vel

            obs[0, 6] = gvec[0]
            obs[0, 7] = gvec[1]
            obs[0, 8] = gvec[2]

            obs[0, 9:21] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 33:45] = last_actions

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)

            # 历史拼接（与训练保持一致）
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            if init == 0:
                init = 1
                for i in range(cfg.env.num_obs_lens):
                    policy_input[0, i * cfg.env.num_single_observations : (i + 1) * cfg.env.num_single_observations] = obs
            else:
                for i in range(cfg.env.num_obs_lens):
                    policy_input[0, i * cfg.env.num_single_observations : (i + 1) * cfg.env.num_single_observations] = hist_obs[i][0, :]

            # ------------ LSTM ONNX 推理（自动维护 est/h/c） ------------
            action = policy.step(policy_input)
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            last_actions = action

            if action_count < stop_state_log:
                play_log.append(
                    [float(count_lowlevel * cfg.sim_config.dt)]
                    + target_q.tolist()
                    + q.tolist()
                    + dq.tolist()
                    + [cmd.vx, cmd.vy, cmd.dyaw]
                    + omega.tolist()
                    + gvec.tolist()
                    + action.tolist()
                    + tau.tolist()
                    + acc.tolist()
                )
            elif action_count == stop_state_log:
                print('play log saved')
                np.savetxt('../analysis/data/play_log_sim2sim.csv', play_log, delimiter=',')
            action_count += 1

        # 低通滤波 + PD 控制
        action_flt = _low_pass_action_filter(action, action_flt, 1.0)
        target_q = 1.0 * action_flt * cfg.control.action_scale + default_dof_pos
        target_q[4]  = np.clip(target_q[4],  -0.8, 0.8)
        target_q[5]  = np.clip(target_q[5],  -0.4, 0.4)
        target_q[10] = np.clip(target_q[10], -0.8, 0.8)
        target_q[11] = np.clip(target_q[11], -0.4, 0.4)

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau

        # 渲染 + 物理步进
        viewer.render()
        mujoco.mj_step(model, data)
        count_lowlevel += 1

    viewer.close()

# --------------------------- 入口 ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM-ONNX Deployment')
    parser.add_argument('--load_model', type=str,
                        default='/home/user/ws/amp_humaniod/logs/action_lag/exported/policies/policy.onnx',
                        help='Path to LSTM ONNX model.')
    parser.add_argument('--terrain', action='store_true', default=False)
    # 可选：隐状态形状（若与默认不同）
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='LSTM num layers')
    args = parser.parse_args()

    class Sim2simCfg(LiteAmpCfg):
        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/xml/scene_terrain.xml'
            else:
                mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/xml/scene_plane.xml'
            sim_duration = 60.0
            dt = 0.005  # 1Khz 底层
            decimation = 4  # 100Hz

        class robot_config:
            kps = np.array([150, 400, 150, 150, 30, 30,
                            150, 400, 150, 150, 30, 30], dtype=np.double)
            kds = np.array([8, 10, 8, 8, 4, 4,
                            8, 10, 8, 8, 4, 4], dtype=np.double)
            tau_limit = 100. * np.ones(12, dtype=np.double)

    # 实例化策略（若训练时 hidden_size/num_layers 不同，请覆盖）
    policy = OnnxDualLstmPolicy(args.load_model)
    if (args.hidden_size != 256) or (args.num_layers != 1):
        policy.reset_states(num_layers=args.num_layers, hidden_size=args.hidden_size)

    run_mujoco(policy, Sim2simCfg())
