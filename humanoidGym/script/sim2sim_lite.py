# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoidGym.envs.lite.lite_config import LiteAmpCfg
import torch
from humanoidGym import GYM_ROOT_DIR
import onnxruntime as ort  # 导入 ONNX Runtime

default_dof_pos = [0.39,-0.0,-0.12,0.74,0.36,0,0.39,-0.0,-0.12,0.74,0.36,0]


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

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

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
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
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions,last_actions,flt):
 
    actons_filtered = actions * flt + last_actions * (1-flt)
    return actons_filtered

def run_mujoco(policy, cfg):
    global default_dof_pos
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)#载入初始化位置由XML决定
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)     # 10
    tau = np.zeros((cfg.env.num_actions), dtype=np.double)     # 10
    action = np.zeros((cfg.env.num_actions), dtype=np.double)       # 10
    action_flt = np.zeros((cfg.env.num_actions), dtype=np.double)   # 10
    last_actions = np.zeros((cfg.env.num_actions), dtype=np.double) # 10
    last_vel = np.zeros((3), dtype=np.double)
    hist_obs = deque(maxlen=cfg.env.num_obs_lens)
    for _ in range(cfg.env.num_obs_lens):
        hist_obs.append(np.zeros([1, cfg.env.num_single_observations], dtype=np.double)) # 39

    count_lowlevel = 0
    init=0
    play_log = []
    action_count = 0
    stop_state_log = 500
    phase = 0
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)#从mujoco获取仿真数据
        acc = (v-last_vel)/cfg.sim_config.dt
        last_vel = v

        q = q[-cfg.env.num_actions:]

        dq = dq[-cfg.env.num_actions:]

        
        if 1:
            # 1000hz ->50hz
            if count_lowlevel % cfg.sim_config.decimation == 0:

                obs = np.zeros([1, cfg.env.num_single_observations], dtype=np.float32) #1,45
            
                if count_lowlevel<2000:
                    cmd.vx=1
                    cmd.vy=0
                    cmd.dyaw= 0
                elif count_lowlevel<4000:
                    cmd.vx=0.0
                    cmd.vy=0
                    cmd.dyaw= 0
                elif count_lowlevel<6000:
                    cmd.vx=-0.5
                    cmd.vy=0
                    cmd.dyaw= 0
                elif count_lowlevel<8000:
                    cmd.vx=0.0
                    cmd.vy=0
                    cmd.dyaw= 0
                else:
                    cmd.vx=0.5
                    cmd.vy=0
                    cmd.dyaw= 0

             
                obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
               
                obs[0, 3] = omega[0] *cfg.normalization.obs_scales.ang_vel
                obs[0, 4] = omega[1] *cfg.normalization.obs_scales.ang_vel
                obs[0, 5] = omega[2] *cfg.normalization.obs_scales.ang_vel
                if 1:
                    obs[0, 6] = gvec[0] #*cfg.normalization.obs_scales.quat
                    obs[0, 7] = gvec[1] #*cfg.normalization.obs_scales.quat
                    obs[0, 8] = gvec[2] #*cfg.normalization.obs_scales.quat
           
                obs[0, 9:21] = (q-default_dof_pos) * cfg.normalization.obs_scales.dof_pos #g关节角度顺序依据修改为样机
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = last_actions#上次控制指令
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                #print(cfg.normalization.obs_scales.dof_pos,cfg.normalization.obs_scales.dof_vel,cfg.normalization.obs_scales.quat,cfg.normalization.obs_scales.ang_vel)
                hist_obs.append(obs)
                # hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32) #1,705,和isaac一致
                if init==0:
                    print("init buf")
                    init=1
                    for i in range(cfg.env.num_obs_lens):#15
                        policy_input[0, i * cfg.env.num_single_observations : (i + 1) * cfg.env.num_single_observations] = obs
                else:
                    for i in range(cfg.env.num_obs_lens):#15
                        policy_input[0, i * cfg.env.num_single_observations : (i + 1) * cfg.env.num_single_observations] = hist_obs[i][0, :]

                # print("obs_tensor shape:", policy_input.shape)  # 应该是 (1, num_observations)
                ort_inputs = {
                    'input': policy_input  # shape: (1, num_observations)
                }
                action = policy.run(None, ort_inputs)[0].squeeze(0)  # 移除batch维度

                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
         
                last_actions = action

                if action_count < stop_state_log:
                    play_log.append(
                        [float(count_lowlevel*cfg.sim.dt)]
                        + target_q.tolist()
                        + q.tolist()
                        + dq.tolist()
                        + [cmd.vx,cmd.vy,cmd.dyaw]
                        + omega.tolist()
                        + gvec.tolist()
                        + action.tolist()
                        + tau.tolist()
                        + acc.tolist()
                    )
                elif action_count==stop_state_log:
                    print('play log saved')
                    np.savetxt('../analysis/data/play_log_sim2sim.csv', play_log, delimiter=',')

                action_count = action_count+1

                # lcm->mujoco
                # print("action_flt:")
                #print("action_flt:",action_flt)
                #print("q:",q)
                #print("target_q:",target_q)
            # cfg.control.exp_avg_decay = cfg.sim_config.dt * 50
            action_flt=_low_pass_action_filter(action,action_flt,1)            

            target_q = 1 * action_flt * cfg.control.action_scale + default_dof_pos
            target_q[4] = np.clip(target_q[4],-0.8,0.8)
            target_q[5] = np.clip(target_q[5],-0.4,0.4)
            target_q[10] = np.clip(target_q[10],-0.8,0.8)
            target_q[11] = np.clip(target_q[11],-0.4,0.4)
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                             target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau
            viewer.render()
        mujoco.mj_step(model, data)
       
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='/home/user/ws/amp_humaniod/logs/miniloong/exported/policies/policy.onnx',
                        help='Run to load from.')
    # parser.add_argument('--load_model', type=str, default='/home/tim/github/Miniloong/miniloong-humanoid-yu/logs/miniloong_july/exported/policies/lite_yuhan.pt',
    #                      help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default=False)
    args = parser.parse_args()

    class Sim2simCfg(LiteAmpCfg):

        class sim_config:
            if args.terrain:
                # mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v2_full/xml/world_terrain.xml'
                mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/xml/scene_terrain.xml'
            else:
                # mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v2_full/xml/scene_plane.xml'
                mujoco_model_path = f'{GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/xml/scene_plane.xml'
            sim_duration = 60.0
            dt = 0.005 #1Khz底层
            decimation = 4 # 100Hz

        class robot_config:
            # kps = np.array([200, 300, 200, 200,  30, 30, \
            #                 200, 300, 200, 200,  30, 30], dtype=np.double)#PD和isacc内部一致
            # kds = np.array([8, 10, 8, 8, 4, 4, \
            #                 8, 10, 8, 8, 4, 4], dtype=np.double)
            kps = np.array([150, 400, 150, 150,  30, 30, \
                            150, 400, 150, 150,  30, 30], dtype=np.double)#PD和isacc内部一致
            kds = np.array([8, 10, 8, 8, 4, 4, \
                            8, 10, 8, 8, 4, 4,], dtype=np.double)
            # kps = np.array([200, 300, 200, 200,  40, 40, \
            #                 200, 300, 200, 200,  40, 40], dtype=np.double)#PD和isacc内部一致
            # kds = np.array([10, 6, 6, 10, 4, 4, \
            #                 10, 6, 6, 10, 4, 4,], dtype=np.double)
            tau_limit = 100. * np.ones(12, dtype=np.double)#nm

    # policy = torch.jit.load(args.load_model)# 有一个可能得原因是这个 pt文件里只有权重，而没有网络结构。 所以 只能用 torch.load去加载，不能用torch.jit.load
    policy = ort.InferenceSession(args.load_model)  # 使用 ONNX Runtime 加载模型
    run_mujoco(policy, Sim2simCfg())