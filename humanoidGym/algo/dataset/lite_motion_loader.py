import glob
import json

import numpy as np
import torch


def quat_rotate_inverse(q, v):
    """
    【核心新增】使用四元数旋转将速度从世界坐标系转换到局部坐标系
    输入：
        q: 局部坐标系相对于世界坐标系的旋转四元数（numpy数组，shape=(N,4)，格式：x,y,z,w）
        v: 世界坐标系下的速度向量（numpy数组，shape=(N,3)）
    输出：
        v_local: 局部坐标系下的速度向量（numpy数组，shape=(N,3)）
    """
    # 提取四元数的向量部分（x,y,z）和标量部分（w）
    q_vec = q[:, :3]  # (N,3)
    q_w = q[:, 3]     # (N,)
    
    # 按公式计算旋转后的局部速度
    # a = v * (2w² - 1)
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]  # (N,3)
    # b = cross(q_vec, v) * 2w
    b = np.cross(q_vec, v) * 2.0 * q_w[:, np.newaxis]  # (N,3)
    # c = q_vec * (2 * q_vec · v)
    dot_product = np.sum(q_vec * v, axis=1)[:, np.newaxis]  # (N,1)，点积结果
    c = q_vec * dot_product * 2.0  # (N,3)
    
    # 局部速度 = a - b + c
    return a - b + c

class LongAMPLoader:
    ###########################################################################
    
    TOTAL_DATA_SIZE = 39+6+1+7

    # 1. 重力向量（3维，新增）- 目标数据中位置：0~2
    GRAVITY_SIZE = 3
    GRAVITY_START_IDX = 0
    GRAVITY_END_IDX = GRAVITY_START_IDX + GRAVITY_SIZE  # 3



    # 2. 关节角度（12维）- 目标数据中位置：3~14
    JOINT_ANGLE_SIZE = 12
    JOINT_ANGLE_START_IDX = GRAVITY_END_IDX  # 3（接重力向量）
    JOINT_ANGLE_END_IDX = JOINT_ANGLE_START_IDX + JOINT_ANGLE_SIZE  # 15

    # 3. 关节角速度（12维）- 目标数据中位置：15~26
    JOINT_VEL_SIZE = 12
    JOINT_VEL_START_IDX = JOINT_ANGLE_END_IDX  # 15（接关节角度）
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE  # 27

    # 4. 末端位置（6维，toe_pos）- 目标数据中位置：27~32
    END_EFFECTOR_POS_SIZE = 6
    END_EFFECTOR_POS_START_IDX = JOINT_VEL_END_IDX  # 27（接关节角速度）
    END_EFFECTOR_POS_END_IDX = END_EFFECTOR_POS_START_IDX + END_EFFECTOR_POS_SIZE  # 33

    # 5. 末端姿态角（6维，toe_rpy）- 目标数据中位置：33~38
    END_EFFECTOR_RPY_SIZE = 6
    END_EFFECTOR_RPY_START_IDX = END_EFFECTOR_POS_END_IDX  # 33（接末端位置）
    END_EFFECTOR_RPY_END_IDX = END_EFFECTOR_RPY_START_IDX + END_EFFECTOR_RPY_SIZE  # 39

    # 6.base vel: lin_vel and ang_vel
    BASE_VEL_SIZE = 6
    BASE_VEL_START_IDX = END_EFFECTOR_RPY_END_IDX
    BASE_VEL_END_IDX = BASE_VEL_START_IDX + BASE_VEL_SIZE

    # base height
    BASE_HEIGHT_SIZE = 1
    BASE_HEIGHT_START_IDX = BASE_VEL_END_IDX
    BASE_HEIGHT_END_IDX = BASE_HEIGHT_START_IDX + BASE_HEIGHT_SIZE

    # base pose
    BASE_POSE_SIZE = 3
    BASE_POSE_START_IDX  = BASE_HEIGHT_END_IDX
    BASE_POSE_END_IDX = BASE_POSE_START_IDX + BASE_POSE_SIZE

    # base quat
    BASE_QUAT_SIZE = 4
    BASE_QUAT_START_IDX  = BASE_POSE_END_IDX
    BASE_QUAT_END_IDX = BASE_QUAT_START_IDX + BASE_QUAT_SIZE




    # 兼容旧接口（如需保留可启用，不影响新逻辑）
    # JOINT_POSE_SIZE = TOTAL_DATA_SIZE
    # JOINT_POSE_START_IDX = 0
    # JOINT_POSE_END_IDX = TOTAL_DATA_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """Expert dataset：加载39维目标数据（3重力+12关节角+12关节角速度+6末端位置+6末端姿态角）

        数据来源：从新格式frame_data中截取目标维度，拼接为指定顺序
        截取后内部结构（目标数据）：
        - 0~2：3维重力向量（gravity_proj）
        - 3~14：12关节角度
        - 15~26：12关节角速度
        - 27~32：6末端位置（toe_pos）
        - 33~38：6末端姿态角（toe_rpy）

        Args:
            device: 数据加载设备（CPU/GPU）
            time_between_frames: 帧间时间间隔（秒）
            data_dir: 数据目录（备用）
            preload_transitions: 是否预加载过渡数据
            num_preload_transitions: 预加载数据量
            motion_files: 运动数据文件路径列表
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # 轨迹存储变量（维度更新为39）
        self.trajectories = []  # 截取后39维数据
        self.trajectories_full = []  # 同trajectories（保持接口兼容）
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # 轨迹总时长（秒）
        self.trajectory_weights = []  # 轨迹采样权重
        self.trajectory_frame_durations = []  # 单帧时长（秒）
        self.trajectory_num_frames = []  # 轨迹总帧数

        # 加载数据并截取39维目标数据（核心修改：适配新格式索引）
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])  # 新格式原始数据（帧数×52维）
                
                ###################################################################
                # 核心修改：从新格式中分别截取目标数据各部分，按需求顺序拼接
                ###################################################################
                # 1. 重力向量（新格式索引7~10，3维）
                base_quat_data = motion_data[:,3:7]
                gravity_data = motion_data[:, 7:10]
                # 2. 关节角度（新格式索引16~28，12维）
                joint_angle_data = motion_data[:, 16:28]
                # 3. 关节角速度（新格式索引28~40，12维）
                joint_vel_data = motion_data[:, 28:40]
                # 4. 末端位置（新格式索引40~46，6维）
                end_pos_data = motion_data[:, 40:46]
                # 5. 末端姿态角（新格式索引46~52，6维）
                end_rpy_data = motion_data[:, 46:52]
                # 6. base vel
                base_lin_vel_data = motion_data[:,10:13]
                base_ang_vel_data = motion_data[:,13:16]

                # 7. base_height
                base_height_data = motion_data[:,2]

                # 8. root state
                root_state_data = motion_data[:,:7] 

                base_lin_vel_data = quat_rotate_inverse(base_quat_data, base_lin_vel_data)  # (帧数,3)
                base_ang_vel_data = quat_rotate_inverse(base_quat_data, base_ang_vel_data)  # (帧数,3)

                # 按需求顺序拼接：重力 → 关节角度 → 关节速度 → 末端位置 → 末端姿态
                target_data = np.concatenate(
                    [gravity_data, joint_angle_data, joint_vel_data, end_pos_data, end_rpy_data, base_lin_vel_data,base_ang_vel_data, base_height_data[:,np.newaxis],root_state_data],
                    axis=1  # 按列拼接（维度方向）
                )

                # 验证目标数据维度（修改：从36改为39）
                assert target_data.shape[1] == self.TOTAL_DATA_SIZE, \
                    f"目标数据维度错误！预期{self.TOTAL_DATA_SIZE}维，实际{target_data.shape[1]}维"

                # 加载到设备并存储（维度已更新为39）
                self.trajectories.append(
                    torch.tensor(target_data, dtype=torch.float32, device=device)
                )
                self.trajectories_full.append(
                    torch.tensor(target_data, dtype=torch.float32, device=device)
                )

                # 轨迹元信息（无修改，与数据格式无关）
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                # 轨迹总时长 = (帧数-1)×单帧时长
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(motion_data.shape[0])

            # 打印加载信息（修改：数据维度显示39）
            print(f"Loaded motion: {motion_file} | 时长: {traj_len:.2f}s | 帧数: {motion_data.shape[0]} | 数据维度: {target_data.shape[1]}")

        # 归一化轨迹采样权重（无修改）
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # 预加载过渡数据（s→s_next，维度自动变为39）
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"\nPreloading {num_preload_transitions} transitions...")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)

            # 预加载当前帧和下一帧（均为39维）
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f"Preload finished | preloaded_s shape: {self.preloaded_s.shape}")  # 输出格式：(num, 39)

        # 合并所有轨迹（备用，维度39）
        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    # ------------------------------ 以下为工具方法（无核心逻辑修改，仅适配维度）------------------------------
    def weighted_traj_idx_sample(self):
        """按权重采样单个轨迹索引（无修改）"""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """按权重批量采样轨迹索引（无修改）"""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """采样单个轨迹的随机时间（确保能取到下一帧，无修改）"""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0.0, self.trajectory_lens[traj_idx] * np.random.uniform() - subst)

    def traj_time_sample_batch(self, traj_idxs):
        """批量采样轨迹的随机时间（无修改）"""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        """线性插值（命名保留slerp以兼容旧接口，无修改）"""
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        """获取指定轨迹的39维数据（无修改，维度由TOTAL_DATA_SIZE控制）"""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """获取指定轨迹在指定时间的39维帧（线性插值，无修改）"""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]  # 轨迹总帧数
        idx_low = int(np.floor(p * n))
        idx_high = int(np.ceil(p * n))
        # 防止索引越界
        idx_high = min(idx_high, n - 1)
        idx_low = max(idx_low, 0)

        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """批量获取指定轨迹在指定时间的39维帧（无修改）"""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)
        idx_low = np.maximum(idx_low, 0)
        idx_high = np.minimum(idx_high, n - 1)

        batch_size = len(traj_idxs)
        # 维度从36改为39（由TOTAL_DATA_SIZE自动控制）
        all_frame_starts = torch.zeros(batch_size, self.TOTAL_DATA_SIZE, device=self.device)
        all_frame_ends = torch.zeros(batch_size, self.TOTAL_DATA_SIZE, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """获取指定轨迹在指定时间的完整39维帧（无修改）"""
        return self.get_frame_at_time(traj_idx, time)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """批量获取指定轨迹在指定时间的完整39维帧（无修改）"""
        return self.get_frame_at_time_batch(traj_idxs, times)

    def get_frame(self):
        """随机获取一个39维帧（无修改）"""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """随机获取一个完整39维帧（无修改）"""
        return self.get_frame()

    def get_full_frame_batch(self, num_frames):
        """批量获取随机39维帧（优先用预加载数据，无修改）"""
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)


    ###########################################################################
    # 数据获取接口（核心修改：新增重力接口+更新原有接口索引）
    ###########################################################################
    @property
    def observation_dim(self):
        """观察空间维度（固定为39，由TOTAL_DATA_SIZE控制）"""
        return self.TOTAL_DATA_SIZE-7

    @property
    def num_motions(self):
        """加载的运动轨迹数量（无修改）"""
        return len(self.trajectory_names)

    # ------------------------------ 新增：重力向量获取接口 ------------------------------
    @staticmethod
    def get_gravity(pose):
        """获取39维帧中的3维重力向量（0~2维）"""
        return pose[LongAMPLoader.GRAVITY_START_IDX : LongAMPLoader.GRAVITY_END_IDX]

    @staticmethod
    def get_gravity_batch(poses):
        """批量获取重力向量"""
        return poses[:, LongAMPLoader.GRAVITY_START_IDX : LongAMPLoader.GRAVITY_END_IDX]

    # ------------------------------ 更新：关节角度获取接口（索引3~15） ------------------------------
    @staticmethod
    def get_joint_angles(pose):
        """获取39维帧中的12关节角度（3~14维）"""
        return pose[LongAMPLoader.JOINT_ANGLE_START_IDX : LongAMPLoader.JOINT_ANGLE_END_IDX]

    @staticmethod
    def get_joint_angles_batch(poses):
        """批量获取关节角度"""
        return poses[:, LongAMPLoader.JOINT_ANGLE_START_IDX : LongAMPLoader.JOINT_ANGLE_END_IDX]

    # ------------------------------ 更新：关节角速度获取接口（索引15~27） ------------------------------
    @staticmethod
    def get_joint_velocities(pose):
        """获取维帧中的12关节角速度（15~26维）"""
        return pose[LongAMPLoader.JOINT_VEL_START_IDX : LongAMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_velocities_batch(poses):
        """批量获取关节角速度"""
        return poses[:, LongAMPLoader.JOINT_VEL_START_IDX : LongAMPLoader.JOINT_VEL_END_IDX]

    # ------------------------------ 更新：末端位置获取接口（索引27~33） ------------------------------
    @staticmethod
    def get_end_effector_positions(pose):
        """获取维帧中的6末端位置（27~32维）"""
        return pose[LongAMPLoader.END_EFFECTOR_POS_START_IDX : LongAMPLoader.END_EFFECTOR_POS_END_IDX]

    @staticmethod
    def get_end_effector_positions_batch(poses):
        """批量获取末端位置"""
        return poses[:, LongAMPLoader.END_EFFECTOR_POS_START_IDX : LongAMPLoader.END_EFFECTOR_POS_END_IDX]

    # ------------------------------ 更新：末端姿态角获取接口（索引33~39） ------------------------------
    @staticmethod
    def get_end_effector_rpy(pose):
        """获取维帧中的6末端姿态角（toe_rpy，33~38维）"""
        return pose[LongAMPLoader.END_EFFECTOR_RPY_START_IDX : LongAMPLoader.END_EFFECTOR_RPY_END_IDX]

    @staticmethod
    def get_end_effector_rpy_batch(poses):
        """批量获取末端姿态角"""
        return poses[:, LongAMPLoader.END_EFFECTOR_RPY_START_IDX : LongAMPLoader.END_EFFECTOR_RPY_END_IDX]
    

    @staticmethod
    def get_base_vel(pose):
        """获取维帧中的6末端姿态角（toe_rpy，33~38维）"""
        return pose[LongAMPLoader.BASE_VEL_START_IDX : LongAMPLoader.BASE_VEL_END_IDX]

    @staticmethod
    def get_base_vel_batch(poses):
        """批量获取末端姿态角"""
        return poses[:, LongAMPLoader.BASE_VEL_START_IDX : LongAMPLoader.BASE_VEL_END_IDX]
    

    @staticmethod
    def get_base_height(pose):
        """获取维帧中的6末端姿态角（toe_rpy，33~38维）"""
        return pose[LongAMPLoader.BASE_HEIGHT_START_IDX : LongAMPLoader.BASE_HEIGHT_END_IDX]

    @staticmethod
    def get_base_height_batch(poses):
        """批量获取末端姿态角"""
        return poses[:, LongAMPLoader.BASE_HEIGHT_START_IDX : LongAMPLoader.BASE_HEIGHT_END_IDX]
    
    @staticmethod
    def get_root_pos(pose):
        """获取维帧中的6末端姿态角（toe_rpy，33~38维）"""
        return pose[LongAMPLoader.BASE_POSE_START_IDX : LongAMPLoader.BASE_POSE_END_IDX]

    @staticmethod
    def get_root_pos_batch(poses):
        """批量获取末端姿态角"""
        return poses[:, LongAMPLoader.BASE_POSE_START_IDX : LongAMPLoader.BASE_POSE_END_IDX]
    
    @staticmethod
    def get_root_rot(poses):
        """批量获取末端姿态角"""
        return poses[LongAMPLoader.BASE_QUAT_START_IDX : LongAMPLoader.BASE_QUAT_END_IDX]

    @staticmethod
    def get_root_rot_batch(poses):
        """批量获取末端姿态角"""
        return poses[:, LongAMPLoader.BASE_QUAT_START_IDX : LongAMPLoader.BASE_QUAT_END_IDX]


    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """生成AMP训练用的过渡数据批次（s, s_next），均为39维（无修改，维度自动适配）"""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs,:-7]
                s_next = self.preloaded_s_next[idxs,:-7]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                s = self.get_frame_at_time_batch(traj_idxs, times)
                s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            yield s, s_next




