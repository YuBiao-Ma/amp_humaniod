import json
import numpy as np
import sys
from pathlib import Path

# ------------- helpers for math -----------------

def finite_diff(data, dt):
    """
    data: [T, D] numpy array
    returns vel: [T, D] forward difference / dt,
    with vel[0] duplicated from vel[1] for continuity.
    """
    T, D = data.shape
    vel = np.zeros_like(data, dtype=np.float32)
    if T > 1:
        vel[1:] = (data[1:] - data[:-1]) / dt
        vel[0] = vel[1]
    return vel


def quat_mul(q1, q2):
    """
    Hamilton product of two quats.
    q1,q2: [...,4] assumed [w,x,y,z]
    return: [...,4] [w,x,y,z]
    """
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.concatenate([w, x, y, z], axis=-1)


def quat_inv(q):
    """
    Inverse of unit quaternion.
    q: [...,4] [w,x,y,z]
    For unit quats, inverse is conjugate.
    """
    w, x, y, z = np.split(q, 4, axis=-1)
    return np.concatenate([w, -x, -y, -z], axis=-1)


def quat_to_angular_vel(q, dt):
    """
    Approximate angular velocity from discrete quaternion trajectory.

    q: [T,4] unit quats, [w,x,y,z]
    dt: float (frame time)

    Steps:
    - dq_t = q_t * inv(q_{t-1})
    - convert dq_t to axis-angle
    - omega_t = axis * (angle / dt)

    Returns [T,3], with ang_vel[0] = ang_vel[1].
    """
    T = q.shape[0]
    ang_vel = np.zeros((T, 3), dtype=np.float32)
    if T <= 1:
        return ang_vel

    q_prev = q[:-1]
    q_curr = q[1:]

    dq = quat_mul(q_curr, quat_inv(q_prev))  # [T-1,4]
    dq = dq / np.linalg.norm(dq, axis=-1, keepdims=True)

    w = dq[:, 0]
    xyz = dq[:, 1:]

    w_clamped = np.clip(w, -1.0, 1.0)

    angle = 2.0 * np.arccos(w_clamped)      # [T-1]
    sin_half = np.sqrt(1.0 - w_clamped**2)  # = sin(angle/2)

    axis = np.zeros_like(xyz)
    small_mask = sin_half < 1e-8

    if np.any(~small_mask):
        axis[~small_mask] = xyz[~small_mask] / sin_half[~small_mask, None]

    # small-angle fallback
    if np.any(small_mask):
        axis[small_mask] = xyz[small_mask] * (2.0 / dt)

    omega = axis * (angle[:, None] / dt)    # [T-1,3]

    ang_vel[1:] = omega.astype(np.float32)
    ang_vel[0] = ang_vel[1]

    return ang_vel


def drop_indices(arr, drop_ids):
    """
    arr: [T, J] numpy
    drop_ids: list[int] (0-based)
    returns:
      arr_new: [T, J'] with columns removed
      keep_mask: boolean mask of kept joints
    """
    all_idx = np.arange(arr.shape[1])
    keep_mask = ~np.isin(all_idx, np.array(drop_ids))
    return arr[:, keep_mask], keep_mask


# ------------- frame slicing helper -----------------

def slice_and_concat(arr, ranges):
    """
    arr: [T, D] numpy array
    ranges: list of (start, end) with python slicing semantics [start:end]

    We take each slice arr[start:end] and stack them in the given order.
    Returns [T_concat, D]
    """
    clips = [arr[s:e] for (s, e) in ranges]
    if len(clips) == 1:
        return clips[0]
    return np.concatenate(clips, axis=0)


# ------------- main pipeline -----------------

def convert_motion(in_path, out_path):
    # 1. 读原始json
    with open(in_path, "r") as f:
        raw = json.load(f)

    # 2. 基础量
    fps = float(raw["fps"])
    dt = 1.0 / fps  # FrameDuration

    # 这里字段名根据你最新脚本：
    # - dof_pos           -> 关节角
    # - local_toe_pos     -> 脚端位置 (局部系/期望你当成foot_pos)
    # - local_tar_pos     -> 脚的target位置
    root_pos  = np.array(raw["root_pos"],        dtype=np.float32)  # [T,3]
    root_rot  = np.array(raw["root_rot"],        dtype=np.float32)  # [T,4]
    joint_pos = np.array(raw["dof_pos"],         dtype=np.float32)  # [T,J]
    toe_pos   = np.array(raw["local_toe_pos"],   dtype=np.float32)  # [T,F*3]
    tar_pos   = np.array(raw["local_tar_pos"],   dtype=np.float32)  # [T,3] (单一目标点或摆动脚目标)

    # 3. 先根据给定的帧区间，把这些轨迹裁出来并按顺序拼接
    clip_ranges = [
        # (760, 810),
        # (490, 540),
        # (2140, 2210),
        (0, root_pos.shape[0]),
    ]

    root_pos  = slice_and_concat(root_pos,  clip_ranges)
    root_rot  = slice_and_concat(root_rot,  clip_ranges)
    joint_pos = slice_and_concat(joint_pos, clip_ranges)
    toe_pos   = slice_and_concat(toe_pos,   clip_ranges)
    tar_pos   = slice_and_concat(tar_pos,   clip_ranges)

    # 现在 T 变成拼接后总长度
    T = root_pos.shape[0]

    # 4. 删掉指定关节索引
    DROP_JOINTS = [13,14,19,20,21,26,27,28]
    joint_pos_filt, keep_mask = drop_indices(joint_pos, DROP_JOINTS)

    # 5. 计算速度
    # base 线速度
    lin_vel = finite_diff(root_pos, dt)        # [T,3]

    # base 角速度（由四元数差分近似得到）
    ang_vel = quat_to_angular_vel(root_rot, dt)  # [T,3]

    # 关节速度，用差分后再按keep_mask过滤
    joint_vel_full = finite_diff(joint_pos, dt)      # [T,J]
    joint_vel_filt = joint_vel_full[:, keep_mask]    # [T,J']

    # 脚当前轨迹速度
    foot_vel = finite_diff(toe_pos, dt)              # [T,F*3]

    # 脚target速度（比如摆动轨迹的期望移动速度）
    foot_vel2 = finite_diff(tar_pos, dt)             # [T,3]

    # 6. 拼每一帧的特征向量
    # 顺序：
    #   root_pos(3)
    #   root_rot(4)
    #   joint_pos_filtered(J')
    #   toe_pos(F*3)
    #   tar_pos(3)
    #   lin_vel(3)
    #   ang_vel(3)
    #   joint_vel_filtered(J')
    #   foot_vel(F*3)
    #   foot_vel2(3)
    frames = []
    for t in range(T):
        frame_vec = np.concatenate([
            root_pos[t],
            root_rot[t],
            joint_pos_filt[t],
            toe_pos[t],
            tar_pos[t],
            lin_vel[t],
            ang_vel[t],
            joint_vel_filt[t],
            foot_vel[t],
            foot_vel2[t],
        ], axis=0)
        frames.append(frame_vec.tolist())

    # 7. 组最终导出的字典
    out_dict = {
        "LoopMode": "Wrap",
        "FrameDuration": dt,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 1,
        "Frames": frames,
    }

    # 8. 写出到 out_path
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)


# ------------- CLI entry -----------------

if __name__ == "__main__":
    # 你可以像这样直接跑，也可以走命令行参数
    in_path = '/home/user/ws/amp_humaniod/humanoidGym/datasets/g1/mocap_motions/obstacles1_subject2_right_jump.json'
    out_path = '/home/user/ws/amp_humaniod/humanoidGym/datasets/g1/mocap_motions/obstacles1_subject2_right_jump.json'

    convert_motion(in_path, out_path)
    print(f"done -> {out_path}")
