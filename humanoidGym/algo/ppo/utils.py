from typing import Callable
import torch
import importlib
import math
import numpy as np

_EPS = np.finfo(float).eps * 4.0

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # add at least one full length trajectory
    trajectories = trajectories + (torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),)
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )

def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name (str): The function name. The format should be 'module:attribute_name'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.

    Returns:
        Callable: The function loaded from the module.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)
    
def permute_with_sign(x: torch.Tensor, perm_sign: list) -> torch.Tensor:
    """
    对张量的最后一维进行带符号的置换操作
    Args:
        x: 输入张量 (..., N) 其中 N 是最后一个维度的长度
        perm_sign: 新位置的元素定义列表，例如 [3, -1, 0] 表示：
            - 新位置0 = 原位置3的元素（正号）
            - 新位置1 = 原位置1的元素（负号）
            - 新位置2 = 原位置0的元素（正号）
    Returns:
        变换后的张量，形状与输入相同
    """
    # 验证输入合法性
    assert len(perm_sign) == x.size(-1), "perm_sign长度必须与最后一维长度相同"
    assert all(abs(idx) < x.size(-1) for idx in perm_sign), "索引越界"

    # 构造变换矩阵
    n = x.size(-1)
    device = x.device
    dtype = x.dtype
    
    # 解析符号和原始索引
    transform = torch.zeros(n, n, dtype=dtype, device=device)
    for new_idx, item in enumerate(perm_sign):
        orig_idx = abs(item)  # 原始索引位置
        sign = 1 if item >= 0 else -1  # 符号
        
        # 确保索引有效性
        if orig_idx >= n:
            raise ValueError(f"无效索引 {item}，有效范围 [{-n+1}, {n-1}]")
        
        transform[new_idx, orig_idx] = sign

    # 应用矩阵乘法（自动处理batch维度）
    return torch.matmul(x, transform.T)  # 转置以匹配矩阵乘法维度

def generate_unitree_swap_pairs(dict_dofs):
    swap_pairs = []
    # 遍历所有关节名称
    for joint in dict_dofs:
        # 仅处理以'left_'开头的关节
        if joint.startswith('left_'):
            # 生成对应的右侧关节名称
            right_joint = joint.replace('left_', 'right_', 1)
            # 检查右侧关节是否存在
            if right_joint in dict_dofs:
                swap_pairs.append((joint, right_joint))
    return swap_pairs

def generate_long_swap_pairs(dict_dofs):
    swap_pairs = []
    # 遍历所有关节名称
    for joint in dict_dofs:
        # 仅处理以'left_'开头的关节
        if '_l_' in joint:
            # 生成对应的右侧关节名称
            right_joint = joint.replace('_l_', '_r_', 1)
            # 检查右侧关节是否存在
            if right_joint in dict_dofs:
                swap_pairs.append((joint, right_joint))
    print(swap_pairs)
    return swap_pairs
    
def swapPositions(ls, pos1, pos2, negative=None):
    if negative :
        ls[pos1], ls[pos2] = -ls[pos2], -ls[pos1]
    else:
        ls[pos1], ls[pos2] = ls[pos2], ls[pos1]
    return ls

def mirror_dof(start_idx,dict_dofs):
    swap_names = generate_long_swap_pairs(dict_dofs)
    dof_idxs = list(range(len(dict_dofs.keys())))
    dof_idxs = [idx + start_idx for idx in dof_idxs] 
    for _,(left,right) in enumerate(swap_names):
        if 'roll' in left :
            left_idx = dict_dofs[left]
            right_idx = dict_dofs[right]
            swapPositions(dof_idxs,left_idx,right_idx,True)
        elif 'yaw' in left :
            left_idx = dict_dofs[left]
            right_idx = dict_dofs[right]
            swapPositions(dof_idxs,left_idx,right_idx,True)
        else:
            left_idx = dict_dofs[left]
            right_idx = dict_dofs[right]
            swapPositions(dof_idxs,left_idx,right_idx,False)
    print(dof_idxs)
    return dof_idxs

def gen_mirror_ls(start_idx,name_type,dict_dofs):
    if name_type == "base_lin_vel":
        ls = [start_idx + idx for idx in list(range(3))]
        ls[1] = -ls[1]
        return ls
    elif name_type == "base_ang_vel":
        ls = [start_idx + idx for idx in list(range(3))]
        ls[0],ls[2] = -ls[0],-ls[2]
        return ls
    elif name_type == "projected_gravity":
        ls = [start_idx + idx for idx in list(range(3))]
        ls[1] = -ls[1]
        return ls
    elif name_type == "commands":
        ls = [start_idx + idx for idx in list(range(3))]
        ls[1] = -ls[1]
        return ls
    elif name_type == "phase":
        ls = [start_idx + idx for idx in list(range(2))]
        ls[0],ls[1] = ls[1],ls[0]
        return ls        
    elif name_type == "dofs":
        return mirror_dof(start_idx,dict_dofs)
    elif name_type == "stand_cmd":
        ls = [start_idx + idx for idx in list(range(1))]
        return ls
    else:
        raise ValueError(f"Invalid mirror obs type '{name_type}'")

def build_mirror_ls(dict_dofs,obs_list):
    mirror_ls = []
    start_idx = 0
    for part in obs_list:
        mirror_ls.extend(gen_mirror_ls(start_idx,part,dict_dofs))
        start_idx = len(mirror_ls) 
    return mirror_ls

def data_augmentation_func(obs: torch.Tensor, actions: torch.Tensor, env: object, is_critic: bool):
    #self.env.cfg.env.num_single_observation
    if obs is not None:
        # obs only 
        n,d = obs.size()
        obs_reshape = obs.reshape(-1,env.cfg.env.num_single_observations)
        obs_batch_mirror = permute_with_sign(obs_reshape,env.obs_mirror_ls)
        obs_batch_mirror = obs_batch_mirror.reshape(n,d)
        # append obs_batch
        obs_aug = torch.cat([obs,obs_batch_mirror],dim=0)
        return obs_aug,None
    elif actions is not None:
        # action only
        actions_batch_mirror = permute_with_sign(actions,env.action_mirror_ls)
        # append action batch
        actions_aug = torch.cat([actions,actions_batch_mirror],dim=0)
        return None,actions_aug
    else:
        raise ValueError('both obs and action is None')
    
def smooth_decay(current_step: int, start_step: int, decay_duration: int) -> float:
    if decay_duration <= 0:
        raise ValueError("decay_duration must be a positive integer")
    if current_step < start_step:
        return 1.0
    elif current_step >= start_step + decay_duration:
        return 0.0
    else:
        x = (current_step - start_step) / decay_duration
        return (math.cos(math.pi * x) + 1) / 2
    
def smooth_decay_se(
    current_step: int,
    start_step: int,
    decay_duration: int,
    start_value: float = 1.0,
    end_value: float = 0.0
) -> float:
    """
    带起始/结束值控制的平滑衰减函数
    
    Args:
        current_step: 当前时间步
        start_step: 衰减开始时间步
        decay_duration: 衰减持续时间步数（必须为正）
        start_value: 衰减起始值（默认1.0）
        end_value: 衰减结束值（默认0.0）
    
    Returns:
        当前时间步对应的插值结果
    """
    if decay_duration <= 0:
        raise ValueError("decay_duration必须为正整数")
    
    if current_step < start_step:
        return start_value
    elif current_step >= start_step + decay_duration:
        return end_value
    else:
        # 计算衰减进度比例
        progress = (current_step - start_step) / decay_duration
        # 应用余弦插值
        return start_value + (end_value - start_value) * (1 - (math.cos(math.pi * progress) + 1) / 2)

def smooth_transition(
    current_step: int,
    start_step: int,
    transition_duration: int,
    start_value: float,
    end_value: float
) -> float:
    """
    在指定时间窗口内平滑地从起始值过渡到结束值
    
    参数:
        current_step: 当前时间步
        start_step: 过渡开始的时间步
        transition_duration: 过渡持续的时间步数
        start_value: 过渡开始时的值
        end_value: 过渡结束时的值
    
    返回:
        当前时间步对应的插值结果
    """
    # 验证参数有效性
    if transition_duration <= 0:
        raise ValueError("transition_duration必须为正整数")
    
    # 在开始时间之前保持起始值
    if current_step < start_step:
        return start_value
    
    # 在过渡期之后保持结束值
    if current_step >= start_step + transition_duration:
        return end_value
    
    # 在过渡期间执行平滑插值
    progress = (current_step - start_step) / transition_duration
    # 使用余弦函数确保平滑过渡
    # 这里使用1减去余弦函数，因为我们需要从0→1的平滑增长因子
    weight = (1 - math.cos(math.pi * progress)) / 2
    
    # 线性插值：在起始值和结束值之间过渡
    return start_value + (end_value - start_value) * weight


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out
