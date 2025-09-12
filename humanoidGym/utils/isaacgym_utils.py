import os
import numpy as np
import random
import torch
from typing import List, Tuple

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def random_quat(U):
    u1 = U[:,0].unsqueeze(1)
    u2 = U[:,1].unsqueeze(1)
    u3 = U[:,2].unsqueeze(1)
    q1 = torch.sqrt(1-u1)*torch.sin(2*torch.pi*u2)
    q2 = torch.sqrt(1-u1)*torch.cos(2*torch.pi*u2)
    q3 = torch.sqrt(u1)*torch.sin(2*torch.pi*u3)
    q4 = torch.sqrt(u1)*torch.cos(2*torch.pi*u3)
    Q = torch.cat([q1,q2,q3,q4],dim=-1)
    return Q

def torch_rand_float_piecewise(segments: List[Tuple[float, float, float]],
                               shape: Tuple[int, ...],
                               device: torch.device) -> torch.Tensor:
    """
    生成分段均匀分布的随机浮点数张量
    
    参数:
    segments - 分段配置列表，每个元素为元组 (min, max, probability_weight)
               例如: [(0.5, 1.0, 0.4), (1.0, 2.0, 0.3), (2.0, 5.0, 0.3)]
               min: 分段最小值
               max: 分段最大值
               probability_weight: 该分段的概率权重 (无需归一化)
    shape - 输出张量的形状 (如 (100, 12) 表示100个环境，每个环境12个关节)
    device - 张量所在的设备 ('cpu' 或 'cuda')
    
    返回:
    Tensor - 指定形状的分段均匀随机数张量
    
    示例:
    # 创建分三段采样的随机数:
    #   40% 在 [0.5, 1.0) 区间
    #   30% 在 [1.0, 2.0) 区间
    #   30% 在 [2.0, 5.0] 区间
    segments = [
        (0.5, 1.0, 0.4),
        (1.0, 2.0, 0.3),
        (2.0, 5.0, 0.3)
    ]
    damping_values = torch_rand_float_piecewise(
        segments, 
        (num_envs, num_dofs), 
        device
    )
    """
    # 验证输入有效性
    if not segments:
        raise ValueError("Segments list cannot be empty")
    
    min_vals, max_vals, weights = zip(*segments)
    total_samples = torch.prod(torch.tensor(shape)).item()
    
    # 计算每个分段的实际采样数量 (基于权重)
    weights_tensor = torch.tensor(weights, device=device)
    probs = weights_tensor / weights_tensor.sum()
    segment_counts = (probs * total_samples).long()
    
    # 处理可能因四舍五入导致的计数差异
    diff = total_samples - segment_counts.sum().item()
    if diff != 0:
        # 将差异添加到权重最大的分段
        max_index = torch.argmax(weights_tensor).item()
        segment_counts[max_index] += diff
    
    # 为每个分段生成样本
    samples = []
    for (min_val, max_val), count in zip(zip(min_vals, max_vals), segment_counts):
        if count > 0:
            # 在分段内均匀采样
            segment_samples = torch.rand(count.item(), device=device) * (max_val - min_val) + min_val
            samples.append(segment_samples)
    
    # 合并所有样本并重塑为所需形状
    full_samples = torch.cat(samples)[:total_samples]  # 确保不会超过所需数量
    return full_samples.reshape(shape)


