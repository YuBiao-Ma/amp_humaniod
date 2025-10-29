import os
import json
import numpy as np
import torch
import joblib  # 用 joblib.load 读

import types
import sys
import numpy as np

if not hasattr(np, "_core"):
    fake_core = types.ModuleType("numpy._core")
    import numpy.core as np_core
    for attr in dir(np_core):
        setattr(fake_core, attr, getattr(np_core, attr))
    sys.modules["numpy._core"] = fake_core

# --- 配置输入输出路径 ---
src_path = "/home/user/ws/amp_humaniod/humanoidGym/datasets/obstacles2_subject2_stair_up.pkl"
dst_path = "/home/user/ws/amp_humaniod/humanoidGym/datasets/obstacles2_subject2_stair_up.json"

# --- 读取 joblib pkl ---
data = joblib.load(src_path)  # 这一步会还原成 dict / numpy arrays / etc.

# --- 把 data 变成 JSON 可序列化的纯 Python 基础类型 ---
def to_json_safe(obj):
    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy 标量 -> Python 标量
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_ ,)):
        return bool(obj)

    # torch.Tensor -> list
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    # dict -> 递归转换
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    # list / tuple -> 递归转换
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    # 其他对象：如果有 __dict__，尝试用字段展开
    if hasattr(obj, "__dict__"):
        return to_json_safe(obj.__dict__)

    # 最后兜底：转成字符串（防止 json.dump 卡死）
    return str(obj)

safe_data = to_json_safe(data)

# --- 写出成 JSON 文件 ---
with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(safe_data, f, ensure_ascii=False, indent=2)

print(f"done -> {dst_path}")
