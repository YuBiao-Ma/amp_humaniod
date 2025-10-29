# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from humanoidGym.utils import trimesh
from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy.ndimage import binary_dilation


class TerrainParkour:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.added_trimesh = None
        if cfg.curriculum:
            self.curriculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.evaluated_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                             self.cfg.horizontal_scale,
                                                                                             self.cfg.vertical_scale,
                                                                                             self.cfg.slope_treshold)

            if self.added_trimesh is not None:
                self.vertices, self.triangles = trimesh.combine_trimeshes(
                    (self.vertices, self.triangles),
                    self.added_trimesh,
                )

            half_edge_width = int(1)
            structure = np.ones((half_edge_width * 2 + 1, 1))
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)


    def selected_terrain(self):
        
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)
            
            
          
            
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
            if i == 0:
                climb_terrain(terrain, depth=0.5, platform_size=4.)
            if i == 2:
                pyramid_stairs_terrain(terrain, step_width=0.33, step_height=0.15, platform_size=3.)
            
            if i == 1:
                gap_terrain(terrain, gap_size=0.5, platform_size=4.)
                
            # if i == 3:
            #     balance_beam_terrain(
            #     terrain, 0.5,
            #     corridor_w=2, margin=0.0,
            #     beam_w_easy=0.7,   # 难度=0 时的木宽(米)
            #     beam_w_hard=0.25,   # 难度=1 时的木宽(米)  ≥ 足底宽的 ~1.2×
            #     pit_h_easy=10,    # 坑深(米)
            #     pit_h_hard=10,
            #     notch_w=0.0,       # 中央缺口的宽度(米)
            # )
            if i ==4:
                terrain_utils.pyramid_sloped_terrain(terrain, slope=0.5, platform_size=3.)
                # terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.35, step_height=0.18, platform_size=3.)
            # if i == 0:
            #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.35, step_height=0.15, platform_size=3.)
           
            self.add_terrain_to_map_myb(terrain, i, j)
    
    def add_terrain_to_map_myb(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = 0 - (0.2) * self.env_width
        env_origin_y = (0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, 0]


    def evaluated_terrain(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

    def curriculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

    # def selected_terrain(self):
    #     terrain_type = self.cfg.terrain_kwargs.pop('type')
    #     for k in range(self.cfg.num_sub_terrains):
    #         # Env coordinates in the world
    #         (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

    #         terrain = terrain_utils.SubTerrain("terrain",
    #                                            width=self.width_per_env_pixels,
    #                                            length=self.width_per_env_pixels,
    #                                            vertical_scale=self.vertical_scale,
    #                                            horizontal_scale=self.horizontal_scale)

    #         eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
    #         self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty, i = 0, j = 0):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        amplitude = 0.1 + 0.2 * difficulty
        slope = difficulty * 0.8 + 0.1
        step_height = 0.05 + 0.3 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.5
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 0.8 * difficulty + 0.05
        tilt_width = 0.52 - 0.04 * difficulty
        stair_step_width = 0.30 + random.random() * 0.04
        if choice < self.proportions[0]:
            terrain_utils.wave_terrain(terrain, num_waves=5, amplitude=amplitude)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[1]:
            if choice < (self.proportions[0] + self.proportions[1]) / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            pyramid_stairs_terrain(terrain, step_width=stair_step_width, step_height=step_height,
                                                 platform_size=1.)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
            #                                      downsampled_scale=0.2)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[5]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=4.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[6]:
            climb_terrain(terrain, depth=pit_depth, platform_size=4.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)

        elif choice < self.proportions[7]:
            # discrete_blocks_terrain(terrain, difficulty,
            #                 corridor_w=2, margin=0.0,
            #                 size_easy=0.9, size_hard=0.25,
            #                 gap_easy=0.05,  gap_hard=0.15,
            #                 h_easy=10,    h_hard=10,
            #                 stagger=True, p_drop=0.0)
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)



        elif choice < self.proportions[8]:
            # 平衡木：难度越大越窄
            balance_beam_terrain(
                terrain, difficulty,
                corridor_w=2, margin=0.0,
                beam_w_easy=0.7,   # 难度=0 时的木宽(米)
                beam_w_hard=0.25,   # 难度=1 时的木宽(米)  ≥ 足底宽的 ~1.2×
                pit_h_easy=10,    # 坑深(米)
                pit_h_hard=10,
                notch_w=0.0,       # 中央缺口的宽度(米)
            )
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)

        else:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        return terrain


    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5 ) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def balance_beam_terrain(terrain, difficulty,
                         corridor_w=0.4, margin=0.0,
                         beam_w_easy=0.35, beam_w_hard=0.12,
                         pit_h_easy=0.25,  pit_h_hard=0.45,
                         notch_w=0.0):
    # 参数插值
    t = float(np.clip(difficulty, 0.0, 1.0))
    lerp = lambda a, b: a + (b - a) * t
    beam_w = lerp(beam_w_easy, beam_w_hard)
    pit_h  = lerp(pit_h_easy,  pit_h_hard)

    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    hf = terrain.height_field_raw
    H, W = hf.shape

    # 像素化
    px_beam   = max(1, int(round(beam_w   / hs)))
    px_corr   = max(1, int(round(corridor_w / hs)))
    px_notch  = max(0, int(round(abs(notch_w) / hs)))
    pit_val   = -int(round(pit_h / vs))

    # 全部设坑
    hf[:, :] = pit_val

    # === 中央走廊（横向带，居中于 center_x） ===
    center_x = H // 2
    xLcorr = max(0, center_x - px_corr // 2)
    xRcorr = min(H, center_x + (px_corr + 1) // 2)
    hf[xLcorr:xRcorr, :] = 0

    # === 横向平衡木（建议与走廊同中心；如需偏移可加 jitter） ===
    x_center = center_x
    xL = max(0, x_center - px_beam // 2)
    xR = min(H, x_center + (px_beam + 1) // 2)
    hf[:,xL:xR] = 0  # 注意：横向必须是 hf[xL:xR, :]

    # 可选：在横梁中心挖缺口
    if px_notch > 0:
        center_y = W // 2
        yL = max(0, center_y - px_notch // 2)
        yR = min(W, center_y + (px_notch + 1) // 2)
        hf[xL:xR, yL:yR] = pit_val


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    """
    生成“上楼梯 → 中心平台 → 下楼梯”的走廊地形，并把左右两侧设为深坑防绕路。

    几何结构 (沿 x 前进方向):
        [上楼梯(逐级升高)] [平台(恒高)] [下楼梯(逐级降低)]

    参数 (米):
        step_width (float):    每一级台阶在 x 方向的长度
        step_height (float):   每一级台阶相对前一级增加/减少的高度
        platform_size (float): 中间平台在 x 方向的长度
    """

    # -------- 连续量(米) -> height_field_raw 索引单位 --------
    step_width_idx    = int(step_width    / terrain.horizontal_scale)
    step_height_idx   = int(step_height   / terrain.vertical_scale)
    platform_len_idx  = int(platform_size / terrain.horizontal_scale)

    # 防止出现 0 导致死循环
    step_width_idx   = max(step_width_idx,   1)
    step_height_idx  = max(step_height_idx,  1)
    platform_len_idx = max(platform_len_idx, 1)

    # -------- 定义中间走廊，左右两边挖深坑以防绕路 --------
    center_y = terrain.width // 2

    # 走廊半宽（米）：和 climb_terrain 保持同一风格，稍微随机，避免策略死记
    width_rand = 1.0 + 1.0 * np.random.random()
    half_corridor = 2.0 * width_rand  # 走廊半宽(米)

    y1 = int(center_y - half_corridor / terrain.horizontal_scale)
    y2 = int(center_y + half_corridor / terrain.horizontal_scale)

    # 边界裁剪
    y1 = max(y1, 0)
    y2 = min(y2, terrain.height_field_raw.shape[1])

    # 把走廊外的区域变成深坑（禁止绕行）
    terrain.height_field_raw[:, :y1] = -1000
    terrain.height_field_raw[:, y2:] = -1000

    # -------- 生成楼梯剖面 (沿 x 方向对称：上→平台→下) --------
    total_len_x = terrain.length  # x 方向一共有多少格
    # 平台长度不能超过整段，否则后面就没“下楼梯”空间了
    platform_len_idx = min(platform_len_idx, total_len_x)

    # 我们把整段 x 分成三段：
    # [0            : left_len )   -> 上楼梯 (高度递增)
    # [left_len     : plat_end )   -> 平台 (恒定最高高度)
    # [plat_end     : total_len_x) -> 下楼梯 (高度递减)

    # 先把平台放在整个走廊的“中间”
    left_len = (total_len_x - platform_len_idx) // 2
    right_len = total_len_x - platform_len_idx - left_len
    x_left_start = 0
    x_left_end   = left_len
    x_plat_start = x_left_end
    x_plat_end   = x_plat_start + platform_len_idx
    x_right_start = x_plat_end
    x_right_end   = total_len_x  # = x_plat_end + right_len

    # -------- 左半段：上楼梯 (高度逐级升高) --------
    # 我们把 [0 : left_len) 按 step_width_idx 块切开
    # 第0段高度 = 0
    # 第1段高度 = step_height_idx
    # 第2段高度 = 2*step_height_idx
    # ...
    # 顶部的那一级高度就是最高高度，后面平台会用这个同一个高度
    if left_len > 0:
        num_left_steps = (left_len + step_width_idx - 1) // step_width_idx  # ceil
    else:
        num_left_steps = 0

    # 计算最高台阶的高度（平台高度）
    if num_left_steps > 0:
        top_height_val = (num_left_steps - 1) * step_height_idx
    else:
        top_height_val = 0

    # 逐级填充左半段
    for s in range(num_left_steps):
        xs = x_left_start + s * step_width_idx
        xe = min(x_left_start + (s + 1) * step_width_idx, x_left_end)
        h_val = s * step_height_idx  # 随着s上升
        terrain.height_field_raw[xs:xe, y1:y2] = h_val

    # -------- 中间平台：恒定高度 top_height_val --------
    terrain.height_field_raw[x_plat_start:x_plat_end, y1:y2] = top_height_val

    # -------- 右半段：下楼梯 (高度逐级降低) --------
    # 对 [x_right_start : total_len_x) 同样按 step_width_idx 切分
    # 第一段 = top_height_val
    # 第二段 = top_height_val - step_height_idx
    # 直到降到0或更低，不降成负数以下（地面以下就不必要了，防止出现坑台阶）
    if right_len > 0:
        num_right_steps = (right_len + step_width_idx - 1) // step_width_idx
    else:
        num_right_steps = 0

    for s in range(num_right_steps):
        xs = x_right_start + s * step_width_idx
        xe = min(x_right_start + (s + 1) * step_width_idx, x_right_end)
        h_val = top_height_val - s * step_height_idx
        if h_val < 0:
            h_val = 0
        terrain.height_field_raw[xs:xe, y1:y2] = h_val

    return terrain




def discrete_blocks_terrain(terrain, difficulty,
                            corridor_w=0.4, margin=0.10,
                            size_easy=0.28, size_hard=0.14,
                            gap_easy=0.10,  gap_hard=0.25,
                            h_easy=0.12,    h_hard=0.30,
                            stagger=True,   p_drop=0.0):
    """
    用高度场生成“离散踏块 + 中央横向走廊”的地形。
    - 难度↑：方块更小、间隙更大、台阶更高
    - 中央一条横向走廊（沿 y 全宽、沿 x 为一条带），走廊内不放块
    - 其余区域为深坑（负高），方块与走廊表面高度为 0
    """
    # ---- 难度到几何参数（单位：米）----
    t = float(np.clip(difficulty, 0.0, 1.0))
    lerp = lambda a, b: a + (b - a) * t
    block_size = lerp(size_easy, size_hard)   # 方块边长  ↓ 难度↑→变小
    gap        = lerp(gap_easy,  gap_hard)    # 块间距    ↑ 难度↑→变大
    block_h    = lerp(h_easy,    h_hard)      # 台阶高度  ↑ 难度↑→变高

    # ---- 像素化 ----
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    hf = terrain.height_field_raw
    H, W = hf.shape

    px_size   = max(1, int(round(block_size / hs)))
    px_gap    = max(1, int(round(gap        / hs)))
    px_stride = px_size + px_gap
    px_margin = int(round(margin     / hs))
    px_corr   = max(1, int(round(corridor_w / hs)))

    # ---- 全图初始化为坑 ----
    base_depth = -int(round(block_h / vs))  # 注意：height_field_raw 存整数高度
    hf[:, :] = base_depth

    # ---- 中央横向走廊（沿 x 居中的一条带）----
    center_x = H // 2
    x_corridor_start = max(0, center_x - px_corr // 2)
    x_corridor_end   = min(H, center_x + (px_corr + 1) // 2)
    hf[x_corridor_start:x_corridor_end, :] = 0  # 走廊置 0

    # ---- 方块只放在走廊外：上带 & 下带 ----
    lanes_x = [
        (px_margin, max(px_margin, x_corridor_start - px_margin)),                # 上侧
        (min(H - px_margin, x_corridor_end + px_margin), H - px_margin)           # 下侧
    ]
    y0 = px_margin
    y1 = W - px_margin
    if y1 - y0 < px_size:
        return  # 横向空间不足

    # ---- 在两侧带分别铺设离散方块网格 ----
    for xa, xb in lanes_x:
        if xb - xa < px_size:
            continue
        x_centers = np.arange(xa + px_size // 2, xb - (px_size - 1) // 2, px_stride)
        y_centers = np.arange(y0 + px_size // 2, y1 - (px_size - 1) // 2, px_stride)

        for r, cy in enumerate(y_centers):
            cx_row = x_centers.copy()
            if stagger and (r % 2 == 1):
                cx_row = cx_row + px_stride // 2
                cx_row = cx_row[(cx_row >= xa + px_size // 2) & (cx_row <= xb - (px_size - 1) // 2)]
            for cx in cx_row:
                if p_drop > 0.0 and np.random.rand() < p_drop:
                    continue
                xL = int(cx - px_size // 2); xR = int(cx + (px_size + 1) // 2)
                yL = int(cy - px_size // 2); yR = int(cy + (px_size + 1) // 2)
                hf[xL:xR, yL:yR] = 0


def gap_terrain(terrain, gap_size, platform_size=3.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = int(center_x - 1 / terrain.horizontal_scale)
    x2 = int(center_x + 2 / terrain.horizontal_scale)
    x3 = x1 - gap_size
    x4 = x2 + gap_size
    width = 1 + 1.0 * np.random.random()
    half_width =2* width 
    y1 = int(center_y - half_width / terrain.horizontal_scale)
    y2 = int(center_y + half_width / terrain.horizontal_scale)
    x5 = gap_size

    terrain.height_field_raw[:,:] = -1000
    terrain.height_field_raw[x5:x3, y1: y2] = 0
    terrain.height_field_raw[x1: x2, y1: y2] = 0
    terrain.height_field_raw[x4:, y1: y2] = 0

def climb_terrain(terrain, depth, platform_size=1.):
    # 把真实高度(米)换算到 height_field_raw 的离散高度单位
    depth_idx = int(depth / terrain.vertical_scale)

    # --- 定义两个阶梯带在 x 方向的位置和长度（沿前进方向） ---
    # 第一道阶梯：大概出现在 x ≈ 1m
    length1 = 1.0 + 0.2 * np.random.random()  # 随机长度, 1.0~1.2 m
    x1 = int(1.0 / terrain.horizontal_scale)
    x2 = int((1.0 + length1) / terrain.horizontal_scale)

    # 第二道阶梯：大概出现在 x ≈ 6m
    length2 = 1.0 + 0.2 * np.random.random()  # 同样随机一点长度
    x3 = int(6.0 / terrain.horizontal_scale)
    x4 = int((6.0 + length2) / terrain.horizontal_scale)

    # --- 定义中间走廊（允许机器人走的 y 区域），两侧全是深坑 ---
    center_y = terrain.width // 2  # 地形宽度的一半（左右方向中线）
    # 仿照 gap_terrain 的做法：随机走廊宽度
    width = 1.0 + 1.0 * np.random.random()   # 基础半宽（米量级）
    half_width = 2 * width                   # 再放大一点走廊，让它不是太窄

    y1 = int(center_y - half_width / terrain.horizontal_scale)
    y2 = int(center_y + half_width / terrain.horizontal_scale)

    # 边界安全裁剪，避免数组越界
    y1 = max(y1, 0)
    y2 = min(y2, terrain.height_field_raw.shape[1])

    # --- 第一步：把左右两边（走廊外）挖成深坑 ---
    # 这样机器人如果尝试从侧面绕开台阶，会掉下去
    terrain.height_field_raw[:, :y1] = -1000
    terrain.height_field_raw[:, y2:] = -1000

    # --- 第二步：把中间走廊里，特定的 x 段抬高为台阶/障碍 ---
    # 也就是让它必须“爬上/跨过”这些区段
    terrain.height_field_raw[x1:x2, y1:y2] = depth_idx
    terrain.height_field_raw[x3:x4, y1:y2] = depth_idx

    # 走廊的其他地方我们不改，保持原高度（通常是 0）
    # 这样机器人可以在走廊里平走 + 必须翻两个台阶
    # 而且没法绕侧面，因为侧面全是 -1000 的深坑


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows - 1, :] += (hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols - 1] += (hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows - 1, :num_cols - 1] += (
                    hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (
                    hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0