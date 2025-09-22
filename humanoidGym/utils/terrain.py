from numpy.random import choice
from scipy import interpolate
import numpy as np
from isaacgym import terrain_utils
from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        # each sub terrain length
        self.env_length = cfg.terrain_length
        # each sub terrain width
        self.env_width = cfg.terrain_width
        # each terrain type proportion
        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # self.platform is size of platform for some terrain type, like pit, gap, slope
        self.platform = cfg.platform
        # max_difficulty is based on num_rows
        # terrain difficulty is from 0 to max
        self.max_difficulty = (cfg.num_rows-1)/cfg.num_rows

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # border_size is whole terrain border
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # whole terrain cols
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        # whole terrain rows
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        self.idx = 0
        
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()  
              
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.7, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            # i j select row col position in whole terrain
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

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
            if i == 3:
                terrain_utils.random_uniform_terrain(terrain, min_height=-0.1, max_height=0.15, step=0.005, downsampled_scale=0.2)
            if i == 1:
                terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.33, step_height=0.15, platform_size=3.)
                
            if i == 2:
               terrain_utils.discrete_obstacles_terrain(terrain, 0.15, 1, 1.5, 30, platform_size=3.)
            if i ==0:
                terrain_utils.pyramid_sloped_terrain(terrain, slope=0.5, platform_size=3.)

            if i == 4:
                terrain_utils.random_uniform_terrain(terrain, min_height=-0.1, max_height=0.15, step=0.005, downsampled_scale=0.2)

            if i == 5:
                terrain_utils.random_uniform_terrain(terrain, min_height=-0.1, max_height=0.15, step=0.005, downsampled_scale=0.2)
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

        env_origin_x = 0 - (0.1) * self.env_width
        env_origin_y = (0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, 0]
    
    # choice select terrain type, difficulty select row, row increase difficulty increase
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        rought_flat_min_height = - self.cfg.rough_flat_range[0] - difficulty * (self.cfg.rough_flat_range[1] - self.cfg.rough_flat_range[0]) / self.max_difficulty
        rought_flat_max_height = self.cfg.rough_flat_range[0] + difficulty * (self.cfg.rough_flat_range[1] - self.cfg.rough_flat_range[0]) / self.max_difficulty
        slope = self.cfg.slope_range[0] + difficulty * (self.cfg.slope_range[1] - self.cfg.slope_range[0]) / self.max_difficulty
        rought_slope_min_height = - self.cfg.rough_slope_range[0] - difficulty * (self.cfg.rough_slope_range[1] - self.cfg.rough_slope_range[0]) / self.max_difficulty
        rought_slope_max_height = self.cfg.rough_slope_range[0] + difficulty * (self.cfg.rough_slope_range[1] - self.cfg.rough_slope_range[0]) / self.max_difficulty
        stair_width = self.cfg.stair_width_range[0] + difficulty * (self.cfg.stair_width_range[1] - self.cfg.stair_width_range[0]) / self.max_difficulty
        stair_height = self.cfg.stair_height_range[0] + difficulty * (self.cfg.stair_height_range[1] - self.cfg.stair_height_range[0]) / self.max_difficulty
        discrete_obstacles_height = self.cfg.discrete_height_range[0] + difficulty * (self.cfg.discrete_height_range[1] - self.cfg.discrete_height_range[0]) / self.max_difficulty

        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        amplitude = 0.2 + 0.333 * difficulty
        
        if choice < self.proportions[0]:
            idx = 1
            #self.add_fractal_noise(terrain, 0.01)
            return terrain
        elif choice < self.proportions[1]:
            idx = 2
            terrain_utils.random_uniform_terrain(terrain, 
                                                 min_height=rought_flat_min_height, 
                                                 max_height=rought_flat_max_height, 
                                                 step=0.005, 
                                                 downsampled_scale=0.2)
            self.add_fractal_noise(terrain, self.cfg.fractal_noise_strength)
        elif choice < self.proportions[3]:
            idx = 4
            if choice < self.proportions[2]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, 
                                                 slope=slope, 
                                                 platform_size=self.platform)
            terrain_utils.random_uniform_terrain(terrain, 
                                                 min_height=rought_slope_min_height, 
                                                 max_height=rought_slope_max_height,
                                                 step=0.005, 
                                                 downsampled_scale=0.2)
            self.add_fractal_noise(terrain, self.cfg.fractal_noise_strength)
            
        elif choice < self.proportions[5]:
            idx = 6
            if choice < self.proportions[4]:
                idx = 5
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, 
                                                 slope=slope, 
                                                 platform_size=self.platform)
            self.add_fractal_noise(terrain, self.cfg.fractal_noise_strength)
            
        elif choice < self.proportions[7]:
            idx = 8
            if choice<self.proportions[6]:
                idx = 7
                stair_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, 
                                                 step_width=stair_width, 
                                                 step_height=stair_height, 
                                                 platform_size=self.platform)
        elif choice < self.proportions[8]:
            idx = 9
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, 
                                                     discrete_obstacles_height, 
                                                     rectangle_min_size, 
                                                     rectangle_max_size, 
                                                     num_rectangles, 
                                                     platform_size=self.platform)
            self.add_fractal_noise(terrain, self.cfg.fractal_noise_strength)

        # self.add_fractal_noise(terrain, self.cfg.fractal_noise_strength)
        self.idx = idx
        return terrain
    
    def generate_fractal_noise(self, shape, octaves=4, persistence=0.5, lacunarity=2.0):
        rows, cols = shape
        noise = np.zeros(shape)
        amplitude = 1.0
        for octave in range(octaves):
            freq = lacunarity ** octave
            grid_rows = int(rows / freq) + 1
            grid_cols = int(cols / freq) + 1
            grid = np.random.randn(grid_rows, grid_cols)
            x = np.linspace(0, cols, grid_cols)
            y = np.linspace(0, rows, grid_rows)
            interp_func = interpolate.interp2d(x, y, grid, kind='linear')
            x_new = np.linspace(0, cols, cols)
            y_new = np.linspace(0, rows, rows)
            layer = interp_func(x_new, y_new)
            noise += layer * amplitude
            amplitude *= persistence
        # 归一化到 [-1, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
        return noise

    def add_fractal_noise(self, terrain, strength):
        strength_units = strength / self.cfg.vertical_scale
        if strength == 0:
            return
        noise = self.generate_fractal_noise(terrain.height_field_raw.shape)
        scaled_noise = (noise * strength_units).astype(np.int16)

        terrain.height_field_raw += scaled_noise
        # 防止溢出
        terrain.height_field_raw = np.clip(
            terrain.height_field_raw,
            np.iinfo(np.int16).min,
            np.iinfo(np.int16).max
        )
    
    # row col select position in whole terrain
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = self.idx

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


# class Terrain:
#     def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

#         self.cfg = cfg
#         self.num_robots = num_robots
#         self.type = cfg.mesh_type
#         if self.type in ["none", 'plane']:
#             return
#         self.env_length = cfg.terrain_length
#         self.env_width = cfg.terrain_width
#         self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

#         self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
#         self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

#         self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
#         self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

#         self.border = int(cfg.border_size/self.cfg.horizontal_scale)
#         self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
#         self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

#         self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
#         if cfg.curriculum:
#             self.curiculum()
#         elif cfg.selected:
#             self.selected_terrain()
#         else:    
#             self.randomized_terrain()   
        
#         self.heightsamples = self.height_field_raw
#         if self.type=="trimesh":
#             self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
#                                                                                             self.cfg.horizontal_scale,
#                                                                                             self.cfg.vertical_scale,
#                                                                                             self.cfg.slope_treshold)
    
#     def randomized_terrain(self):
#         for k in range(self.cfg.num_sub_terrains):
#             # Env coordinates in the world
#             (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

#             choice = np.random.uniform(0, 1)
#             difficulty = np.random.choice([0.5, 0.75, 0.9])
#             terrain = self.make_terrain(choice, difficulty)
#             self.add_terrain_to_map(terrain, i, j)
        
#     def curiculum(self):
#         for j in range(self.cfg.num_cols):
#             for i in range(self.cfg.num_rows):
#                 difficulty = i / self.cfg.num_rows
#                 choice = j / self.cfg.num_cols + 0.001

#                 terrain = self.make_terrain(choice, difficulty)
#                 self.add_terrain_to_map(terrain, i, j)

#     def selected_terrain(self):
#         terrain_type = self.cfg.terrain_kwargs.pop('type')
#         for k in range(self.cfg.num_sub_terrains):
#             # Env coordinates in the world
#             (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

#             terrain = terrain_utils.SubTerrain("terrain",
#                               width=self.width_per_env_pixels,
#                               length=self.width_per_env_pixels,
#                               vertical_scale=self.vertical_scale,
#                               horizontal_scale=self.horizontal_scale)

#             eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
#             self.add_terrain_to_map(terrain, i, j)
    
#     def make_terrain(self, choice, difficulty):
#         terrain = terrain_utils.SubTerrain(   "terrain",
#                                 width=self.width_per_env_pixels,
#                                 length=self.width_per_env_pixels,
#                                 vertical_scale=self.cfg.vertical_scale,
#                                 horizontal_scale=self.cfg.horizontal_scale)
#         slope = difficulty * 0.4
#         amplitude = 0.01 + 0.07 * difficulty
#         step_height = 0.05 + 0.18 * difficulty
#         discrete_obstacles_height = 0.05 + difficulty * 0.1
#         stepping_stones_size = 1.5 * (1.05 - difficulty)
#         stone_distance = 0.05 if difficulty==0 else 0.1
#         gap_size = 1. * difficulty
#         pit_depth = 1. * difficulty
#         if choice < self.proportions[0]:
#             if choice < self.proportions[0]/ 2:
#                 slope *= -1
#             terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
#         elif choice < self.proportions[1]:
#             terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
#             terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
#         elif choice < self.proportions[3]:
#             if choice<self.proportions[2]:
#                 step_height *= -1
#             terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3.)
#         elif choice < self.proportions[4]:
#             num_rectangles = 20
#             rectangle_min_size = 1.
#             rectangle_max_size = 2.
#             terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
#         elif choice < self.proportions[5]:
#             terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
#         elif choice < self.proportions[6]:
#             gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
#         else:
#             pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
#         return terrain

#     def add_terrain_to_map(self, terrain, row, col):
#         i = row
#         j = col
#         # map coordinate system
#         start_x = self.border + i * self.length_per_env_pixels
#         end_x = self.border + (i + 1) * self.length_per_env_pixels
#         start_y = self.border + j * self.width_per_env_pixels
#         end_y = self.border + (j + 1) * self.width_per_env_pixels
#         self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

#         env_origin_x = (i + 0.5) * self.env_length
#         env_origin_y = (j + 0.5) * self.env_width
#         x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
#         x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
#         y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
#         y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
#         env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
#         self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

# def gap_terrain(terrain, gap_size, platform_size=1.):
#     gap_size = int(gap_size / terrain.horizontal_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale)

#     center_x = terrain.length // 2
#     center_y = terrain.width // 2
#     x1 = (terrain.length - platform_size) // 2
#     x2 = x1 + gap_size
#     y1 = (terrain.width - platform_size) // 2
#     y2 = y1 + gap_size
   
#     terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
#     terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

# def pit_terrain(terrain, depth, platform_size=1.):
#     depth = int(depth / terrain.vertical_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale / 2)
#     x1 = terrain.length // 2 - platform_size
#     x2 = terrain.length // 2 + platform_size
#     y1 = terrain.width // 2 - platform_size
#     y2 = terrain.width // 2 + platform_size
#     terrain.height_field_raw[x1:x2, y1:y2] = -depth