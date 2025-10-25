
from humanoidGym import GYM_ROOT_DIR
from humanoidGym.algo.dataset.motion_loader import AMPLoader
from humanoidGym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch, torchvision
import torch.nn.functional as F
import os
import random
import math
import time
from collections import deque

from humanoidGym.algo.ppo.utils import build_mirror_ls
from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg
from humanoidGym.utils import exponential_progress, quat_apply_yaw
from humanoidGym.utils.terrain_parkour import TerrainParkour

from humanoidGym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
import cv2
import time as pytime

class AmpG1HeightRobot(LeggedRobot):
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.cfg = cfg
        # get terrain type idx
        self.wave_start_idx = 0
        self.wave_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:1]))
        self.slope_start_idx = self.wave_end_idx
        self.slope_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:2]))
        self.stairup_start_idx = self.slope_end_idx
        self.stairup_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:3]))
        self.stairdown_start_idx = self.stairup_end_idx
        self.stairdown_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:4]))
        self.discrete_start_idx = self.stairdown_end_idx
        self.discrete_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:5]))
        self.gap_start_idx = self.discrete_end_idx
        self.gap_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:6]))
        self.pit_start_idx = self.gap_end_idx
        self.pit_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:7]))
        self.tilt_start_idx = self.pit_end_idx
        self.tilt_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:8]))
        self.crawl_start_idx = self.tilt_end_idx
        self.crawl_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:9]))
        self.roughflat_start_idx = self.crawl_end_idx
        self.roughflat_end_idx = self.cfg.env.num_envs
        
       
      

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
       
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[0], self.cfg.depth.resized[1]),
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
          # for debug
        self.debug_viz = True
        self.lookat_id = 0

        self.global_counter = 0
        self.total_env_steps_counter = 0

 

        if self.cfg.rewards.reward_curriculum:
            self.reward_curriculum_coef = [schedule[2] for schedule in self.cfg.rewards.reward_curriculum_schedule]


        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # amp related
        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

        self.discriminator = None # assigned in runner
        self.amp_state_normalizer = None # assigned in runner
        self.style_reward_normalizer = None # assigned in runner
        self.cur_amp_state_obs = None # assigned in runner
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs


    def update_current_amp_state(self,obs):
        self.cur_amp_state_obs = obs.clone().detach()
        
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        feet_names = self.cfg.asset.foot_name
        # knee_names = self.cfg.asset.knee_name
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # use the sensor to acquire contact force, may be more accurate
        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # add randomization related 
        self.init_randomize_props()
        self.init_randomize_lag()
        
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []

        # if(self.cfg.depth.use_camera):
        #     # All robots of Tilt and Crawl needs depth camera
        #     self.cfg.depth.camera_num_envs = min(self.cfg.depth.camera_num_envs, self.num_envs)
        #     self.depth_index_without_crawl_tilt = np.random.choice(range(self.tilt_start_idx), self.cfg.depth.camera_num_envs
        #                                                      - (self.crawl_end_idx - self.tilt_start_idx), replace=False)
        #     self.depth_index_without_crawl_tilt = np.sort(self.depth_index_without_crawl_tilt).astype(np.int)
        #     self.depth_index = np.concatenate((self.depth_index_without_crawl_tilt, range(self.tilt_start_idx, self.crawl_end_idx))).astype(np.int)
        #     self.depth_index_inverse = -np.ones(self.num_envs, dtype=np.int)
        #     for i in range(len(self.depth_index)):
        #         self.depth_index_inverse[self.depth_index[i]] = i

        if self.cfg.depth.use_camera:
            # 1) 相机 env 数量裁剪到 [0, num_envs]
            self.cfg.depth.camera_num_envs = int(min(max(self.cfg.depth.camera_num_envs, 0), self.num_envs))

            # 2) 无偏随机抽取 cams 个 env（不放回）
            #    用生成器可选固定种子，便于复现实验；没有就用系统随机
            rng = np.random.default_rng(getattr(self.cfg, "seed", None))
            self.depth_index = np.sort(
                rng.choice(self.num_envs, size=self.cfg.depth.camera_num_envs, replace=False)
            ).astype(np.int32)   # 有相机的 env_id 列表（已排序，便于后续二分/遍历）

            # 3) 反向索引：env_id -> 在“相机子集”里的下标；没有相机的 env 置 -1
            self.depth_index_inverse = -np.ones(self.num_envs, dtype=np.int32)
            self.depth_index_inverse[self.depth_index] = np.arange(self.depth_index.size, dtype=np.int32)
        else:
            # 不用相机时，保持接口一致
            self.depth_index = np.empty((0,), dtype=np.int32)
            self.depth_index_inverse = -np.ones(self.num_envs, dtype=np.int32)
        
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            # move up for uneven
            pos[2] += self.base_init_state[2]
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if(self.cfg.depth.use_camera and i in self.depth_index):
                self.attach_camera(i, env_handle, actor_handle)
            
        self.init_post_randomize_props()

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        # self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(knee_names)):
        #     self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            
        # AMP related
        ee_names = ['left_ankle_roll_link', 'right_ankle_roll_link','left_wrist_yaw_link', 'right_wrist_yaw_link']
        actor_body_names = self.gym.get_actor_rigid_body_names(
         self.envs[0], self.actor_handles[0]
        )
        self.end_effector_indices = []
        for name in ee_names:
            idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
            assert idx != -1, f"Rigid body '{name}' not found。检查 URDF 名称或 collapse_fixed_joints 设置。"
            assert actor_body_names[idx] == name, f"拿到 {actor_body_names[idx]}，不是期望的 {name}"
            self.end_effector_indices.append(idx)
        self.end_effector_indices = torch.tensor(self.end_effector_indices, dtype=torch.long, device=self.device)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros((self.cfg.env.num_single_observations-self.cfg.env.num_height),dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.
        
        return noise_vec
    
    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.rigid_body_lin_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[...,7:10]

        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:,:, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
        # feet multiple points height
        self.feet_height_points = self._init_feet_height_points()
        self.left_feet_heights = self._get_left_feet_heights()
        self.right_feet_heights = self._get_right_feet_heights()
        
    def _init_mirror(self):
        # need to be modified
        self.obs_mirror_ls = build_mirror_ls(self.dof_dict,self.cfg.asset.obs_mirror)
        self.action_mirror_ls = build_mirror_ls(self.dof_dict,['dofs'])

    def _init_action_scales(self):        
        self.action_scales = torch.tensor(self.cfg.control.action_scales).to(self.device).unsqueeze(0)
        
    def _init_buffers(self):
       
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # setup training counter
        self.ppo_iter = 0
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        


        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        self.sensor_forces = force_sensor_readings.view(self.num_envs, 2, 6)[..., :3]


        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.phase_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.base_height_points = self._init_base_height_points()
            self.forward_height_points = self._init_forward_height_points()
            self.measured_heights = self._get_heights()
            self.measured_forward_heights = self._get_forward_heights()

        else:
            self.measured_heights = 0.0
            
        if self.cfg.control.use_filter:
            self.action_filterd = torch.zeros(self.num_envs, self.num_actions,
                                            dtype=torch.float,
                                            device=self.device, requires_grad=False)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_dict = {}
        
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.dof_dict[name] = i      
            self.default_dof_pos[i] = angle
            
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        print(self.dof_dict)
        # history of observations
        self.obs_history = deque(maxlen=self.cfg.env.num_obs_lens)
        self.critic_obs_history = deque(maxlen=self.cfg.env.critic_num_obs_lens)
        
        for _ in range(self.cfg.env.num_obs_lens):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_observations,dtype=torch.float, device=self.device))
        
        for _ in range(self.cfg.env.critic_num_obs_lens):
            self.critic_obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_critic_single_observations,dtype=torch.float, device=self.device))
        self._init_foot()
        self._init_mirror()
        self._init_action_scales()
        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.cfg.depth.camera_num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
       
    
 
    
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.episode_sums["task"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums["style"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # reward curriculum
            if self.cfg.rewards.reward_curriculum:
                for j in range(len(self.cfg.rewards.reward_curriculum_term)):
                    if(name == self.cfg.rewards.reward_curriculum_term[j]):
                        rew *= self.reward_curriculum_coef[j]

            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        if self.discriminator is not None and self.amp_state_normalizer is not None :
            next_state_amp_obs = self.get_amp_observations()
            task_rew = self.rew_buf
            tot_rew, style_rew = self.discriminator.predict_amp_reward(self.cur_amp_state_obs, next_state_amp_obs, task_rew, self.dt, self.amp_state_normalizer, self.style_reward_normalizer)
            self.episode_sums["task"] += task_rew
            self.episode_sums["style"] += style_rew
            self.rew_buf = tot_rew
            
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:,:, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        self.feet_height = footpos_in_body_frame[:,:,2]
        
        # feet multiple point heights 
        self.feet_height_points = self._init_feet_height_points()
        self.left_feet_heights = self._get_left_feet_heights()
        self.right_feet_heights = self._get_right_feet_heights()
        
        # contact musk
        self.contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.
        
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
            
        for _ in range(self.cfg.control.decimation):
            
            actions_scaled = self.actions * self.action_scales
 
            if self.cfg.domain_rand.add_action_lag:
                self.action_lag_buffer[:,:,1:] = self.action_lag_buffer[:,:,:self.cfg.domain_rand.action_lag_timesteps_range[1]].clone()
                self.action_lag_buffer[:,:,0] = actions_scaled.clone()
                lagged_actions_scaled = self.action_lag_buffer[torch.arange(self.num_envs),:,self.action_lag_timestep.long()]
            else:
                lagged_actions_scaled = actions_scaled
                
            if self.cfg.control.use_filter:
                self.action_filterd = self.exp_avg_filter(lagged_actions_scaled, self.action_filterd,self.cfg.control.exp_avg_decay) 
                self.torques = self._compute_torques(self.action_filterd).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(lagged_actions_scaled).view(self.torques.shape)
                
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        env_ids, termination_privileged_obs = self.post_physics_step()
        self.extras['termination_id'] = env_ids
        self.extras['termination_privileged_obs'] = termination_privileged_obs[env_ids]

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
            # interpolation = torch.rand((self.cfg.depth.camera_num_envs, 1, 1), device=self.device)
            # self.extras["depth"] = self.depth_buffer[:, -1] * interpolation + self.depth_buffer[:, -2] * (1-interpolation)
        else:
            self.extras["depth"] = None

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf,self.rew_buf, self.reset_buf, self.extras
    

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        # self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        start_time = pytime.time()
        self.gym.start_access_image_tensors(self.sim)
        # for i in range(self.num_envs):
        for i in range(len(self.depth_index)):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                                self.envs[self.depth_index[i]],
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            # if(i == 0): print(torch.mean(depth_image)) # for debug, sometimes isaacgym will return all -inf depth image if not config properly

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                                 dim=0)
        self.gym.end_access_image_tensors(self.sim)
        print('acquiring depth image time:', pytime.time() -start_time)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_critic_obs()
        self.reset_idx(env_ids)
        
        self.update_depth_buffer()

        # after reset idx, the base_lin_vel, base_ang_vel, projected_gravity, height has changed, so should be re-computed
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts = contact
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

        return env_ids, termination_privileged_obs
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        vel_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        self.vel_violate = ((vel_error > 1.5) & (self.commands[:, 0] < 0.)) | ((vel_error < -1.5) & (self.commands[:, 0] > 0.))
        self.vel_violate *= (self.terrain_levels > 3)
        
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>0.8, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.vel_violate
    
    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    
    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            # add z value from amp data
            # amp_root_z = AMPLoader.get_root_pos_batch(frames)[:,2]
            # self.root_states[env_ids, 2] += amp_root_z
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            root_pos = AMPLoader.get_root_pos_batch(frames)
            root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
            self.root_states[env_ids, :3] = root_pos
            
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn

        # the base y position of tilt and gap envs can not deviate too far from the origin center
        tilt_env_ids = env_ids[torch.where(env_ids >= self.tilt_start_idx)]
        tilt_env_ids = tilt_env_ids[torch.where(tilt_env_ids < self.tilt_end_idx)]
        gap_env_ids = env_ids[torch.where(env_ids >= self.gap_start_idx)]
        gap_env_ids = gap_env_ids[torch.where(gap_env_ids < self.gap_end_idx)]
        tilt_and_gap_env_ids = torch.concatenate((tilt_env_ids, gap_env_ids))

        if self.custom_origins:
            self.root_states[tilt_and_gap_env_ids] = self.base_init_state
            self.root_states[tilt_and_gap_env_ids, :3] += self.env_origins[tilt_and_gap_env_ids]
            self.root_states[tilt_and_gap_env_ids, :1] += torch_rand_float(-1., 1., (len(tilt_and_gap_env_ids), 1), device=self.device) # x position within 1m of the center
            self.root_states[tilt_and_gap_env_ids, 1:2] += torch_rand_float(-0.0, 0.0, (len(tilt_and_gap_env_ids), 1),
                                                               device=self.device)
        else:
            self.root_states[tilt_and_gap_env_ids] = self.base_init_state
            self.root_states[tilt_and_gap_env_ids, :3] += self.env_origins[tilt_and_gap_env_ids]

        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            # if robot could move terrain_length, reset in a difficult terrain
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            # If the tracking reward is above 80% of the maximum, increase the range of commands
            self.update_command_curriculum(env_ids)
            
        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
            # self._resample_commands_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
            # self._resample_commands(env_ids)
        
        self._resample_commands(env_ids)
        
        self.refreshable_randomize_props(env_ids)
        self.refreshable_randomize_lag(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        if self.cfg.control.use_filter:
            self.action_filterd[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
            
        for i in range(self.critic_obs_history.maxlen):
            self.critic_obs_history[i][env_ids] *= 0
    
    def _post_physics_step_callback(self):
        self.update_feet_state()
        return super()._post_physics_step_callback()
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def update_reward_curriculum(self, current_iter):
        for i in range(len(self.cfg.rewards.reward_curriculum_schedule)):
            percentage = (current_iter - self.cfg.rewards.reward_curriculum_schedule[i][0]) / \
                         (self.cfg.rewards.reward_curriculum_schedule[i][1] - self.cfg.rewards.reward_curriculum_schedule[i][0])
            percentage = max(min(percentage, 1), 0)
            self.reward_curriculum_coef[i] = (1 - percentage) * self.cfg.rewards.reward_curriculum_schedule[i][2] + \
                                          percentage * self.cfg.rewards.reward_curriculum_schedule[i][3]


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # small vel and yaw set to zero for idol
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
    def _resample_commands_amp(self, env_ids, frames):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        root_orn = AMPLoader.get_root_rot_batch(frames)
        amp_lin_vel = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))[:,:2]
        amp_ang_vel = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))[:,2]

        self.commands[env_ids, :2] = amp_lin_vel
        self.commands[env_ids, 2] = amp_ang_vel
        
    
    def get_amp_observations(self):
       
        joint_pos = self.dof_pos
        #foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
    
        # 位置保存在 [:, :, 0:3]
        rb_pos = self.rigid_body_states[:, :, 0:3]
         # —— 取出脚端 world-space 位置 —— 
        foot_world = rb_pos[:, self.end_effector_indices, :]       # [N, 4, 3]
        # 取出 left_ankle_roll_link 和 right_ankle_roll_link
        # —— 取出 pelvis 世界坐标 —— 
        pelvis_world = self.root_states[:, :3].unsqueeze(1)       # [N, 1, 3]
          # —— 相对位置 = foot_world - pelvis_world —— 
        # —— world-space 相对位置 —— 
        foot_rel = foot_world - pelvis_world  # [N, 4, 3]
        # —— 展平 —— 
        foot_rel_flat = foot_rel.view(-1, 3)  # [N*4, 3]
        # —— 把 base_quat 重复成对应的 [N*4, 4] —— 
        quat_rep = self.base_quat.unsqueeze(1).repeat(1, 4, 1).view(-1, 4)  # [N*4, 4]
        # —— 用 quat_rotate_inverse 旋转到 body frame —— 
        foot_body_flat = quat_rotate_inverse(quat_rep, foot_rel_flat)    # [N*4, 3]
        # —— 再 reshape 回 [N, 12] —— 
        foot_pos = foot_body_flat.view(self.num_envs, 12)  

        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        z_pos = self.root_states[:, 2:3]

        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
    
    def compute_observations(self):
        """ Computes observations
        """    
        single_obs = torch.cat((
                            self.commands[:, :3] * self.commands_scale,
                            self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.actions
                            ),dim=-1)
        
        single_privileged_obs = torch.cat((
                                    self.commands[:, :3] * self.commands_scale, #3 
                                    self.base_lin_vel * self.obs_scales.lin_vel, #3
                                    self.base_ang_vel  * self.obs_scales.ang_vel, #3
                                    self.projected_gravity,#3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,#21
                                    self.dof_vel * self.obs_scales.dof_vel,#21
                                    self.actions,#21
                                    #self.contact_forces[:,self.feet_indices].view(self.num_envs,-1),#2 与真实长度不相符
                                    self.rand_push_force[:,:2],#2
                                    self.friction,#1
                                    self.feet_height #2 与真实长度不相符
                                    ),dim=-1)
        
        # add noise if needed
        if self.add_noise:
            single_obs += (2 * torch.rand_like(single_obs) - 1) * self.noise_scale_vec

        self.obs_history.append(single_obs)
        obs_history = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)],dim=1)
        self.obs_buf = obs_history.reshape(self.num_envs, -1)
            
        self.critic_obs_history.append(single_privileged_obs)
        critic_obs_history = torch.stack([self.critic_obs_history[i] for i in range(self.critic_obs_history.maxlen)],dim=1)
        critic_obs_history = critic_obs_history.reshape(self.num_envs, -1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            privileged_obs_buf = torch.cat((heights,critic_obs_history), dim=-1)
            self.obs_buf = torch.cat([heights,self.obs_buf],dim=-1)
        else:
            privileged_obs_buf = critic_obs_history
            
        self.privileged_obs_buf = privileged_obs_buf
        
        self.extras["observations"] = {}
        self.extras["observations"]["critic"] = self.privileged_obs_buf
        self.extras["observations"]["rnd_state"] = self.privileged_obs_buf
        
    def compute_termination_critic_obs(self):
        """ Computes observations
        """         
        single_privileged_obs = torch.cat((
                                    self.commands[:, :3] * self.commands_scale, #3 
                                    self.base_lin_vel * self.obs_scales.lin_vel, #3
                                    self.base_ang_vel  * self.obs_scales.ang_vel, #3
                                    self.projected_gravity,#3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,#21
                                    self.dof_vel * self.obs_scales.dof_vel,#21
                                    self.actions,#21
                                    #self.contact_forces[:,self.feet_indices].view(self.num_envs,-1),#2 与真实长度不相符
                                    self.rand_push_force[:,:2],#2
                                    self.friction,#1
                                    self.feet_height #2 与真实长度不相符
                                    ),dim=-1)
        # 避免直接修改critic obs history
        critic_obs_history = self.critic_obs_history.copy()
        critic_obs_history.append(single_privileged_obs)
        critic_obs_history = torch.stack([critic_obs_history[i] for i in range(critic_obs_history.maxlen)],dim=1)
        critic_obs_history = critic_obs_history.reshape(self.num_envs, -1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            privileged_obs_buf = torch.cat((heights,critic_obs_history), dim=-1)
        else:
            privileged_obs_buf = critic_obs_history

        return privileged_obs_buf
        
    def get_observations(self):
        if not self.extras:
            self.extras["observations"] = {}
            self.extras["observations"]["critic"] = self.privileged_obs_buf
            self.extras["observations"]["rnd_state"] = self.privileged_obs_buf
            # self.extras["depth"] = self.depth_buffer[:, -2] 
            
        return self.obs_buf, self.extras
    

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
    
    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[1]
            camera_props.height = self.cfg.depth.original[0]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_y_angle = np.random.uniform(config.y_angle[0], config.y_angle[1])

            camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
            camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])


            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_x_angle),
                                                           np.radians(camera_y_angle), np.radians(camera_z_angle))
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = TerrainParkour(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    
    def _init_forward_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_forward_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_forward_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_forward_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_forward_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_forward_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_forward_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_forward_height_points), self.forward_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_forward_height_points), self.forward_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def get_forward_map(self):
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_forward_heights, -1,
                             1.) * self.obs_scales.height_measurements

#--------------------------------------------------------------------------------------------------------------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        #print("base_height:", base_height)
       # print("self.cfg.rewards.base_height_target:", self.cfg.rewards.base_height_target)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long stepss
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
       # print("contact:", contact)         # ← 在这里打印出 contact#################################################
        forces_z = self.contact_forces[:, self.feet_indices, 2]
       # print("contact forces z:", forces_z)

        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.6) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew
    
    def _reward_exp_action_smooothness(self):
        # 动作越发顺滑越好
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return torch.exp(-1e-2*(term_1 + term_2 + term_3))
    
    def _reward_action_smooth(self):
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions
                + self.last_last_actions
            ),
            dim=1,
        )
        
    def _reward_power_dist(self):
        # Penalize power dist
        return torch.var(self.torques*self.dof_vel, dim=1)
    
    def _reward_power(self):
        return torch.sum(torch.abs(self.torques*self.dof_vel),dim=1)
    
    def _reward_exp_energy(self):
        return torch.exp(-1e-6*torch.sum(torch.square(self.dof_vel * self.torques),dim=1))
    
    def _reward_ankle_pitch_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[4,10]] * self.torques[:,[4,10]]),dim=1)
        return torch.exp(-(1e-6*energy))

    def _reward_ankle_roll_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[5,11]] * self.torques[:,[5,11]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_ankle_action_pitch(self):
        return torch.sum(torch.square(self.actions[:, [4,10]]), dim=1)
    
    def _reward_ankle_action_roll(self):
        return torch.sum(torch.square(self.actions[:, [5,11]]), dim=1)
    
    def _reward_foot_slip(self):
      
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    
    def _reward_feet_contact_forces(self):
        diff = torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force),dim=1)
        diff = 0.1*torch.clamp(diff,min=torch.zeros_like(diff))
        reward = torch.exp(-diff)
        return reward
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_yaw_error_when_rate_matches(self):
        # 参数（可调）
        k_yaw = 1.0        # 偏航角误差权重（平方惩罚）
        k_rate = 1.5       # 偏航速率误差权重（平方惩罚）


        # 计算速率匹配条件（绝对误差）
        rate_err = self.base_ang_vel[:, 2] - self.commands[:, 2]
        

     
        desired_yaw = self.commands[:, 3]

        # yaw 角误差（-pi..pi）
        yaw_diff = self.rpy[:, 2] - desired_yaw
        yaw_err = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

        # 基础惩罚：二次惩罚（更平滑、可微）
        yaw_pen = k_yaw * (yaw_err ** 2)
        rate_pen = k_rate * (rate_err ** 2)

        # 当速率接近时，放大 yaw 惩罚（鼓励同时满足角度与速率）
        penalty = rate_pen+yaw_pen


        return penalty


    def _reward_cheat(self):
        # penalty cheating to bypass the obstacle
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:self.roughflat_start_idx, 1], forward[:self.roughflat_start_idx, 0])
        cheat = (heading > 1.0) | (heading < -1.0)
        cheat_penalty = torch.zeros(self.num_envs, device=self.device)
        cheat_penalty[:self.roughflat_start_idx] = cheat
        return cheat_penalty
    
    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices,
                        :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)

        edge_reward = torch.zeros_like(rew)
        edge_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
        return edge_reward