
from humanoidGym import GYM_ROOT_DIR
from humanoidGym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import os
import random
import time
from humanoidGym.algo.ppo.utils import build_mirror_ls
from humanoidGym.utils import exponential_progress

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class LongUnevenRobot(LeggedRobot):
    
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
        knee_names = self.cfg.asset.knee_name
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
    
        
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
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
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
            
        self.init_post_randomize_props()

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_observations,dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.
        
        # noise_vec[:3] = 0. # commands
        # noise_vec[3:4] = 0. # stand_cmd
        # noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[7:10] = noise_scales.gravity * noise_level
        # noise_vec[10:10+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[10+self.num_actions:10+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[10+2*self.num_actions:10+3*self.num_actions] = 0.
        
        # noise_vec[:190] = 0. # commands,height
        # noise_vec[190:191] = 0. # stand_cmd
        # noise_vec[191:194] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[194:197] = noise_scales.gravity * noise_level
        # noise_vec[197:197+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[197+self.num_actions:197+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[197+2*self.num_actions:197+3*self.num_actions] = 0.
        
        # noise_vec[:190] = 0. # commands,height
        # # noise_vec[190:191] = 0. # stand_cmd
        # noise_vec[190:193] = 0. # lin vel 
        # noise_vec[193:196] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[196:199] = noise_scales.gravity * noise_level
        # noise_vec[199:199+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[199+self.num_actions:199+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[199+2*self.num_actions:199+3*self.num_actions] = 0.
        
        return noise_vec
    
    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.knee_state = self.rigid_body_states_view[:, self.knee_indices, :]
        
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:,:, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
        self.knee_pos = self.knee_state[:, :, :3]
        
        # feet multiple points height
        self.feet_height_points = self._init_feet_height_points()
        self.left_feet_heights = self._get_left_feet_heights()
        self.right_feet_heights = self._get_right_feet_heights()
        
    def _init_mirror(self):
        # need to be modified
        self.obs_mirror_ls = build_mirror_ls(self.dof_dict,self.cfg.asset.obs_mirror)
        self.action_mirror_ls = build_mirror_ls(self.dof_dict,['dofs'])
        
    def _init_action_scales(self):
        self.action_scales = self.cfg.control.action_scale*torch.ones(1,self.num_actions,device=self.device)
        # set roll and yaw to 0.25
        #{'J_hip_l_roll': 0, 'J_hip_l_yaw': 1, 'J_hip_l_pitch': 2, 'J_knee_l_pitch': 3, 'J_ankle_l_pitch': 4, 'J_ankle_l_roll': 5, 'J_hip_r_roll': 6, 'J_hip_r_yaw': 7, 'J_hip_r_pitch': 8, 'J_knee_r_pitch': 9, 'J_ankle_r_pitch': 10, 'J_ankle_r_roll': 11}
        # self.action_scales[:,[0,1,5,6,7,11]] = 0.25
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self._init_mirror()
        self._init_action_scales()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.knee_state = self.rigid_body_states_view[:, self.knee_indices, :]
        
        self.knee_pos = self.knee_state[:, :, :3]
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
        
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

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

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf,self.rew_buf, self.reset_buf, self.extras
    
    def _post_physics_step_callback(self):
        self.update_feet_state()

        self.phase_length_buf += 1
        self.phase = self._get_phase()
        self.compute_ref_state()
        self.stance_mask = self._get_gait_phase()
        self.contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.
        return super()._post_physics_step_callback()
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # if random.random() > 0.05:
        #     self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        #     self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        #     if self.cfg.commands.heading_command:
        #         self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        #     else:
        #         self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        # else:
        #     self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        #     self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        #     if self.cfg.commands.heading_command:
        #         self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        #     else:
        #         self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
        
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # small vel and yaw set to zero for idol
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > self.cfg.commands.stand_com_threshold).unsqueeze(1)
            
    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        #phase = self.episode_length_buf * self.dt / cycle_time
        if self.cfg.commands.sw_switch:
            stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
            self.phase_length_buf[stand_command] = 0
            phase = (self.phase_length_buf * self.dt / cycle_time) * (~stand_command)
        else:
            phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        #phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos > 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        
        stance_mask[torch.abs(sin_pos) < 0.05] = 1

        return stance_mask

    # def compute_ref_state(self):
    #     #phase = self._get_phase()
    #     sin_pos = torch.sin(2 * torch.pi * self.phase)
    #     sin_pos_l = sin_pos.clone()
    #     sin_pos_r = sin_pos.clone()
    #     self.ref_dof_pos = torch.zeros_like(self.dof_pos)
    #     scale_1 = self.cfg.rewards.target_joint_pos_scale
    #     scale_2 = 2 * scale_1
    #     # left swing
    #     sin_pos_l[sin_pos_l > 0] = 0
    #     self.ref_dof_pos[:, 2] = -sin_pos_l * scale_1
    #     self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
    #     self.ref_dof_pos[:, 4] = -sin_pos_l * scale_1
    #     # print(phase[0], sin_pos_l[0])
    #     # right
    #     sin_pos_r[sin_pos_r < 0] = 0
    #     self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
    #     self.ref_dof_pos[:, 9] = -sin_pos_r * scale_2
    #     self.ref_dof_pos[:, 10] = sin_pos_r * scale_1

    #     self.ref_dof_pos[torch.abs(sin_pos) < 0.05] = 0.
        
    #     self.ref_action = 2 * self.ref_dof_pos
        
    #     self.ref_dof_pos += self.default_dof_pos
        
    def compute_ref_state(self):
        #phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left swing
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = -sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        # self.ref_dof_pos[:, 4] = -sin_pos_l * scale_1
        # print(phase[0], sin_pos_l[0])
        # right
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = -sin_pos_r * scale_2
        # self.ref_dof_pos[:, 10] = sin_pos_r * scale_1

        self.ref_dof_pos[torch.abs(sin_pos) < 0.05] = 0.
        
        self.ref_action = 2 * self.ref_dof_pos
        
        self.ref_dof_pos += self.default_dof_pos

    def compute_observations(self):
        """ Computes observations
        """
        sin_pos = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)

        diff = self.dof_pos - self.ref_dof_pos
        stand_cmd = (torch.norm(self.commands[:, :3], dim=1,keepdim=True) <= self.cfg.commands.stand_com_threshold)
        
        single_obs = torch.cat((
                            self.commands[:, :3] * self.commands_scale,
                            self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.actions
                            ),dim=-1)
        

        # single_obs = torch.cat((
        #                     self.commands[:, :3] * self.commands_scale,
        #                     self.base_lin_vel * self.obs_scales.lin_vel,
        #                     self.base_ang_vel  * self.obs_scales.ang_vel,
        #                     self.projected_gravity,
        #                     (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                     self.dof_vel * self.obs_scales.dof_vel,
        #                     self.actions
        #                     ),dim=-1)
        
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     single_obs = torch.cat([heights,single_obs],dim=-1)
        
        # single_privileged_obs = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             self.commands[:, :3] * self.commands_scale,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             self.actions,
        #                             sin_pos,
        #                             cos_pos,
        #                             self.friction,
        #                             self.stance_mask,
        #                             diff,
        #                             self.rand_push_force[:,:2],
        #                             self.contact_forces[:,self.feet_indices].view(self.num_envs,-1),
        #                             stand_cmd,
        #                             self.contact_mask
        #                             ),dim=-1)
        
        # single_privileged_obs = torch.cat((
        #                             self.commands[:, :3] * self.commands_scale,
        #                             self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             self.actions,
        #                             sin_pos,
        #                             cos_pos,
        #                             self.friction,
        #                             self.stance_mask,
        #                             diff,
        #                             self.rand_push_force[:,:2],
        #                             self.contact_forces[:,self.feet_indices].view(self.num_envs,-1),
        #                             stand_cmd,
        #                             self.contact_mask
        #                             ),dim=-1)
        
        single_privileged_obs = torch.cat((
                                    self.commands[:, :3] * self.commands_scale,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_pos,
                                    cos_pos,
                                    self.stance_mask,
                                    diff,
                                    self.contact_forces[:,self.feet_indices].view(self.num_envs,-1),
                                    stand_cmd,
                                    self.rand_push_force[:,:2],
                                    self.friction,
                                    self.contact_mask,
                                    self.feet_height 
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
            
        self.privileged_obs_buf = privileged_obs_buf
        
        self.extras["observations"] = {}
        self.extras["observations"]["critic"] = self.privileged_obs_buf
        self.extras["observations"]["rnd_state"] = self.privileged_obs_buf
        
    def get_observations(self):
        if not self.extras:
            self.extras["observations"] = {}
            self.extras["observations"]["critic"] = self.privileged_obs_buf
            self.extras["observations"]["rnd_state"] = self.privileged_obs_buf
        return self.obs_buf, self.extras
    
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        
        # stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        # pos_target[stand_command] = self.default_dof_pos.clone()
        
        diff = joint_pos - pos_target
        # # scale up ankle part
        # diff[:,[4,10]] = diff[:,[4,10]] * 2.0
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        # r[stand_command] = 1.0
        return r
    
    def _reward_ankle_joint_pos(self):
        """
        ankle joint pos should be the pos around knee pos div with 1.67 
        """
        ankle_pitch_index = [4,10]
        knee_pitch_index = [3,9]
        
        ankle_pos = self.dof_pos[:,ankle_pitch_index]*-1.805#1.67
        knee_pos = self.dof_pos[:,knee_pitch_index]
        
        diff = ankle_pos - knee_pos
        r = torch.exp(-2 * torch.norm(diff,dim=1)) 
        
        return r
        

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase().clone()
        # stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)
    
    def _reward_no_fly(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        contact_average = torch.mean(contact*1.0,dim=1)
        fly_mask = torch.zeros_like(contact_average)
        reward = torch.where(contact_average != fly_mask, 1.0, -1.0)

        return reward
    
    def _reward_idol_feet_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = torch.ones((self.num_envs, 2), device=self.device)
        reward = torch.mean(torch.where(contact == stance_mask, 1.0, -10.0))
        
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        reward = torch.where(stand_command, reward.clone(),
                         torch.zeros_like(reward))
        return reward
    
    def _reward_idol_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        reward = torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        reward = torch.where(stand_command, reward.clone(),
                         torch.zeros_like(reward))
        return reward
        
    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.rpy[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 350), dim=1)
        # move_command = (torch.norm(self.commands[:, :3], dim=1) > self.cfg.commands.stand_com_threshold)
        # r = torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 350), dim=1)
        # r = torch.where(move_command, r.clone(),
        #                 torch.zeros_like(r))
        # return r

    #{'J_hip_l_roll': 0, 'J_hip_l_yaw': 1, 'J_hip_l_pitch': 2, 'J_knee_l_pitch': 3, 'J_ankle_l_pitch': 4, 'J_ankle_l_roll': 5, 'J_hip_r_roll': 6, 'J_hip_r_yaw': 7, 'J_hip_r_pitch': 8, 'J_knee_r_pitch': 9, 'J_ankle_r_pitch': 10, 'J_ankle_r_roll': 11}
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, [0,1,5]]
        right_yaw_roll = joint_diff[:, [6,7,11]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    
    def _reward_default_joint_yaw(self):
        joint_diff = self.dof_pos - self.default_dof_pos
        yaw = joint_diff[:, [1,7]]
        yaw = torch.norm(yaw, dim=1)
        yaw = torch.clamp(yaw - 0.1, 0, 50)
        return torch.exp(-yaw * 100)
    
    def _reward_default_joint_roll(self):
        joint_diff = self.dof_pos - self.default_dof_pos
        yaw = joint_diff[:, [0,6]]
        yaw = torch.norm(yaw, dim=1)
        yaw = torch.clamp(yaw - 0.1, 0, 50)
        return torch.exp(-yaw * 100)

    def _reward_default_joint_ankle_roll(self):
        joint_diff = self.dof_pos - self.default_dof_pos
        yaw = joint_diff[:, [5,11]]
        yaw = torch.norm(yaw, dim=1)
        yaw = torch.clamp(yaw - 0.1, 0, 50)
        return torch.exp(-yaw * 100)
    
    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    # def _reward_tracking_lin_vel(self):
    #     """
    #     Tracks linear velocity commands along the xy axes. 
    #     Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
    #     """
    #     stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     lin_vel_error_square = torch.sum(torch.square(
    #         self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     lin_vel_error_abs = torch.sum(torch.abs(
    #         self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     r_square = torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
    #     r_abs = torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
    #     r = torch.where(stand_command, r_abs, r_square)
      
    #     return r

    # def _reward_tracking_ang_vel(self):
    #     """
    #     Tracks angular velocity commands for yaw rotation.
    #     Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
    #     """   
    #     stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     ang_vel_error_square = torch.square(
    #         self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     ang_vel_error_abs = torch.abs(
    #         self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     r_square = torch.exp(-ang_vel_error_square * self.cfg.rewards.tracking_sigma)
    #     r_abs = torch.exp(-ang_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
    #     r = torch.where(stand_command, r_abs, r_square)

    #     return r 
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # get new feet height in local frame
        #feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        # self.feet_height
        # delta_z = feet_z - self.last_feet_z
        # self.feet_height += delta_z
        # self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should larger than target feet height at the peak
        rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 2.0
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.05)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.01 * torch.sum(torch.abs(self.actions), dim=1)
        return 2.0*term_1 + term_2 + term_3
    
    def _reward_hip_yaw_roll_action_smoothness(self):
        hip_yaw_roll_index = [0,1,6,7]
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions)[:,hip_yaw_roll_index], dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions)[:,hip_yaw_roll_index], dim=1)
        term_3 = 0.01 * torch.sum(torch.abs(self.actions[:,hip_yaw_roll_index]), dim=1)
        return 2.0*term_1 + term_2 + term_3
    
    def _reward_hip_yaw_action_smoothness(self):
        hip_yaw_index = [1,7]
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions)[:,hip_yaw_index], dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions)[:,hip_yaw_index], dim=1)
        term_3 = 0.01 * torch.sum(torch.abs(self.actions[:,hip_yaw_index]), dim=1)
        return 2.0*term_1 + term_2 + term_3
    
    def _reward_hip_roll_action_smoothness(self):
        hip_roll_index = [0,6]
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions)[:,hip_roll_index], dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions)[:,hip_roll_index], dim=1)
        term_3 = 0.01 * torch.sum(torch.abs(self.actions[:,hip_roll_index]), dim=1)
        return 2.0*term_1 + term_2 + term_3
    
    def _reward_ankle_pitch_action_smoothness(self):
        ankle_pitch_index = [4,10]
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions)[:,ankle_pitch_index], dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions)[:,ankle_pitch_index], dim=1)
        term_3 = 0.01 * torch.sum(torch.abs(self.actions[:,ankle_pitch_index]), dim=1)
        return 2.0*term_1 + term_2 + term_3
    
    def _reward_ankle_roll_action_smoothness(self):
        ankle_roll_index = [5,11]
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions)[:,ankle_roll_index], dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions)[:,ankle_roll_index], dim=1)
        term_3 = 0.01 * torch.sum(torch.abs(self.actions[:,ankle_roll_index]), dim=1)
        return 2.0*term_1 + term_2 + term_3
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_stand_still(self):
        # penalize motion at zero commands
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r
    
    def _reward_stand_still_pen(self):
        # penalize motion at zero commands
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r
    
    def _reward_idol_exp_energy(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.exp(-1e-6*torch.sum(torch.square(self.dof_vel * self.torques),dim=1))
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r
    
    def _reward_idol_root_vel(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.norm(self.root_states[:, 7:13], dim=1)
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r

    def _reward_idol_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        actions = self.actions.clone()*10
        last_actions = self.last_actions.clone()*10
        last_last_actions = self.last_last_actions.clone()*10
        
        term_1 = torch.sum(torch.square(
            last_actions - actions), dim=1)
        term_2 = torch.sum(torch.square(
            actions + last_last_actions - 2 * last_actions), dim=1)
        #term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        r = term_1 + term_2 #+ term_3
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r

    # def _reward_stand_still(self):
    #     # penalize motion at zero commands
    #     stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.target_dof_pos), dim=1))
    #     r = torch.where(stand_command, r.clone(),
    #                     torch.zeros_like(r))
    #     return r
    
    def _reward_stand_still_power(self):
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        r = torch.sum(torch.abs(self.torques*self.dof_vel),dim=1)
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r
    
    def _reward_simple_feet_min_distance(self):
        # get local frame feet pos
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        # compute y distance
        y_dist = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1]) - 0.18
        return torch.clamp(y_dist,max=0.0)
    
    def _reward_simple_knee_min_distance(self):
        cur_kneepos_translated = self.knee_pos - self.root_states[:, 0:3].unsqueeze(1)
        kneepos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            kneepos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_kneepos_translated[:, i, :])
        # compute y distance
        y_dist = torch.abs(kneepos_in_body_frame[:,0,1] - kneepos_in_body_frame[:,1,1]) - 0.18
        return torch.clamp(y_dist,max=0.0)
    
    def _reward_feet_rotation(self):
        feet_quat = self.feet_state[:,:,3:7]
        feet_euler_xyz = get_euler_xyz_tensor(feet_quat)
        rotation = torch.sum(torch.square(feet_euler_xyz[:,:,:2]),dim=[1,2])
        # rotation = torch.sum(torch.square(feet_euler_xyz[:,:,1]),dim=1)
        r = torch.exp(-rotation*15)
        return r
    
    def _reward_exp_action_smooothness(self):
        # 动作越发顺滑越好
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return torch.exp(-1e-2*(term_1 + term_2 + term_3))
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_power_dist(self):
        # Penalize power dist
        return torch.var(self.torques*self.dof_vel, dim=1)
    
    def _reward_power(self):
        return torch.sum(torch.abs(self.torques*self.dof_vel),dim=1)
    
    def _reward_exp_energy(self):
        return torch.exp(-1e-6*torch.sum(torch.square(self.dof_vel * self.torques),dim=1))
    
    def _reward_ankle_roll_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[5,11]] * self.torques[:,[5,11]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_ankle_pitch_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[4,10]] * self.torques[:,[4,10]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_ankle_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[4,5,10,11]] * self.torques[:,[4,5,10,11]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_hip_roll_yaw_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[0,1,6,7]] * self.torques[:,[0,1,6,7]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_hip_yaw_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[1,7]] * self.torques[:,[1,7]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_foot_normal_reward(self):
        # 获取原始接触力向量 [num_envs, num_feet, 3]
        contact_vectors = self.contact_forces[:, self.feet_indices, :3]
        
        # 计算接触力模长并添加极小值防止除零
        force_magnitude = torch.norm(contact_vectors, dim=-1, keepdim=True) + 1e-6
        
        # 归一化得到接触力方向（即推断的接触面法线）
        inferred_normals = contact_vectors / force_magnitude
        
        # 计算与理想地面法线（z轴方向）的对齐程度
        # 直接取z分量的值，等价于与[0,0,1]的点积
        vertical_alignment = inferred_normals[..., 2]  # shape: [num_envs, num_feet]
        
        # 创建接触掩码（排除非接触状态的脚）
        contact_mask = (force_magnitude.squeeze(-1) > 5.0)  # 5N力阈值，避免噪声干扰
        
        # 计算掩码加权后的对齐奖励
        alignment_reward = torch.sum(vertical_alignment * contact_mask, dim=-1) / (
            torch.sum(contact_mask, dim=-1) + 1e-6)
        
        # 可选：添加非线性增强
        alignment_reward = torch.where(
            alignment_reward > 0.8, 
            alignment_reward * 2.0,  # 高对齐区域奖励加倍
            alignment_reward * 0.5    # 低对齐区域奖励减半
        )
        
        return alignment_reward
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smooth(self):
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions
                + self.last_last_actions
            ),
            dim=1,
        )
    
    def _reward_hip_roll_default(self):
        hip_roll_differ = torch.mean(torch.square(self.dof_pos[:,[0,6]]),dim=-1)
        return torch.exp(-hip_roll_differ)
    
    def _reward_hip_roll_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[0,6]] * self.torques[:,[0,6]]),dim=1)
        return  torch.exp(-(1e-6*energy))
    
    def _reward_hip_pitch_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[2,8]] * self.torques[:,[2,8]]),dim=1)
        return  torch.exp(-(1e-6*energy))
    
    def _reward_knee_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[3,9]] * self.torques[:,[3,9]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation_pen(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_contact_momentum(self):
        return torch.sum(torch.abs(self.feet_vel[:,:,2]*self.contact_forces[:,self.feet_indices,2]),dim=1)
    
    # def _reward_feet_mass_scaled_contact_forces(self):
    #     return torch.sum(
    #         (
    #             self.contact_forces[:, self.feet_indices, 2]
    #             - self.base_mass.mean() * 9.8 / 2
    #         ).clip(min=0.0),
    #         dim=1,
    #     )
        
    def _reward_foot_landing_vel(self):
        z_vels = self.feet_vel[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5
        # about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        about_to_land = (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
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
    
    def _reward_ankle_torque_limits(self):
        ankle_torques = self.torques[:,[4,5,10,11]]
        ankle_torques_limit = 0.5*self.torque_limits[[4,5,10,11]]
        return torch.sum((torch.abs(ankle_torques) - ankle_torques_limit).clip(min=0.), dim=1)
    
    def _reward_hip_roll_yaw_torque_limits(self):
        hip_roll_yaw_torques = self.torques[:,[0,1,6,7]]
        hip_roll_yaw_torques_limit = 0.5*self.torque_limits[[0,1,6,7]]
        return torch.sum((torch.abs(hip_roll_yaw_torques) - hip_roll_yaw_torques_limit).clip(min=0.), dim=1)
    
    def _reward_feet_height_var(self):
        left = torch.var(self.left_feet_heights,dim=-1)
        right = torch.var(self.right_feet_heights,dim=-1)
        reward = torch.exp(-(1e3*(left + right)/2))
        return reward
    
    def _reward_ankle_pitch_limit(self):
        ankle_pitch_index = [4,10]
        picth_limit_pen = torch.sum((self.dof_pos[:,ankle_pitch_index] - 0.2).clip(max=0),dim=-1)
        return picth_limit_pen