
from humanoidGym import GYM_ROOT_DIR
# from humanoidGym.algo.dataset.motion_loader import AMPLoader
from humanoidGym.algo.dataset.lite_motion_loader import LongAMPLoader
from humanoidGym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import os
import random
import time
from humanoidGym.algo.ppo.utils import build_mirror_ls
from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg
from humanoidGym.utils import exponential_progress

from humanoidGym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


class LiteRobot(LeggedRobot):
    
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

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # amp related
        if self.cfg.env.reference_state_initialization:
            self.amp_loader = LongAMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

        self.discriminator = None # assigned in runner
        self.amp_state_normalizer = None # assigned in runner
        self.style_reward_normalizer = None # assigned in runner
        self.cur_amp_state_obs = None # assigned in runner
        
    def update_current_amp_state(self,obs):
        self.cur_amp_state_obs = obs.clone().detach()
    

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
     

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            armature_val = 0.0
            for key in self.cfg.asset.joint_amatures.keys():
                if key in name:
                    armature_val = self.cfg.asset.joint_amatures[key]
                    break
            
            props["armature"][i] = armature_val

        for i in range(len(props)):
            if self.cfg.domain_rand.randomize_joint_friction:
                props["friction"][i] *= self.joint_friction_coeffs[env_id]
            if self.cfg.domain_rand.randomize_joint_damping:
                props["damping"][i] *= self.joint_damping_coeffs[env_id]
            if self.cfg.domain_rand.randomize_joint_armature:
                props["armature"][i] *= self.joint_armature_coeffs[env_id,i]
                # props["armature"][i] = self.joint_armature_coeffs[env_id]

        return props
    

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

        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0])
        print("joint armatures:", dof_props['armature'])  # 看是不是你设置的 [0.07031, 0.059633, ...]


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
        ee_names = ['link_left_ankle_roll', 'link_right_ankle_roll']
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
        
        return noise_vec
    
    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        
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
        super()._init_buffers()
        self._init_foot()
        self._init_mirror()
        self._init_action_scales()
        
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

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
            
        for _ in range(self.cfg.control.decimation):
            
            actions_scaled = self.actions * self.action_scales
 
            if self.cfg.domain_rand.add_action_lag:
                self.action_lag_buffer[:,:,1:] = self.action_lag_buffer[:,:,:self.cfg.domain_rand.max_lag_timesteps].clone()
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

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf,self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
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
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts = contact
        
        return env_ids, termination_privileged_obs
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>0.7, torch.abs(self.rpy[:,0])>0.7)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
    
    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = LongAMPLoader.get_joint_angles_batch(frames)
        self.dof_vel[env_ids] = LongAMPLoader.get_joint_velocities_batch(frames)
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
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            root_pos = LongAMPLoader.get_root_pos_batch(frames)
            root_pos += self.env_origins[env_ids]
            self.root_states[env_ids, :3] = root_pos
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            root_pos = LongAMPLoader.get_root_pos_batch(frames)
            root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
            self.root_states[env_ids, :3] = root_pos
            
        root_orn = LongAMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, LongAMPLoader.get_base_vel_batch(frames)[:,:3])
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, LongAMPLoader.get_base_vel_batch(frames)[:,3:6])

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

            if self.cfg.domain_rand.add_action_lag:
                self.update_action_lag_curriculum(env_ids)
            
        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
          
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

        if self.cfg.domain_rand.add_action_lag:
            self.extras["episode"]["max_action_lag_timestep"] = self.action_lag_timesteps_range[1]
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

    def update_action_lag_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.action_lag_timesteps_range[1] = np.clip(self.action_lag_timesteps_range[1] + 5, 0., self.cfg.domain_rand.max_lag_timesteps)
        
    
            
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
        # —— 把 base_quat 重复成对应的 [N*2, 4] —— 
        quat_rep = self.base_quat.unsqueeze(1).repeat(1, 2, 1).view(-1, 4)  # [N*4, 4]
        # —— 用 quat_rotate_inverse 旋转到 body frame —— 
        foot_body_flat = quat_rotate_inverse(quat_rep, foot_rel_flat)    # [N*4, 3]
        # —— 再 reshape 回 [N, 6] —— 
        foot_pos = foot_body_flat.view(self.num_envs, 6)  
        left_foot_rpy =  get_euler_xyz_in_tensor(self.rigid_state[:,self.feet_indices[0], 3:7])
        right_foot_rpy = get_euler_xyz_in_tensor(self.rigid_state[:,self.feet_indices[1], 3:7])

        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        z_pos = self.root_states[:, 2:3]
        gravity =self.projected_gravity

        return torch.cat((joint_pos, joint_vel,foot_pos, base_lin_vel, base_ang_vel, z_pos), dim=-1)
    
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
        return self.obs_buf, self.extras
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
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
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
    
    def _reward_hip_action_pitch(self):
        return torch.sum(torch.square(self.actions[:, [0,6]]), dim=1)* (torch.norm(self.commands[:, :2], dim=1) < 0.5)

    
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
