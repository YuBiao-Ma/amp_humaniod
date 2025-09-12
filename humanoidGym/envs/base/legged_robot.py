import time
from warnings import WarningMessage
import numpy as np
import os
from collections import deque


from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil


import torch
from torch import Tensor
from typing import Tuple, Dict
import torch.nn.functional as F




from humanoidGym.envs.base.base_task import BaseTask
from humanoidGym.utils.math import wrap_to_pi
from humanoidGym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from humanoidGym.utils.isaacgym_utils import torch_rand_float_piecewise
from humanoidGym.utils.helpers import class_to_dict
from humanoidGym.utils.terrain import Terrain
from humanoidGym.utils.math import *
from .legged_robot_config import LeggedRobotCfg

from humanoidGym import GYM_ROOT_DIR

class LeggedRobot(BaseTask):
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
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        
    def exp_avg_filter(self, x, avg, alpha=0.8):
        """
        Simple exponential average filter
        """
        avg = alpha*x + (1-alpha)*avg
        return avg

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        # if self.cfg.control.use_filter:
        #     self.actions = self.exp_avg_filter(self.actions, self.last_actions)
            
        for _ in range(self.cfg.control.decimation):
            
            actions_scaled = self.actions * self.cfg.control.action_scale
 
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
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts = contact

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

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
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        
        self.refreshable_randomize_props(env_ids)
        self.refreshable_randomize_lag(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
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
            
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        # rigid shapes set rand friction and restitution
        if self.cfg.domain_rand.randomize_friction:
            for s in range(len(props)):
                props[s].friction = self.friction[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            for s in range(len(props)):
                props[s].restitution = self.restitution[env_id]
            
        return props

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

        for i in range(len(props)):
            if self.cfg.domain_rand.randomize_joint_friction:
                props["friction"][i] *= self.joint_friction_coeffs[env_id]
            if self.cfg.domain_rand.randomize_joint_damping:
                props["damping"][i] = self.joint_damping_coeffs[env_id]
            if self.cfg.domain_rand.randomize_joint_armature:
                props["armature"][i] = self.joint_armatures[env_id]
             
        return props

    def _process_rigid_body_props(self, props, env_id):
        
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass += self.payload_masses[env_id]
        # randomize center of mass
        if self.cfg.domain_rand.randomize_com:
            props[0].com += gymapi.Vec3(self.com_displacement[env_id,0],self.com_displacement[env_id,1],self.com_displacement[env_id,2])
         # randomize all link 
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                # low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                # inertia_scale = np.random.uniform(low_bound, high_bound)
                props[i].mass *= self.inertia_scale[env_id,i]
                props[i].inertia.x.x *= self.inertia_scale_x[env_id,i]
                props[i].inertia.y.y *= self.inertia_scale_y[env_id,i]
                props[i].inertia.z.z *= self.inertia_scale_z[env_id,i]    
        # self.base_mass[env_id] = props[0].mass
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            
        # if self.cfg.domain_rand.push_robots:
        #     self._push_robots()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
            
        
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

        # set small commands to zero,need to change here?
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # actions_scaled = actions * self.cfg.control.action_scale
 
        # if self.cfg.domain_rand.add_action_lag:
        #     self.action_lag_buffer[:,:,1:] = self.action_lag_buffer[:,:,:self.cfg.domain_rand.action_lag_timesteps_range[1]].clone()
        #     self.action_lag_buffer[:,:,0] = actions_scaled.clone()
        #     lagged_actions_scaled = self.action_lag_buffer[torch.arange(self.num_envs),:,self.action_lag_timestep.long()]
        # else:
        #     lagged_actions_scaled = actions_scaled
        
        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.p_gains*self.randomized_p_gains
            d_gains = self.d_gains*self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        torques = p_gains*(actions + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains*self.dof_vel
        
        if self.cfg.domain_rand.randomize_coulomb_friction:
            torques = torques - self.joint_viscous * self.dof_vel - self.joint_coulomb * torch.sign(self.dof_vel)

        if self.cfg.domain_rand.randomize_motor_strength:
            torques = torques*self.torque_multi
            
        if self.cfg.domain_rand.randomize_rfi:
            torques += self.torque_limits*torch_rand_float(self.cfg.domain_rand.rfi_st[0], self.cfg.domain_rand.rfi_st[1],(self.num_envs, self.num_dofs), device=self.device) + self.rfi_ep_offest
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos *  torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device) #self.init_joint_scale[env_ids] + self.init_joint_offset[env_ids]
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # def _push_robots(self):
    #     """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
    #     """
    #     env_ids = torch.arange(self.num_envs, device=self.device)
    #     push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
    #     if len(push_env_ids) == 0:
    #         return
    #     max_vel = self.cfg.domain_rand.max_push_vel_xy
    #     self.rand_push_force[:, :2] = torch_rand_float(
    #         -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
    #     self.root_states[:, 7:9] =  self.rand_push_force[:, :2] # lin vel x/y
        
    #     env_ids_int32 = push_env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                 gymtorch.unwrap_tensor(self.root_states),
    #                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
   
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def set_ppo_iter(self,value):
        self.ppo_iter = value
    
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
            self.measured_heights = self._get_heights()

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
    # -------------------------------domain rand ------------------------------------------------
    def add_randomize_priv_obs(self):
        randomize_priv = torch.cat([self.payload_masses,#1
                                    self.com_displacement, #3
                                    self.friction,#1
                                    self.torque_multi,#12
                                    self.randomized_d_gains,#1
                                    self.randomized_p_gains],dim=-1)
        return randomize_priv
        
    def init_randomize_props(self):
        ''' Initialize torch tensors for random properties
        '''
        if self.cfg.domain_rand.randomize_base_mass:
            self.payload_masses = torch_rand_float(self.cfg.domain_rand.added_mass_range[0], self.cfg.domain_rand.added_mass_range[1], (self.num_envs, 1), device=self.device)
        else:
            self.payload_masses = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        
        if self.cfg.domain_rand.randomize_com:
            self.com_x_displacement = torch_rand_float(self.cfg.domain_rand.com_x[0], self.cfg.domain_rand.com_x[1], (self.num_envs, 1), device=self.device)
            self.com_y_displacement = torch_rand_float(self.cfg.domain_rand.com_y[0], self.cfg.domain_rand.com_y[1], (self.num_envs, 1), device=self.device)
            self.com_z_displacement = torch_rand_float(self.cfg.domain_rand.com_z[0], self.cfg.domain_rand.com_z[1], (self.num_envs, 1), device=self.device)
            
            self.com_displacement = torch.cat([self.com_x_displacement,self.com_y_displacement,self.com_z_displacement],dim=-1)
        else:
            self.com_x_displacement = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.com_y_displacement = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.com_z_displacement = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            
            self.com_displacement = torch.cat([self.com_x_displacement,self.com_y_displacement,self.com_z_displacement],dim=-1)
            
        if self.cfg.domain_rand.randomize_friction:
            self.friction = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (self.num_envs, 1), device=self.device)
        else:
            self.friction = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False) 
                    
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution = torch_rand_float(self.cfg.domain_rand.restitution_range[0],self.cfg.domain_rand.restitution_range[1],(self.num_envs,1),device=self.device)
        else:
            self.restitution = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)   
        
        if self.cfg.domain_rand.randomize_inertia:
            self.inertia_scale = torch_rand_float(self.cfg.domain_rand.randomize_inertia_range[0], self.cfg.domain_rand.randomize_inertia_range[1], (self.num_envs, self.num_bodies), device=self.device)
            self.inertia_scale_x = torch_rand_float(self.cfg.domain_rand.randomize_inertia_range[0], self.cfg.domain_rand.randomize_inertia_range[1], (self.num_envs, self.num_bodies), device=self.device)
            self.inertia_scale_y = torch_rand_float(self.cfg.domain_rand.randomize_inertia_range[0], self.cfg.domain_rand.randomize_inertia_range[1], (self.num_envs, self.num_bodies), device=self.device)
            self.inertia_scale_z = torch_rand_float(self.cfg.domain_rand.randomize_inertia_range[0], self.cfg.domain_rand.randomize_inertia_range[1], (self.num_envs, self.num_bodies), device=self.device)
        else:
            self.inertia_scale = torch.ones(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device,requires_grad=False)
            
        if self.cfg.domain_rand.randomize_joint_friction:                      
            self.joint_friction_coeffs = torch_rand_float(self.cfg.domain_rand.joint_friction_range[0], self.cfg.domain_rand.joint_friction_range[1], (self.num_envs, 1), device=self.device)
        
        if self.cfg.domain_rand.randomize_joint_damping:
            self.joint_damping_coeffs = torch_rand_float(self.cfg.domain_rand.joint_damping_range[0], self.cfg.domain_rand.joint_damping_range[1], (self.num_envs, 1), device=self.device)
        
        if self.cfg.domain_rand.randomize_joint_armature:
            self.joint_armature_coeffs = torch_rand_float(self.cfg.domain_rand.joint_armature_range[0], self.cfg.domain_rand.joint_armature_range[1], (self.num_envs, self.num_dof), device=self.device)
        
    def init_post_randomize_props(self):
        ''' Initialize torch tensors for random properties
        '''
        if self.cfg.domain_rand.randomize_motor_strength:
            self.torque_multi = torch_rand_float(self.cfg.domain_rand.motor_strength[0], self.cfg.domain_rand.motor_strength[1], (self.num_envs, self.num_actions), device=self.device)
        else:
            self.torque_multi = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        
        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
            self.randomized_d_gains = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        else:
            self.randomized_p_gains = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.randomized_d_gains = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            
        if self.cfg.domain_rand.randomize_init_joint_offset:
            self.init_joint_offset = torch_rand_float(self.cfg.domain_rand.init_joint_offset[0], self.cfg.domain_rand.init_joint_offset[1], (self.num_envs, self.num_dofs), device=self.device)
        else:
            self.init_joint_offset = torch.zeros(self.num_envs,self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            
        if self.cfg.domain_rand.randomize_init_joint_scale:
            self.init_joint_scale = torch_rand_float(self.cfg.domain_rand.init_joint_scale[0], self.cfg.domain_rand.init_joint_scale[1],(self.num_envs, self.num_dofs), device=self.device)
        else:
            self.init_joint_scale = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)
        
        if self.cfg.domain_rand.randomize_rfi:
            self.rfi_ep_offest = torch_rand_float(self.cfg.domain_rand.rfi_ep[0], self.cfg.domain_rand.rfi_ep[1],(self.num_envs, self.num_dofs), device=self.device)
            self.rfi_ep_offest *= self.torque_limits
        else:
            self.rfi_ep_offset = torch.zeros(self.num_envs,self.num_dofs,dtype=torch.float,device=self.device,requires_grad=False)
            
        if self.cfg.domain_rand.randomize_motor_zero_offset:
            self.motor_zero_offsets = torch_rand_float(self.cfg.domain_rand.motor_zero_offset_range[0], self.cfg.domain_rand.motor_zero_offset_range[1], (self.num_envs,self.num_actions), device=self.device)
        else:
            self.motor_zero_offsets = torch.zeros(self.num_envs,self.num_actions,dtype=torch.float,device=self.device,requires_grad=False)

        if self.cfg.domain_rand.randomize_coulomb_friction:
            joint_coulomb_range = self.cfg.domain_rand.joint_coulomb_range
            joint_viscous_range = self.cfg.domain_rand.joint_viscous_range
            
            self.joint_coulomb = torch_rand_float(joint_coulomb_range[0],joint_coulomb_range[1],(self.num_envs, self.num_actions), device=self.device)
            self.joint_viscous = torch_rand_float(joint_viscous_range[0],joint_viscous_range[1],(self.num_envs, self.num_actions), device=self.device)

    def init_randomize_lag(self):
        # action lag
        if self.cfg.domain_rand.add_action_lag:   
            self.action_lag_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.action_lag_timesteps_range[1]+1,device=self.device)
            self.action_lag_timestep = torch.randint(self.cfg.domain_rand.action_lag_timesteps_range[0],
                                                  self.cfg.domain_rand.action_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device) 
        else:
            self.action_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.action_lag_timesteps_range[0]
        
        # # prop lag
        # if self.cfg.domain_rand.add_prop_lag:
        #     self.prop_lag_buffer = torch.zeros(self.num_envs,self.num_obs,self.cfg.domain_rand.prop_lag_timesteps_range[1]+1,device=self.device)
        #     self.prop_lag_timestep = torch.randint(self.cfg.domain_rand.prop_lag_timesteps_range[0],
        #                                             self.cfg.domain_rand.prop_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
        # else:
        #     self.prop_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.prop_lag_timesteps_range[1]

    def refreshable_randomize_lag(self,env_ids):
        # action lag
        if self.cfg.domain_rand.add_action_lag:   
            self.action_lag_buffer[env_ids,:,:] = 0.0
            self.action_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.action_lag_timesteps_range[0],
                                                       self.cfg.domain_rand.action_lag_timesteps_range[1]+1,
                                                       (len(env_ids),),device=self.device) 
        # # prop lag
        # if self.cfg.domain_rand.add_prop_lag:
        #     self.prop_lag_buffer[env_ids, :, :] = 0.0
        #     self.prop_lag_timestep = torch.randint(self.cfg.domain_rand.prop_lag_timesteps_range[0],
        #                                             self.cfg.domain_rand.prop_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
    
    # def refreshable_randomize_props(self,env_ids):
    #     if self.cfg.domain_rand.randomize_motor_strength:
    #         self.torque_multi = torch_rand_float(self.cfg.domain_rand.motor_strength[0], self.cfg.domain_rand.motor_strength[1], (self.num_envs, self.num_actions), device=self.device)
     
    #     if self.cfg.domain_rand.randomize_gains:
    #         self.randomized_p_gains = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
    #         self.randomized_d_gains = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)

    #     if self.cfg.domain_rand.randomize_rfi:
    #         self.rfi_ep_offest = torch_rand_float(self.cfg.domain_rand.rfi_ep[0], self.cfg.domain_rand.rfi_ep[1],(self.num_envs, self.num_dofs), device=self.device)
    #         self.rfi_ep_offest *= self.torque_limits
    
    def refreshable_randomize_props(self,env_ids):
        if self.cfg.domain_rand.randomize_motor_strength:
            self.torque_multi[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength[0], self.cfg.domain_rand.motor_strength[1], (len(env_ids), self.num_actions), device=self.device)
     
        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self.randomized_d_gains[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)

        if self.cfg.domain_rand.randomize_rfi:
            self.rfi_ep_offest[env_ids] = torch_rand_float(self.cfg.domain_rand.rfi_ep[0], self.cfg.domain_rand.rfi_ep[1],(len(env_ids), self.num_dofs), device=self.device)
            self.rfi_ep_offest[env_ids] *= self.torque_limits
            
        if self.cfg.domain_rand.randomize_coulomb_friction:
            joint_coulomb_range = self.cfg.domain_rand.joint_coulomb_range
            joint_viscous_range = self.cfg.domain_rand.joint_viscous_range
            
            self.joint_coulomb[env_ids] = torch_rand_float(joint_coulomb_range[0],joint_coulomb_range[1],(len(env_ids), self.num_actions), device=self.device)
            self.joint_viscous[env_ids] = torch_rand_float(joint_viscous_range[0],joint_viscous_range[1],(len(env_ids), self.num_actions), device=self.device)

            
    # -------------------------------domain rand ------------------------------------------------
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

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

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
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

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        ''' Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        '''
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.


    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
#----------------------- terrian related ------------------------------
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

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

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        # get (x,y) coordinate based on measured_points
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
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
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            # height_points is in base frame
            # points is in world frame
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        # points position to int row col
        points = (points/self.terrain.cfg.horizontal_scale).long()

        # px get all robot points(x)
        px = points[:, :, 0].view(-1)
        # py get all robot points(y)
        py = points[:, :, 1].view(-1)
        # clip within whole terrain
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # select (x,y) (x+1,y) (x,y+1), min height as heights
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heightXBotL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightXBotL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_base_heights(self, env_ids=None):
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
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


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

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _init_feet_height_points(self):
        """为每只脚定义采样点（在脚部局部坐标系中),需要对每个urdf专门设计"""
        
        # 在X轴（前后方向）和Y轴（左右方向）均匀采样
        y = torch.tensor([-0.05, 0., 0.05], device=self.device)
        x = torch.tensor([-0.125, -0.05, 0., 0.05, 0.125], device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)
        
        self.num_foot_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_foot_height_points, 3, device=self.device)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_left_feet_heights(self,env_ids=None):
      
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, 0, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        if env_ids:
            points = quat_apply_yaw(self.feet_quat[env_ids,0,:].repeat(1, self.num_foot_height_points), self.feet_height_points[env_ids]) + (self.feet_pos[env_ids,0,:]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.feet_quat[:,0,:].repeat(1, self.num_foot_height_points), self.feet_height_points) + (self.feet_pos[:,0,:]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]

        heights = (heights1 + heights2 + heights3) / 3
        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, 0, 2].view(self.num_envs,-1) - heights
        return feet_height
    
    def _get_right_feet_heights(self,env_ids=None):
      
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, 1, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.feet_quat[env_ids,1,:].repeat(1, self.num_foot_height_points), self.feet_height_points[env_ids]) + (self.feet_pos[env_ids,1,:]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.feet_quat[:,1,:].repeat(1, self.num_foot_height_points), self.feet_height_points) + (self.feet_pos[:,1,:]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]

        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        feet_height =  self.feet_pos[:, 1, 2].view(self.num_envs,-1) - heights

        return feet_height

    #------------ reward functions----------------
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
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_base_height_uneven(self):
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

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
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             3 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    
    def _reward_vertical_contact(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2),dim=-1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

 