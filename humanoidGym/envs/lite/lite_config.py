
from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import glob

MOTION_FILES = glob.glob('humanoidGym/datasets/lite/motion_amp_expert/new_data/*')


class LiteAmpCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0, 0, 0.72] # x,y,z [m]
        orn = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = {
            'joint_left_hip_pitch': 0.39,   # [rad]
            'joint_left_hip_roll':  -0.0,   # [rad]
            'joint_left_hip_yaw':   -0.12,   # [rad]
            'joint_left_knee':  0.74,   # [rad]
            'joint_left_ankle_pitch': 0.36,   # [rad] 
            'joint_left_ankle_roll': 0,   # [rad]

            'joint_right_hip_pitch': 0.39,   # [rad]
            'joint_right_hip_roll':  -0.0,   # [rad]
            'joint_right_hip_yaw':   -0.12,   # [rad]
            'joint_right_knee':  0.74,   # [rad]
            'joint_right_ankle_pitch': 0.36,   # [rad]
            'joint_right_ankle_roll': 0,   # [rad]
        }
    
    class env(LeggedRobotCfg.env):
        num_envs = 5480
        num_single_observations = 45  #num_dof*3 + 9
        num_critic_single_observations = 53
       
        num_actions = 12
        num_obs_lens = 1
        critic_num_obs_lens = 1
        num_observations = num_obs_lens * num_single_observations
        num_privileged_obs = num_critic_single_observations*critic_num_obs_lens #+ 187 
        
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        

    class domain_rand(LeggedRobotCfg.domain_rand):
        
        randomize_friction = True
        friction_range = [0.25, 1.75]
        
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.3
        
        randomize_com = True
        com_x = [-0.03,0.03]
        com_y = [-0.03,0.03]
        com_z = [-0.03,0.03]
        
        randomize_motor_strength = True
        motor_strength = [0.8,1.2]
        
        randomize_gains = True
        kp_range = [0.8,1.2]
        kd_range = [0.8,1.2]
        
        add_action_lag = True
        action_lag_timesteps_range = [0,5]#[10,40]#[0,30]
        max_lag_timesteps = 30
        
        randomize_restitution = False
        restitution_range = [0.0,1.0]
        
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]         
         
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]

        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]

        randomize_joint_armature = True
        joint_armature_range = [0.8, 1.2]    
        
        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 1.0]
        joint_viscous_range = [0.1, 0.9]  
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        #mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [flat,up slope,down slope,discrite]
        terrain_proportions = [0.2, 0.3, 0.3, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {

            'hip_pitch': 150, #277.56
            'hip_roll': 400, #235.42
            'hip_yaw': 150, #153.61           
            'knee': 150, #277.56
            'ankle_pitch': 30, #20.,
            'ankle_roll': 30, #20.,
        }  # [N*m/rad]

        damping = {
                 
         
            'hip_pitch': 8,# 14.08,
            'hip_roll': 10, # 11.92,
            'hip_yaw': 8, # 9.82,
            'knee': 8, # 14.08,
            'ankle_pitch': 4, #1,
            'ankle_roll': 4, #1,
        }

        action_scales = 0.25
      
        decimation = 4
        use_filter = False
        exp_avg_decay = 0.05

    class asset( LeggedRobotCfg.asset ):
       
        
        
        file = '{LEGGED_GYM_ROOT_DIR}/humanoidGym/resources/robots/miniloong_v4_12dof/urdf/miniloong_v4_12dof_c_new_V41_1.urdf'
        name = "miniloong"
        foot_name = ["link_left_ankle_roll","link_right_ankle_roll"]
        knee_name = ["link_left_knee","link_right_knee"]
        # penalize_contacts_on = ["hip_yaw", "knee"]
        penalize_contacts_on = ["knee"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = False
        obs_mirror = ["commands","base_ang_vel","projected_gravity","dofs","dofs","dofs"]
      

        joint_amatures = {
            'hip_pitch': 0.07031,# 0.000488265*12**2
            'hip_roll': 0.05963, # 0.000184053*18**2
            'hip_yaw': 0.03891, # 0.000062254*25**2
            'knee': 0.07031, # 0.000488265*12**2
            'ankle_pitch': 0.03891,#0.000062254*25**2
            'ankle_roll': 0.03891, #0.000062254*25**2
        }
    
    
    class commands:
        curriculum = True
        max_curriculum = 4
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False#True # if true: compute         
        class ranges:
            lin_vel_x = [-1, 1] # min max [m/s]            
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 1
        base_height_target = 0.76
        max_contact_force = 500
        # only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            base_acc = 0.2
            #alive = 0.05
           
            #正则化
            stand_still = -1
            orientation = -1
            base_height = 0.2
            # action_rate = -0.01
            action_smooth = -0.01
            foot_slip = -0.1
            feet_contact_forces = 0.01
            # exp_energy = 0.05
            power_dist = -2e-5
            torques = -0.00001
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            ankle_pitch_energy = 0.1
            ankle_roll_energy = 0.1
            
            # hip_action_pitch = -0.05
            # ankle_action_roll = -0.1     
                
            dof_vel_limits = -0.1
            dof_pos_limits = -10.
            dof_torque_limits = -1
            
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LiteAmpCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AmpOnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        activation = 'elu'
        
    class discriminator:
        reward_coef = 2.0
        reward_lerp = 0.5 # wasabi_reward = (1 - reward_lerp) * style_reward + reward_lerp * task_reward
        style_reward_function = "wasserstein_mapping" # log_mapping, quad_mapping, wasserstein_mapping wasserstein_tanh_mapping
        shape = [512, 256]

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        max_grad_norm = 1
        
        amp_replay_buffer_size = 1000000
        learning_rate = 3e-5
        discriminator_learning_rate = 3e-4
        discriminator_momentum = 0.5
        discriminator_weight_decay = 1e-3
        discriminator_gradient_penalty_coef = 10
        discriminator_loss_function = "PairwiseLoss" # MSELoss, BCEWithLogitsLoss, WassersteinLoss PairwiseLoss WassersteinTanhLoss 
        discriminator_num_mini_batches = 80
        
        class_name = 'AMPPPO'
        
       
    class runner( LeggedRobotCfgPPO.runner ):
        empirical_normalization = True
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 20000
        # run_name = 'vaild_g1_amp'
        experiment_name = 'miniloong'
        
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        normalize_style_reward = True
        