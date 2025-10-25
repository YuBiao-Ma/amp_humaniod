
from humanoidGym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import glob


MOTION_FILES = glob.glob('humanoidGym/datasets/g1/mocap_motions/*')

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz  #62.8318
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  #14.2477
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2 #40.1771
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2 #99.0906
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  #16.7782

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ #0.9070
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ #2.5577
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ #6.3083
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ #1.0681

class G1UnevenAmpCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0, 0, 0.80] # x,y,z [m]
        orn = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_pitch_joint':      -0.1,
           'left_hip_roll_joint':        0.0,
           'left_hip_yaw_joint':         0.0,
           'left_knee_joint':            0.3,
           'left_ankle_pitch_joint':    -0.2,
           'left_ankle_roll_joint':      0.0,

            'right_hip_pitch_joint':    -0.1,
            'right_hip_roll_joint':      0.0,
            'right_hip_yaw_joint':       0.0,
            'right_knee_joint':          0.3,
            'right_ankle_pitch_joint':  -0.2,
            'right_ankle_roll_joint':    0.0,
          
            'waist_yaw_joint':           0.0,

            'left_shoulder_pitch_joint': 0.5,
            'left_shoulder_roll_joint':  0.2,
            'left_shoulder_yaw_joint':  -0.2,
            'left_elbow_joint':          0.3,
            
            'right_shoulder_pitch_joint':0.5,
            'right_shoulder_roll_joint':-0.2,
            'right_shoulder_yaw_joint':  0.2,
            'right_elbow_joint':         0.3,
        }
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_single_observations = 72
        num_critic_single_observations = 80
       
        num_actions = 21
        num_obs_lens = 1
        critic_num_obs_lens = 1
        num_observations = num_obs_lens * num_single_observations
        num_privileged_obs = num_critic_single_observations*critic_num_obs_lens + 187 
      
        forward_height_dim = 525
        reference_state_initialization = False
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
        
        add_action_lag = False
        action_lag_timesteps_range = [0,6]#[10,40]#[0,30]
        
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
        joint_armature_range = [0.008, 0.06]    
        
        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 1.0]
        joint_viscous_range = [0.1, 0.9]   
        
    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        # 525 dim, for depth image prediction
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                     2.0]  # 1mx1.6m rectangle (without center line)
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class depth:
        use_camera = True
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.03, 0.00, 0.35]  # front camera
        y_angle = [25, 30]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 38#58
        buffer_len = 2

        near_clip = 1
        far_clip = 3
        dis_noise = 0.0

        scale = 1
        invert = True

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'left_hip_pitch_joint': STIFFNESS_7520_14,
                     'left_hip_roll_joint':  STIFFNESS_7520_22,
                     'left_hip_yaw_joint': STIFFNESS_7520_14,
                     'left_knee_joint': STIFFNESS_7520_22,
                     'left_ankle_pitch_joint': 2.0 * STIFFNESS_5020,
                     'left_ankle_roll_joint': 2.0 * STIFFNESS_5020,
                     'right_hip_pitch_joint': STIFFNESS_7520_14,
                     'right_hip_roll_joint':  STIFFNESS_7520_22,
                     'right_hip_yaw_joint': STIFFNESS_7520_14,
                     'right_knee_joint': STIFFNESS_7520_22,
                     'right_ankle_pitch_joint': 2.0 * STIFFNESS_5020,
                     'right_ankle_roll_joint': 2.0 * STIFFNESS_5020,
                     'waist_yaw_joint': STIFFNESS_7520_14,
                     'left_shoulder_pitch_joint': STIFFNESS_5020,
                     'left_shoulder_roll_joint': STIFFNESS_5020,
                     'left_shoulder_yaw_joint': STIFFNESS_5020,
                     'left_elbow_joint': STIFFNESS_5020,
                     'right_shoulder_pitch_joint': STIFFNESS_5020,
                     'right_shoulder_roll_joint': STIFFNESS_5020,
                     'right_shoulder_yaw_joint': STIFFNESS_5020,
                     'right_elbow_joint': STIFFNESS_5020,
        }  # [N*m/rad]
        damping = {  'left_hip_pitch_joint': DAMPING_7520_14,
                     'left_hip_roll_joint':  DAMPING_7520_22,
                     'left_hip_yaw_joint': DAMPING_7520_14,
                     'left_knee_joint': DAMPING_7520_22,
                     'left_ankle_pitch_joint': 2.0 * DAMPING_5020,
                     'left_ankle_roll_joint': 2.0 * DAMPING_5020,
                     'right_hip_pitch_joint': DAMPING_7520_14,
                     'right_hip_roll_joint':  DAMPING_7520_22,
                     'right_hip_yaw_joint': DAMPING_7520_14,
                     'right_knee_joint': DAMPING_7520_22,
                     'right_ankle_pitch_joint': 2.0 * DAMPING_5020,
                     'right_ankle_roll_joint': 2.0 * DAMPING_5020,
                     'waist_yaw_joint': DAMPING_7520_14,
                     'left_shoulder_pitch_joint': DAMPING_5020,
                     'left_shoulder_roll_joint': DAMPING_5020,
                     'left_shoulder_yaw_joint': DAMPING_5020,
                     'left_elbow_joint': DAMPING_5020,
                     'right_shoulder_pitch_joint': DAMPING_5020,
                     'right_shoulder_roll_joint': DAMPING_5020,
                     'right_shoulder_yaw_joint': DAMPING_5020,
                     'right_elbow_joint': DAMPING_5020,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scales = [
                0.54754647, 0.35066147, 0.54754647, 0.35066147, 0.43857731, 0.43857731,
                0.54754647, 0.35066147, 0.54754647, 0.35066147, 0.43857731, 0.43857731,
                0.54754647,
                0.43857731, 0.43857731, 0.43857731, 0.43857731,
                0.43857731, 0.43857731, 0.43857731, 0.43857731]
        # action_scales = [
        #         0.54754647, 0.35066147, 0.54754647, 0.35066147, 0.43857731, 0.43857731,
        #         0.54754647, 0.35066147, 0.54754647, 0.35066147, 0.43857731, 0.43857731,
        #         0.01,
        #         0.01,  0.01,  0.01,  0.01,
        #          0.01,  0.01,  0.01,  0.01]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_filter = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/humanoidGym/resources/robots/g1/urdf/g1_29->21dof.urdf'
        foot_name = ["left_ankle_roll_link","right_ankle_roll_link"]
        penalize_contacts_on = ["pelvis_contour_link"]
        terminate_after_contacts_on = ["shoulder_pitch_link" ,"elbow_link","head_link","hip_yaw"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        collapse_fixed_joints = False
        flip_visual_attachments = False
        obs_mirror = ["commands","base_ang_vel","projected_gravity","dofs","dofs","dofs"]
    
    class commands:
        curriculum = True
        max_curriculum = 2.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute         
        class ranges:
            lin_vel_x = [0.0, 1.0] # min max [m/s]      
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [0, 0]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 1.0 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.78
        max_contact_force = 500
        reward_curriculum = True
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]
        # only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            base_acc = 0.2
            #alive = 0.05
            
            #正则化
            stand_still = -1
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
            # ankle_action_pitch = -0.05
            # ankle_action_roll = -0.1    
            yaw_error_when_rate_matches = -0.1
            stumble = -5.0
            # termination = -10
                
            dof_vel_limits = -0.1
            dof_pos_limits = -10.
            dof_torque_limits = -1

            feet_edge = -0.5
            cheat = -0.5
     
                
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

class G1UnevenAmpCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'WMPRunner'
    class policy:
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'

       
        
    class discriminator:
        reward_coef = 2.0#1.0
        reward_lerp = 0.5 # wasabi_reward = (1 - reward_lerp) * style_reward + reward_lerp * task_reward
        style_reward_function = "wasserstein_mapping" # log_mapping, quad_mapping, wasserstein_mapping wasserstein_tanh_mapping
        shape = [512, 256]

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005
        max_grad_norm = 1
        
        amp_replay_buffer_size = 1000000
        learning_rate = 3e-5
        discriminator_learning_rate = 3e-4
        discriminator_momentum = 0.5
        discriminator_weight_decay = 1e-3
        discriminator_gradient_penalty_coef = 10
        discriminator_loss_function = "PairwiseLoss" # MSELoss, BCEWithLogitsLoss, WassersteinLoss PairwiseLoss WassersteinTanhLoss 
        discriminator_num_mini_batches = 80
        
        class_name = 'WMPPPO'
        
        #Random Network Distillation
        # class rnd_cfg:
        #     weight= 1 # initial weight of the RND reward

        #     # note: This is a dictionary with a required key called "mode" which can be one of "constant" or "step".
        #     #   - If "constant", then the weight is constant.
        #     #   - If "step", then the weight is updated using the step scheduler. It takes additional parameters:
        #     #     - max_num_steps: maximum number of steps to update the weight
        #     #     - final_value: final value of the weight
        #     # If None, then no scheduler is used.
        #     weight_schedule=None

        #     reward_normalization=True  # whether to normalize RND reward
        #     gate_normalization=True  # whether to normalize RND gate observations

        #     # -- Learning parameters
        #     learning_rate=0.001  # learning rate for RND

        #     # -- Network parameters
        #     # note: if -1, then the network will use dimensions of the observation
        #     num_outputs=1  # number of outputs of RND network
        #     predictor_hidden_dims = [256,128] # hidden dimensions of predictor network
        #     target_hidden_dims = [256,128]  # hidden dimensions of target network

        # -- Symmetry Augmentation
        # class symmetry_cfg:
        #     use_data_augmentation=False  # this adds symmetric trajectories to the batch
        #     use_mirror_loss=False  # this adds symmetry loss term to the loss function
        #     # coefficient for symmetry loss term
        #     # if 0, then no symmetry loss is used
        #     mirror_loss_coeff=1
            
    class runner( LeggedRobotCfgPPO.runner ):
        empirical_normalization = False
        policy_class_name = "ActorCriticRecurrentWMP"
        max_iterations = 30000
        run_name = 'vaild_g1_amp'
        experiment_name = 'g1_image'
        
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        normalize_style_reward = True
    
    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100
        train_start_steps = 10000