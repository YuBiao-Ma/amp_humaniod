from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from yaml import warnings

from humanoidGym.algo.ppo.replay_buffer import ReplayBuffer
from humanoidGym.algo.ppo.rnd import RandomNetworkDistillation
from humanoidGym.algo.ppo.utils import data_augmentation_func, smooth_transition, string_to_callable
from humanoidGym.utils.helpers import exponential_progress

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

class AMPPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        discriminator,
        amp_data,
        amp_state_normalizer,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        # RND parameters
        rnd_cfg: Union[dict, None] = None,
        # Symmetry parameters
        symmetry_cfg: Union[dict, None] = None,
        
        # amp parameters
        discriminator_learning_rate=0.000025,
        discriminator_momentum=0.9,
        discriminator_weight_decay=0.0005,
        discriminator_gradient_penalty_coef=5,
        discriminator_loss_function="MSELoss",
        discriminator_num_mini_batches=10,
        amp_replay_buffer_size=100000,
        
        **kwargs
    ):
        # params = locals()
        # print("所有参数的值:")
        # for key, value in params.items():
        #     print(f"{key}: {value}")
        if kwargs:
            print("PPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()]))
                
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        
            
        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            # if not use_symmetry:
            #     # warnings.warn("Symmetry not used for learning. We will use it for logging instead.")
            # # If function is a string then resolve it to a function
            # if isinstance(symmetry_cfg["data_augmentation_func"], str):
            #     symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # # Check valid configuration
            # if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
            #     raise ValueError(
            #         "Data augmentation enabled but the function is not callable:"
            #         f" {symmetry_cfg['data_augmentation_func']}"
            #     )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.policy_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        
        self.amp_storage = ReplayBuffer(
            discriminator.observation_dim,
            amp_replay_buffer_size, 
            device
        )
        
        self.amp_data = amp_data
        self.amp_state_normalizer = amp_state_normalizer

        # Discriminator parameters
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_momentum = discriminator_momentum
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_loss_function = discriminator_loss_function
        self.discriminator_num_mini_batches = discriminator_num_mini_batches

        if self.discriminator_loss_function == "WassersteinLoss" or self.discriminator_loss_function == "PairwiseLoss":
            discriminator_optimizer = optim.AdamW #optim.RMSprop
        else:
            discriminator_optimizer = optim.SGD
        self.discriminator_optimizer = discriminator_optimizer(
                                                    self.discriminator.parameters(),
                                                    lr=self.discriminator_learning_rate,
                                                    # momentum=self.discriminator_momentum,
                                                    weight_decay=self.discriminator_weight_decay,
                                                )

        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            rnd_state_shape,
            self.device,
        )
        
        # self.storage.init_bootstrapping_data()

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, amp_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        # record amp obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs, next_critic_obs):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # record next critic obs for next frame prediction
        self.transition.next_critic_observations = next_critic_obs.clone()

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.amp_storage.insert(self.amp_transition.observations,amp_obs)
        self.amp_transition.clear()
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        
    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch):
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss
    
    def update(self,iter):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_subtask_loss = 0
        mean_smooth_loss = 0
        
        # -- AMP loss
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
            
            subtask_data
        ) in generator:
            
            hidden_states = [hid_states_batch[0],hid_states_batch[2]]
            self.actor_critic.act(obs_batch,masks=masks_batch, hidden_states=hidden_states)
    
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.policy_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() 

            # # add subtask update in actor critic model
            if hasattr( self.actor_critic, 'update' ) and callable(self.actor_critic.update):
                #subtask_loss = self.actor_critic.subtask_loss(obs_batch,critic_obs_batch[:,:3])#self.actor_critic.update(obs_batch,critic_obs_batch[:,:3])
                subtask_loss = self.actor_critic.subtask_loss(subtask_data)
                loss+=subtask_loss
                mean_subtask_loss += subtask_loss.item()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch)
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding.detach())
                
            # -- For PPO
            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_smooth_loss += 0.0#smooth_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            # if mean_symmetry_loss is not None:
            #     mean_symmetry_loss += symmetry_loss.item()
            
        amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.discriminator_num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches)
        
        amp_expert_generator = self.amp_data.feed_forward_generator(
                self.discriminator_num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches)
        
        for sample_amp_policy, sample_amp_expert in zip(amp_policy_generator,amp_expert_generator):
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_state_normalizer is not None:
                with torch.no_grad():# 标准化的同时进行更新
                    policy_state = self.amp_state_normalizer(policy_state)
                    policy_next_state = self.amp_state_normalizer(policy_next_state)
                    expert_state = self.amp_state_normalizer(expert_state)
                    expert_next_state = self.amp_state_normalizer(expert_next_state)
            policy_d = self.discriminator(torch.cat([policy_state,policy_next_state],dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state,expert_next_state],dim=-1))
            if self.discriminator_loss_function == "BCEWithLogitsLoss":
                expert_loss = torch.nn.BCEWithLogitsLoss()(expert_d, torch.ones_like(expert_d))
                policy_loss = torch.nn.BCEWithLogitsLoss()(policy_d, torch.zeros_like(policy_d))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    *sample_amp_expert, lambda_=self.discriminator_gradient_penalty_coef)
                
            elif self.discriminator_loss_function == "MSELoss":
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    *sample_amp_expert, lambda_=self.discriminator_gradient_penalty_coef)
                
            elif self.discriminator_loss_function == "WassersteinLoss":
                expert_loss = -expert_d.mean()
                policy_loss = policy_d.mean()
                amp_loss = 0.5 * (expert_loss + policy_loss)
                # grad_pen_loss = self.discriminator.compute_grad_pen(
                #     *sample_amp_expert, lambda_=self.discriminator_gradient_penalty_coef)
                grad_pen_loss = self.discriminator.compute_wgan_pen(
                    *sample_amp_expert, *sample_amp_policy,lambda_=self.discriminator_gradient_penalty_coef)
                
            elif self.discriminator_loss_function == "PairwiseLoss":
                
                #amp_loss =  -torch.nn.functional.logsigmoid(F.tanh(eta*expert_d) - F.tanh(eta*policy_d)).mean()
                # amp_loss =  -torch.nn.functional.logsigmoid(expert_d - policy_d).mean()
                grad_pen_loss = self.discriminator.compute_wgan_pen(
                     *sample_amp_expert, *sample_amp_policy, lambda_=self.discriminator_gradient_penalty_coef)
                # grad_pen_loss = self.discriminator.compute_pair_pen(
                #      *sample_amp_expert, *sample_amp_policy, lambda_=self.discriminator_gradient_penalty_coef)
                #compute_grad_pen
                # grad_pen_loss = self.discriminator.compute_grad_pen(
                #      *sample_amp_expert, lambda_=self.discriminator_gradient_penalty_coef)
                
                diff = expert_d - policy_d
                amp_loss = F.softplus(-diff).mean()
                
                # eta = 0.5
                # diff = F.tanh(eta*expert_d) - F.tanh(eta*policy_d)
                # amp_loss = F.softplus(-diff).mean()
                # grad_pen_loss = self.discriminator.compute_pair_pen(
                #      *sample_amp_expert, *sample_amp_policy, lambda_=self.discriminator_gradient_penalty_coef)
                
            elif self.discriminator_loss_function == "PairwiseTanhLoss":
                
                #amp_loss =  -torch.nn.functional.logsigmoid(F.tanh(eta*expert_d) - F.tanh(eta*policy_d)).mean()
                # amp_loss =  -torch.nn.functional.logsigmoid(expert_d - policy_d).mean()
                # grad_pen_loss = self.discriminator.compute_wgan_pen(
                #      *sample_amp_expert, *sample_amp_policy, lambda_=self.discriminator_gradient_penalty_coef)
                # grad_pen_loss = self.discriminator.compute_pair_pen(
                #      *sample_amp_expert, *sample_amp_policy, lambda_=self.discriminator_gradient_penalty_coef)
                #compute_grad_pen
                grad_pen_loss = self.discriminator.compute_grad_pen(
                     *sample_amp_expert, lambda_=self.discriminator_gradient_penalty_coef)
                eta = 0.1
                diff = F.tanh(eta*expert_d) - F.tanh(eta*policy_d)
                amp_loss = F.softplus(-diff).mean()
                
                # eta = 0.5
                # diff = F.tanh(eta*expert_d) - F.tanh(eta*policy_d)
                # amp_loss = F.softplus(-diff).mean()
                # grad_pen_loss = self.discriminator.compute_pair_pen(
                #      *sample_amp_expert, *sample_amp_policy, lambda_=self.discriminator_gradient_penalty_coef)
                
            elif self.discriminator_loss_function == "WassersteinTanhLoss":
                eta = 0.1
                expert_loss = -F.tanh(eta*expert_d).mean()
                policy_loss = F.tanh(eta*policy_d).mean()
                amp_loss = 0.5 * (expert_loss + policy_loss)
                # grad_pen_loss = self.discriminator.compute_grad_pen(
                #     *sample_amp_expert, lambda_=self.discriminator_gradient_penalty_coef)
                grad_pen_loss = self.discriminator.compute_wgan_pen(
                    *sample_amp_expert, *sample_amp_policy,lambda_=self.discriminator_gradient_penalty_coef)
                #compute_wgan_slack_pen
                # grad_pen_loss = self.discriminator.compute_wgan_slack_pen(
                #     *sample_amp_expert, *sample_amp_policy,lambda_=self.discriminator_gradient_penalty_coef)
            else:
                raise ValueError("Unexpected loss function specified")

            
            discriminator_loss = amp_loss + grad_pen_loss
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            
            # print(self.amp_state_normalizer.training)
            # if self.amp_state_normalizer is not None:
            #     states = torch.cat([policy_state,expert_state],dim=0)
            #     self.amp_state_normalizer.update(states)
                
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            
        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_subtask_loss /= num_updates
        mean_smooth_loss /= num_updates
        
        # -- For AMP
        discriminator_num_updates = self.discriminator_num_mini_batches
        mean_amp_loss /= discriminator_num_updates
        mean_grad_pen_loss /= discriminator_num_updates
        mean_policy_pred /= discriminator_num_updates
        mean_expert_pred /= discriminator_num_updates

        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        # if mean_symmetry_loss is not None:
        #     mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss,mean_subtask_loss,mean_smooth_loss,mean_amp_loss,mean_grad_pen_loss,mean_policy_pred,mean_expert_pred
