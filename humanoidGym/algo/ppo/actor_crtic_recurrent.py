import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from humanoidGym.algo.ppo.modules import Memory, RnnActor, RnnBaselineActor, RnnEstVelActor, RnnEstVelHeightActor, RnnEstVelHeightContactSiamActor, RnnEstVelHeightContactSiamNormActor, RnnEstVelHeightMorePrivSiamNormActor, RnnEstVelHeightPrivSiamActor, RnnEstVelHeightPrivSiamNormActor, RnnEstVelHeightSiamActor, RnnEstVelHeightSiamNormActor, RnnNextLatentActor, RnnNextSiamNormActor, RnnSimpleEstVelActor, mlp_factory
from humanoidGym.algo.ppo.utils import unpad_trajectories
from .actor_critic import ActorCritic, get_activation
import torch.optim as optim
import os
import copy

# class InferenceActorLSTM(torch.nn.Module):
#     def __init__(self, actor, norm_module):
#         super().__init__() 
#         # actor
#         self.actor = copy.deepcopy(actor)
#         # normalize
#         self.norm_module = copy.deepcopy(norm_module)
        
#         # estimator memory
#         self.register_buffer(f'est_hidden_state', torch.zeros(1, 1, 256))
#         self.register_buffer(f'est_cell_state', torch.zeros(1, 1, 256))
        
#         # actor memory
#         self.register_buffer(f'hidden_state', torch.zeros(1, 1, 256))
#         self.register_buffer(f'cell_state', torch.zeros(1, 1, 256))
        
#     def forward(self, x):
#         # normalize 
#         x = self.norm_module(x)
#         x = x.unsqueeze(0)
        
#         mean, (est_h,est_c), (h,c) = self.actor.depoly_forward(x, (self.est_hidden_state,self.est_cell_state),(self.hidden_state,self.cell_state))
#         self.est_hidden_state[:] = est_h
#         self.est_cell_state[:] = est_c
        
#         self.hidden_state[:] = h 
#         self.cell_state[:] = c

#         return mean 

#     @torch.jit.export
#     def reset_memory(self):
#         self.hidden_state[:] = 0.
#         self.cell_state[:] = 0.
        
#         self.est_hidden_state[:] = 0.
#         self.est_cell_state[:] = 0.

class InferenceActorLSTM(torch.nn.Module):
    def __init__(self, actor, norm_module):
        super().__init__() 
        # actor
        self.actor = actor
        # normalize
        self.norm_module = norm_module
        
    def forward(self, x, est_h_prev, est_c_prev, h_prev, c_prev):
        # normalize 
        x = self.norm_module(x)
        x = x.unsqueeze(0)
        mean, (est_h,est_c), (h,c) = self.actor.depoly_forward(x, (est_h_prev,est_c_prev),(h_prev,c_prev))
        return mean, est_h, est_c, h, c

class ActorCriticRecurrent(nn.Module):
    is_recurrent = True
    def __init__(self,   
                 num_prop,
                 num_critic_obs,
                 num_hist,
                 num_actions,
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        
        super(ActorCriticRecurrent, self).__init__()

        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        
        self.actor = RnnNextSiamNormActor(num_prop=num_prop,
                                      actor_dims=[512,256,128],
                                      num_actions=num_actions,
                                      activation=activation,
                                      rnn_num_hidden=256,
                                      rnn_type='lstm',
                                      rnn_num_layers=1)
        
        # self.height_encode = nn.Sequential(nn.Linear(187,128),
        #                            nn.ELU(),
        #                            nn.Linear(128,64),
        #                            nn.ELU(),
        #                            nn.Linear(64,32))
        
        #self.memory_c = Memory(num_critic_obs-187+32, type='lstm', num_layers=1, hidden_size=256)
        self.memory_c = Memory(num_critic_obs, type='lstm', num_layers=1, hidden_size=256)

        print(f"Actor RNN: {self.actor.rnn}")
        print(f"Critic RNN: {self.memory_c}")
        
        self.critic = nn.Sequential(nn.ELU(),
                                   nn.Linear(256,512),
                                   nn.ELU(),
                                   nn.Linear(512,256),
                                   nn.ELU(),
                                   nn.Linear(256,128),
                                   nn.ELU(),
                                   nn.Linear(128,1))
        
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # self.optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        self.actor.rnn.reset(dones)
        self.memory_c.reset(dones)
        self.actor.est_rnn.reset(dones)
    
    def act(self, obs,masks=None, hidden_states=None):
        self.update_distribution(obs,masks,hidden_states)
        return self.distribution.sample()
    
    def update_distribution(self, obs, masks, hidden_states):
        mean = self.act_inference(obs, masks, hidden_states)
        self.distribution = Normal(mean, mean*0. + self.get_std())
    
    def act_inference(self, observations, masks, hidden_states):
        mean = self.actor(observations, masks, hidden_states)
        return mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # critic_observations_encode = torch.cat([critic_observations[...,187:],self.height_encode(critic_observations[...,:187])],dim=-1)
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        value = self.critic(input_c.squeeze(0))
        return value
        
    def get_hidden_states(self):
        return self.actor.rnn.hidden_states, self.memory_c.hidden_states, self.actor.est_rnn.hidden_states
    
    def subtask_loss(self,subtask_data):
        return self.actor.Loss(subtask_data)
    
    def update(self,subtask_data):

        self.optimizer.zero_grad()
        loss = self.subtask_loss(subtask_data)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(),1)
        self.optimizer.step()
        return loss.detach()

