import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self,
                 observation_dim,
                 observation_horizon,
                 device,
                 reward_coef=0.1,
                 reward_lerp=0.3,
                 shape=[1024, 512],
                 style_reward_function="quad_mapping",
                 **kwargs,
                 ):
        if kwargs:
            print("Discriminator.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(Discriminator, self).__init__()
     
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim = observation_dim * observation_horizon
        self.device = device
        self.reward_coef = reward_coef
        self.reward_lerp = reward_lerp
        self.style_reward_function = style_reward_function
        self.shape = shape

        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            # discriminator_layers.append(nn.LayerNorm(hidden_dim))
            # discriminator_layers.append(nn.ELU())
            curr_in_dim = hidden_dim
        discriminator_layers.append(nn.Linear(self.shape[-1], 1))
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.architecture.train()

    def forward(self, x):
        return self.architecture(x)

    def compute_grad_pen(self,   
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.architecture(expert_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen
    
    def compute_wgan_pen(self,
                         expert_state,
                         expert_next_state,
                         policy_state, 
                         policy_next_state,
                         lambda_=10):
        # get lerped data
        with torch.no_grad():
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
            policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
            lerp_factor = torch.rand_like(expert_data,device=expert_data.device)
            lerped_data = lerp_factor*expert_data + (1-lerp_factor)*policy_data
        lerped_data.requires_grad = True
        
        disc = self.architecture(lerped_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=lerped_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        
        grad_norm = grad.norm(2, dim=1)
        # grad_pen = lambda_ * torch.max(grad_norm - 1.0, torch.zeros_like(grad_norm)).pow(2).mean()
        grad_pen = lambda_ * (grad_norm - 1.0).pow(2).mean()
        return grad_pen
    
    def compute_wgan_slack_pen(self,
                         expert_state,
                         expert_next_state,
                         policy_state, 
                         policy_next_state,
                         lambda_=10):
        # get lerped data
        with torch.no_grad():
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
            policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
            lerp_factor = torch.rand_like(expert_data,device=expert_data.device)
            lerped_data = lerp_factor*expert_data + (1-lerp_factor)*policy_data
        lerped_data.requires_grad = True
        
        disc = self.architecture(lerped_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=lerped_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        grad_norm = grad.norm(2, dim=1)
        grad_pen = lambda_ * torch.max(grad_norm - 1.0, torch.zeros_like(grad_norm)).pow(2).mean()
        # grad_pen = lambda_ * (grad_norm - 1.0).pow(2).mean()
        return grad_pen
    
    def compute_pair_pen(self,
                         expert_state,
                         expert_next_state,
                         policy_state, 
                         policy_next_state,
                         lambda_=10):
        
        with torch.no_grad():
            expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
            policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
        
        expert_data.requires_grad = True
        policy_data.requires_grad = True
        # exp_data 
        disc_exp = self.architecture(expert_data)
        ones_exp = torch.ones(disc_exp.size(), device=disc_exp.device)
        grad_exp = autograd.grad(
            outputs=disc_exp, inputs=expert_data,
            grad_outputs=ones_exp, create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        exp_grad_norm = grad_exp.norm(2, dim=1)
        
        # policy data
        disc_pol = self.architecture(policy_data)
        ones_pol = torch.ones(disc_pol.size(), device=disc_pol.device)
        grad_pol = autograd.grad(
            outputs=disc_pol, inputs=policy_data,
            grad_outputs=ones_pol, create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        pol_grad_norm = grad_pol.norm(2, dim=1)
        
        half_lambda = lambda_/2.0
        
        grad_pen = half_lambda *(pol_grad_norm.pow(2).mean() + exp_grad_norm.pow(2).mean())
        
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, dt, state_normalizer=None, style_reward_normalizer=None):
        with torch.no_grad():
            self.eval()
            state_normalizer.eval() # 避免少量数据影响state的标准化，此标准化只在ppo中发生
            if state_normalizer is not None:
                state = state_normalizer(state)
                next_state = state_normalizer(next_state)
            state_normalizer.train()
            
            state_input = torch.cat([state,next_state],dim=-1)
            d = self.architecture(state_input)
            
            if self.style_reward_function == "quad_mapping":
                style_reward = self.reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            elif self.style_reward_function == "log_mapping":
                style_reward = -torch.log(torch.maximum(1 - 1 / (1 + torch.exp(-d)), torch.tensor(0.0001, device=self.device)))
            elif self.style_reward_function == "wasserstein_mapping":
                if style_reward_normalizer is not None:
                    style_reward = style_reward_normalizer(d.clone()) # 标准化的同时进行更新
                    # style_reward_normalizer.update(d)
                else:
                    style_reward = d
                style_reward = self.reward_coef * style_reward
            elif self.style_reward_function == "wasserstein_tanh_mapping":
                #eta = 0.5
                style_reward = self.reward_coef*torch.exp(d)
            else:
                raise ValueError("Unexpected style reward mapping specified")
            style_reward = style_reward * dt

            reward = (1.0 - self.reward_lerp) * style_reward + self.reward_lerp * task_reward.unsqueeze(-1)
            self.train()
        return reward.squeeze(), style_reward.squeeze()