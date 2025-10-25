from humanoidGym.algo.ppo.normalizer import EmpiricalNormalization
from humanoidGym.algo.ppo.utils import smooth_decay, smooth_decay_se
from .modules import MixMlpSlimVaeLongShortRegressionActor, MixMlpVaeLongShortRegressionActor, MixMlpVaeRegressionActor, MixSlimMlpVQVAERegressionActor, MixmlpVQVAERegressionActor, MixmlpVqvaeLongShortRegressionActor, MlpBVAEDeltaLatentHistRegressionActor, MlpBVAEDeltaRegressionActor, MlpBVAENoPhaseRegressionActor, MlpBVAERegressionActor, MlpBVAETcnContactNoPhaseRegressionActor, MlpBVAETcnContactRegressionActor, MlpBVAETcnRegressionActor, MlpBVAETransRegressionActor, MlpBarlowTwinsLongCnnRegressionActor, MlpBaselineActor, MlpBaselineVQVAEActor, MlpHistoryHeightNoPhaseActor, MlpRnnBVAEActor,MlpSimpleLongShortRegressionActor, MlpSimpleMlpRegressionActor, MlpSimpleRegressionActor, MlpSimpleRnnPhaseShiftRegressionActor, MlpSimpleShortLongRegressionActor, MlpVAERegressionActor, MlpVQVAEActor, MlpVQVAECnnActor, MlpVQVAELongHistActor, MlpVQVAELongShortRegressionActor, MlpVQVAEMixedActor, MlpVQVAERegressionActor, MlpVQVAERnnEncodeActor, MlpVQVAEShortHistActor, MlpVaeLongShortBothGradRegressionActor, MlpVaeLongShortRegressionActor, MlpVaeRegressionActor, MlpVqvaeLongEstLayerNormFallPredictRegressionActor, MlpVqvaeLongEstLayerNormRegressionActor, MlpVqvaeLongEstRegressionActor, MlpVqvaeLongShortBothGradRegressionActor, MlpVqvaeVelHeightRegressionActor, get_activation,mlp_factory,MlpRnnFullBVAEActor,MlpVQVAERnnActor,MlpRnnBarlowTwinActor,MlpBarlowTwinsRegressionActor,MlpBarlowTwinsRnnRegressionActor,MixedMlpBarlowTwinsRegressionActor,MlpBarlowTwinsCnnRegressionActor, MlpBarlowTwinsCnnRegressionShortHistActor,MlpBarlowTwinsCnnRegressionCurrentActor,MlpBarlowTwinsCnnRegressionShortHistActorNophase,MlpBarlowTwinsCnnRegressionNoPhaseActor,MlpTransRegressionActor, MlpSimSiamActor, MlpBarlowTwinsCnnDeltaRegressionActor,MlpBarlowTwinsCnnSingleActor,MlpBarlowTwinsCnnSingleNoPhaseActor,MlpBarlowTwinsCnnRegressionDirectPastActor,MlpBarlowTwinsCnnRegressionDirectPastNoPhaseActor,MlpSimSiamSingleStepActor,MlpSimSiamSingleStepHeightActor,MlpSimSiamSingleStepNoPhaseActor,MlpBaselineBarlowRegressionActor,MlpBaselineVAEActor,MlpBarlowTwinsNewCnnRegressionNoPhaseActor,MlpBarlowTwinsNewCnnRegressionActor,MlpBaselineTransActor,MlpBaselineTerrianGuideActor,MlpSimpleRnnRegressionActor,MixSlimMlpVaeRegressionActor
from torch.distributions import Normal
import torch.nn as nn
import torch
import torch.optim as optim


class InferenceActor(nn.Module):
    def __init__(self,actor_module,norm_module):
        super().__init__()
        self.actor_module = actor_module
        self.norm_module = norm_module
    def forward(self,x):
        x_norm = self.norm_module(x)
        y = self.actor_module(x_norm)
        return y
class InferenceActorWMP(nn.Module):
    def __init__(self,actor_module,norm_module):
        super().__init__()
        self.actor_module = actor_module
        self.norm_module = norm_module
    def forward(self,x,h,z):
        x_norm = self.norm_module(x)
        y = self.actor_module.act_inference(x_norm,h,z)
        return y

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  
                 num_prop,
                 num_critic_obs,
                 num_hist,
                 num_actions,
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super(ActorCritic, self).__init__()

        self.kwargs = kwargs

        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_critic_obs = num_critic_obs
        
        self.actor_teacher_backbone = MlpVqvaeLongEstLayerNormFallPredictRegressionActor(num_prop=num_prop,#remove linear vel
                                num_hist=num_hist,
                                num_actions=num_actions,
                                actor_dims=[512,256,128],
                                activation=activation,
                                latent_dim=16)

        # Value function
        critic_layers = mlp_factory(activation,self.num_critic_obs,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # self.optimizer = optim.Adam(self.actor_teacher_backbone.parameters(), lr=1e-3)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def reset(self, dones=None):
        pass

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

    def update_distribution(self, obs):
        mean = self.act_inference(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self,obs_hist, **kwargs):
        mean = self.actor_teacher_backbone(obs_hist)
        return mean
        
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def subtask_loss(self,obs_hist,critic_obs_hist):

        return self.actor_teacher_backbone.VaeLoss(obs_hist,critic_obs_hist)
    
    def update(self,obs_hist,critic_obs_hist):

        self.optimizer.zero_grad()
        loss = self.subtask_loss(obs_hist,critic_obs_hist)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(),1)
        self.optimizer.step()
        return loss.detach()
    

    
class ActorCriticWMP(nn.Module):
    is_recurrent = False

    def __init__(self,
                 num_critic_obs,
                 num_actions,
                 encoder_hidden_dims=[256, 128],
                 wm_encoder_hidden_dims = [64, 32],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 latent_dim = 32,
                 history_dim = 42*5,
                 wm_feature_dim = 1536,
                 wm_latent_dim=16,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCriticWMP, self).__init__()

        activation = get_activation(activation)

        self.latent_dim = latent_dim

 

        mlp_input_dim_a = latent_dim + 3 + wm_latent_dim #latent vector + command + wm_latent
        mlp_input_dim_c = num_critic_obs + wm_latent_dim

        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)

        # World Model Feature Encoder
        wm_encoder_layers = []
        wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        wm_encoder_layers.append(activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                wm_encoder_layers.append(activation)
        self.wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

        # Critic World Model Feature Encoder
        critic_wm_encoder_layers = []
        critic_wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        critic_wm_encoder_layers.append(activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                critic_wm_encoder_layers.append(activation)
        self.critic_wm_feature_encoder = nn.Sequential(*critic_wm_encoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)

        self.critic = nn.Sequential(*critic_layers)



        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, history, wm_feature, **kwargs):
        latent_vector = self.history_encoder(history)
        command = observations[:, :3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        self.update_distribution(concat_observations)
        return self.distribution.sample()

    def get_latent_vector(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:,-3:]
        return linear_vel

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, wm_feature):
        latent_vector = self.history_encoder(history)
        command = observations[:, :3]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        actions_mean = self.actor(concat_observations)
        return actions_mean

    def evaluate(self, critic_observations, wm_feature,  **kwargs):
        wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)
        concat_observations = torch.concat((critic_observations, wm_latent_vector),
                                           dim=-1)


        value = self.critic(concat_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None