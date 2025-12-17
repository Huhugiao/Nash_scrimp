"""
Residual Actor Network for Residual RL.

This network learns to output action corrections (residuals) on top of a frozen base policy.
The output represents delta adjustments: final_action = base_action + residual_action

Key design:
- Same observation processing as main network (RadarEncoder)
- Smaller architecture (3 hidden layers) for efficiency
- tanh output for bounded residuals [-1, 1]
"""

import torch
import torch.nn as nn
import numpy as np
from mlp.alg_parameters_mlp import NetParameters, ResidualRLParameters


class ResidualRadarEncoder(nn.Module):
    """Radar encoder for residual network (shared architecture with main)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NetParameters.RADAR_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, NetParameters.RADAR_EMBED_DIM),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)


class ResidualActorNetwork(nn.Module):
    """
    Residual Actor: outputs action corrections (delta_v, delta_omega).
    
    Input: Same as main actor (75-dim raw observation)
    Output: Residual action in [-1, 1] (will be added to base action)
    """
    def __init__(self):
        super().__init__()
        
        self.hidden_dim = ResidualRLParameters.RESIDUAL_HIDDEN_DIM
        self.num_layers = ResidualRLParameters.RESIDUAL_NUM_LAYERS
        
        # Radar encoder (separate from base model)
        self.radar_encoder = ResidualRadarEncoder()
        
        # Actor backbone (smaller than main network)
        self.backbone = self._build_mlp(
            NetParameters.ACTOR_VECTOR_LEN,  # 19 = 11 scalar + 8 radar embed
            self.hidden_dim,
            self.num_layers
        )
        
        # Output heads
        self.policy_mean = nn.Linear(self.hidden_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, actor_obs):
        """
        Forward pass for residual action.
        
        Args:
            actor_obs: [Batch, 75] raw observation
            
        Returns:
            mean: [Batch, 2] residual action mean
            log_std: [Batch, 2] log standard deviation
        """
        # Process observation: scalar + radar encoding
        actor_scalar = actor_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        actor_radar = actor_obs[:, NetParameters.ACTOR_SCALAR_LEN:]
        actor_radar_emb = self.radar_encoder(actor_radar)
        actor_in = torch.cat([actor_scalar, actor_radar_emb], dim=-1)  # [Batch, 19]
        
        # Forward through backbone
        features = self.backbone(actor_in)
        mean = self.policy_mean(features)
        log_std = self.log_std.expand_as(mean)
        
        return mean, log_std


class ResidualCriticNetwork(nn.Module):
    """
    Critic for residual RL (evaluates combined state value).
    Uses CTDE: sees both tracker and target observations.
    """
    def __init__(self):
        super().__init__()
        
        self.hidden_dim = ResidualRLParameters.RESIDUAL_HIDDEN_DIM
        self.num_layers = ResidualRLParameters.RESIDUAL_NUM_LAYERS
        
        # Radar encoder for critic (processes both tracker and target radar)
        self.radar_encoder = ResidualRadarEncoder()
        
        # Critic backbone
        self.backbone = self._build_mlp(
            NetParameters.CRITIC_VECTOR_LEN,  # 35 = tracker(19) + target(16)
            self.hidden_dim,
            self.num_layers
        )
        
        # Value head
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
        # Initialize
        self.apply(self._init_weights)
        
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, critic_obs):
        """
        Forward pass for value estimation.
        
        Args:
            critic_obs: [Batch, 147] = Tracker(75) + Target(72)
            
        Returns:
            value: [Batch, 1] estimated state value
        """
        # Process tracker part (0 to 75)
        tracker_end = NetParameters.ACTOR_RAW_LEN
        tracker_scalar = critic_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        tracker_radar = critic_obs[:, NetParameters.ACTOR_SCALAR_LEN:tracker_end]
        tracker_radar_emb = self.radar_encoder(tracker_radar)
        tracker_part = torch.cat([tracker_scalar, tracker_radar_emb], dim=-1)  # [19]
        
        # Process target part (75 to 147)
        target_start = tracker_end
        target_scalar = critic_obs[:, target_start:target_start+NetParameters.PRIVILEGED_SCALAR_LEN]
        target_radar = critic_obs[:, target_start+NetParameters.PRIVILEGED_SCALAR_LEN:]
        target_radar_emb = self.radar_encoder(target_radar)
        target_part = torch.cat([target_scalar, target_radar_emb], dim=-1)  # [16]
        
        # Combine
        critic_in = torch.cat([tracker_part, target_part], dim=-1)  # [35]
        
        # Forward
        features = self.backbone(critic_in)
        value = self.value_head(features)
        
        return value


class ResidualPolicyNetwork(nn.Module):
    """
    Combined network for residual RL training.
    Contains both actor (residual) and critic (value).
    """
    def __init__(self):
        super().__init__()
        self.actor = ResidualActorNetwork()
        self.critic = ResidualCriticNetwork()
        
    def forward(self, actor_obs, critic_obs):
        """
        Forward pass for both actor and critic.
        
        Returns:
            mean: residual action mean
            value: state value
            log_std: action log standard deviation
        """
        mean, log_std = self.actor(actor_obs)
        value = self.critic(critic_obs)
        return mean, value, log_std
    
    def get_action(self, actor_obs, deterministic=False):
        """Sample or get deterministic residual action."""
        mean, log_std = self.actor(actor_obs)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = torch.exp(log_std)
            noise = torch.randn_like(mean)
            pre_tanh = mean + std * noise
            return torch.tanh(pre_tanh), pre_tanh, mean, log_std
