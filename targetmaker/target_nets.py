import torch
import torch.nn as nn
import numpy as np
from targetmaker.target_alg_parameters import TargetNetParameters

class RadarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TargetNetParameters.RADAR_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, TargetNetParameters.RADAR_EMBED_DIM),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

class TargetPPOActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.radar_encoder = RadarEncoder()
        
        self.hidden_dim = TargetNetParameters.HIDDEN_DIM
        self.num_layers = TargetNetParameters.NUM_HIDDEN_LAYERS
        
        # --- Actor (Policy) ---
        # Input: TargetObs
        self.actor_net = self._build_mlp(TargetNetParameters.TARGET_VECTOR_LEN, self.hidden_dim, self.num_layers)
        self.mean_head = nn.Linear(self.hidden_dim, TargetNetParameters.ACTION_DIM)
        self.log_std_param = nn.Parameter(torch.zeros(1, TargetNetParameters.ACTION_DIM))
        
        # --- Critic (Value) ---
        # Input: Full State (Target+Tracker)
        self.critic_net = self._build_mlp(TargetNetParameters.STATE_VECTOR_LEN, self.hidden_dim, self.num_layers)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
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

    def _encode(self, obs, is_full_state=False):
        # Helper to encode Scalar+Radar
        # obs: [Batch, LEN]
        if not is_full_state:
            # Target Only
            scalar = obs[:, :TargetNetParameters.TARGET_SCALAR_LEN]
            radar = obs[:, TargetNetParameters.TARGET_SCALAR_LEN:]
            radar_emb = self.radar_encoder(radar)
            return torch.cat([scalar, radar_emb], dim=-1)
        else:
            # Full State
            # Target Part
            c_tar_scalar = obs[:, :TargetNetParameters.TARGET_SCALAR_LEN]
            c_tar_radar = obs[:, TargetNetParameters.TARGET_SCALAR_LEN:TargetNetParameters.TARGET_RAW_LEN]
            c_tar_emb = self.radar_encoder(c_tar_radar)
            c_tar_vec = torch.cat([c_tar_scalar, c_tar_emb], dim=-1)
            
            # Tracker Part
            T_START = TargetNetParameters.TARGET_RAW_LEN
            c_trk_scalar = obs[:, T_START:T_START+TargetNetParameters.TRACKER_SCALAR_LEN]
            c_trk_radar = obs[:, T_START+TargetNetParameters.TRACKER_SCALAR_LEN:]
            c_trk_emb = self.radar_encoder(c_trk_radar)
            c_trk_vec = torch.cat([c_trk_scalar, c_trk_emb], dim=-1)
            
            return torch.cat([c_tar_vec, c_trk_vec], dim=-1)

    def forward(self, actor_obs, critic_obs=None):
        # actor_obs: Target View
        # critic_obs: Full View (Optional, if we want value)
        
        # Policy
        actor_vec = self._encode(actor_obs, is_full_state=False)
        actor_feat = self.actor_net(actor_vec)
        mean = self.mean_head(actor_feat)
        log_std = self.log_std_param.expand_as(mean)
        std = torch.exp(log_std)
        
        value = None
        if critic_obs is not None:
            critic_vec = self._encode(critic_obs, is_full_state=True)
            critic_feat = self.critic_net(critic_vec)
            value = self.value_head(critic_feat)
            
        return mean, std, value
