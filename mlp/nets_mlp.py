import torch
import torch.nn as nn
import numpy as np
from mlp.alg_parameters_mlp import NetParameters

class ProtectingNetMLP(nn.Module):
    def __init__(self):
        super(ProtectingNetMLP, self).__init__()
        
        self.hidden_dim = NetParameters.HIDDEN_DIM
        
        # --- Actor Network ---
        # Input: 27 dims
        # Scalar features: indices 0-10 (11 dims)
        # Radar features: indices 11-26 (16 dims)
        
        self.actor_scalar_enc = nn.Sequential(
            nn.Linear(11, self.hidden_dim // 2),
            nn.Tanh()
        )
        self.actor_radar_enc = nn.Sequential(
            nn.Linear(16, self.hidden_dim // 2),
            nn.Tanh()
        )
        
        # Fused backbone
        self.actor_backbone = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        
        self.policy_mean = nn.Linear(self.hidden_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))
        
        # --- Critic Network ---
        # Input: 27 dims + Context
        # Scalar features: indices 0-10 + Context (indices 27+)
        # Radar features: indices 11-26
        
        critic_scalar_dim = 11 + NetParameters.CONTEXT_LEN
        self.critic_scalar_enc = nn.Sequential(
            nn.Linear(critic_scalar_dim, self.hidden_dim // 2),
            nn.Tanh()
        )
        self.critic_radar_enc = nn.Sequential(
            nn.Linear(16, self.hidden_dim // 2),
            nn.Tanh()
        )
        
        self.critic_backbone = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, actor_obs, critic_obs):
        # --- Actor Forward ---
        a_scalar = actor_obs[..., :11]
        a_radar = actor_obs[..., 11:27]
        
        a_s_emb = self.actor_scalar_enc(a_scalar)
        a_r_emb = self.actor_radar_enc(a_radar)
        
        # Fuse
        a_feat = torch.cat([a_s_emb, a_r_emb], dim=-1)
        a_out = self.actor_backbone(a_feat)
        
        mean = self.policy_mean(a_out)
        
        # --- Critic Forward ---
        # Critic obs: [Tracker(27), Context(N)]
        c_scalar = torch.cat([critic_obs[..., :11], critic_obs[..., 27:]], dim=-1)
        c_radar = critic_obs[..., 11:27]
        
        c_s_emb = self.critic_scalar_enc(c_scalar)
        c_r_emb = self.critic_radar_enc(c_radar)
        
        c_feat = torch.cat([c_s_emb, c_r_emb], dim=-1)
        c_out = self.critic_backbone(c_feat)
        
        value = self.value_head(c_out)
        
        # Expand log_std to match batch size if needed, though usually broadcasted
        log_std = self.log_std.expand_as(mean)
        
        return mean, value, log_std
