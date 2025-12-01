import torch
import torch.nn as nn
import numpy as np
from mlp.alg_parameters_mlp import NetParameters

class ProtectingNetMLP(nn.Module):
    def __init__(self):
        super(ProtectingNetMLP, self).__init__()
        
        self.hidden_dim = NetParameters.HIDDEN_DIM
        
        # --- Actor Network ---
        # Input: 27 dims (Direct Concatenation of Scalar + Radar)
        # Structure: 3 Hidden Layers
        self.actor_backbone = nn.Sequential(
            nn.Linear(NetParameters.ACTOR_VECTOR_LEN, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        
        self.policy_mean = nn.Linear(self.hidden_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))
        
        # --- Critic Network (CTDE) ---
        # Input: 51 dims (Tracker Obs + Target Obs)
        # Structure: 3 Hidden Layers
        self.critic_backbone = nn.Sequential(
            nn.Linear(NetParameters.CRITIC_VECTOR_LEN, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
        # --- Q-Network ---
        # Input: Critic Obs (51) + Action (2) = 53 dims
        # Structure: 3 Hidden Layers
        self.q_backbone = nn.Sequential(
            nn.Linear(NetParameters.CRITIC_VECTOR_LEN + NetParameters.ACTION_DIM, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, actor_obs, critic_obs):
        # --- Actor Forward ---
        # actor_obs: [Batch, 27]
        a_out = self.actor_backbone(actor_obs)
        mean = self.policy_mean(a_out)
        
        # Expand log_std to match batch size
        log_std = self.log_std.expand_as(mean)
        
        # --- Critic Forward ---
        # critic_obs: [Batch, 51]
        c_out = self.critic_backbone(critic_obs)
        value = self.value_head(c_out)
        
        return mean, value, log_std

    def forward_q(self, critic_obs, action):
        # Flatten inputs to handle [Batch, Seq, Dim] or [Batch, Dim]
        state_flat = critic_obs.reshape(-1, critic_obs.shape[-1])
        action_flat = action.reshape(-1, action.shape[-1])
        
        # Concatenate State + Action
        q_in = torch.cat([state_flat, action_flat], dim=-1)
        
        # Forward
        q_val = self.q_backbone(q_in)
        
        # Reshape back to original batch structure
        return q_val.view(*critic_obs.shape[:-1], 1)