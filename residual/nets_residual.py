import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual.alg_parameters_residual import NetParameters


class ResidualActorNetwork(nn.Module):
    """
    Residual Actor: Outputs bounded correction to base action
    Input: Radar (64-dim) + Base Action (2-dim) + Velocity (2-dim) = 68-dim
    Output: residual_mean [-max_scale, +max_scale], log_std
    """
    def __init__(self, 
                 input_dim=None,
                 action_dim=None,
                 hidden_dim=None,
                 num_layers=None,
                 max_scale=None):
        super().__init__()
        
        if input_dim is None:
            input_dim = NetParameters.RESIDUAL_INPUT_DIM  # 68
        if action_dim is None:
            action_dim = NetParameters.ACTION_DIM
        if hidden_dim is None:
            hidden_dim = NetParameters.RESIDUAL_HIDDEN_DIM
        if num_layers is None:
            num_layers = NetParameters.RESIDUAL_NUM_LAYERS
        if max_scale is None:
            max_scale = NetParameters.RESIDUAL_MAX_SCALE
            
        self.max_scale = max_scale
        
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize with small weights (start with small residuals)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        
    def forward(self, radar, base_action, velocity):
        # Concatenate inputs
        x = torch.cat([radar, base_action, velocity], dim=-1)
        h = self.feature_net(x)

        # IMPORTANT: return raw_mean (unbounded). Squashing happens at sampling/execution time.
        raw_mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, NetParameters.LOG_STD_MIN, NetParameters.LOG_STD_MAX)
        return raw_mean, log_std


class ResidualCriticNetwork(nn.Module):
    """
    Value function for residual policy
    Input: Radar (64) + Base Action (2) + Velocity (2) = 68-dim
    """
    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 num_layers=None):
        super().__init__()
        
        if input_dim is None:
            input_dim = NetParameters.RESIDUAL_INPUT_DIM  # 68 (same as actor)
        if hidden_dim is None:
            hidden_dim = NetParameters.RESIDUAL_HIDDEN_DIM
        if num_layers is None:
            num_layers = NetParameters.RESIDUAL_NUM_LAYERS
        
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
            
        self.feature_net = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, radar, base_action, velocity):
        x = torch.cat([radar, base_action, velocity], dim=-1)
        features = self.feature_net(x)
        return self.value_head(features)


class ResidualPolicyNetwork(nn.Module):
    """
    Residual Policy Module:
    - Actor: Outputs bounded residual action from radar + base_action + velocity
    - Critic: Estimates value from radar + base_action + velocity
    - Fusion: Simple addition (base + residual, clamped)
    """
    def __init__(self):
        super().__init__()
        self.actor = ResidualActorNetwork()
        self.critic = ResidualCriticNetwork()
    
    @staticmethod
    def fuse_actions(base_action, residual_action):
        """
        Simple additive fusion of base and residual actions.
        
        Args:
            base_action: (batch, action_dim) - frozen base policy output
            residual_action: (batch, action_dim) - residual correction
        
        Returns:
            fused_action: (batch, action_dim) - clamped to [-1, 1]
        """
        return torch.clamp(base_action + residual_action, -1.0, 1.0)
