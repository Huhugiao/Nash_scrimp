import torch
import torch.nn as nn
import numpy as np
from residual.alg_parameters_residual import NetParameters


class ResidualActorNetwork(nn.Module):
    """
    Residual Actor: Outputs bounded correction to base action
    Input: Radar only (64-dim)
    Output range: [-max_scale, +max_scale]
    """
    def __init__(self, 
                 input_dim=None,
                 action_dim=None,
                 hidden_dim=None,
                 num_layers=None,
                 max_scale=None):
        super().__init__()
        
        # Use defaults from parameters
        if input_dim is None:
            input_dim = NetParameters.RADAR_DIM  # 64维雷达
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
        
    def forward(self, radar):
        """
        Args:
            radar: (batch, 64) - normalized radar readings
        Returns:
            mean: (batch, action_dim) - bounded in [-max_scale, max_scale]
            log_std: (batch, action_dim)
        """
        features = self.feature_net(radar)
        raw_mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Bound mean output
        scaled_mean = torch.tanh(raw_mean) * self.max_scale
        
        # Clamp log_std to reasonable range
        log_std = torch.clamp(log_std, -20, 2)
        
        return scaled_mean, log_std


class ResidualCriticNetwork(nn.Module):
    """
    Value function for residual policy
    Input: Radar only (64-dim)
    """
    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 num_layers=None):
        super().__init__()
        
        if input_dim is None:
            input_dim = NetParameters.RADAR_DIM  # 64维雷达
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
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, radar):
        """
        Args:
            radar: (batch, 64) - normalized radar readings
        Returns:
            value: (batch, 1)
        """
        return self.network(radar)


class ResidualPolicyNetwork(nn.Module):
    """
    Simplified Residual Policy Module (Radar-Only):
    - Actor: Outputs bounded residual action from radar
    - Critic: Estimates value from radar
    - No Gate: Actor learns to output ~0 when safe via L2 penalty
    """
    def __init__(self):
        super().__init__()
        
        self.actor = ResidualActorNetwork()
        self.critic = ResidualCriticNetwork()
    
    @staticmethod
    def fuse_actions(base_action, residual_action):
        """
        Fuse base and residual actions.
        
        Args:
            base_action: (batch, action_dim) - frozen base policy output
            residual_action: (batch, action_dim) - residual correction
        
        Returns:
            fused_action: (batch, action_dim) - clamped to [-1, 1]
        """
        fused = base_action + residual_action
        return torch.clamp(fused, -1.0, 1.0)
