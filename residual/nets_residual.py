import torch
import torch.nn as nn
import numpy as np
from residual.alg_parameters_residual import NetParameters


class ResidualActorNetwork(nn.Module):
    """
    Gated Residual Actor: Outputs bounded correction to base action + safety gate
    Input: Radar (64-dim) + Base Action (2-dim) = 66-dim
    Output: 
        - residual_mean: [-max_scale, +max_scale]
        - log_std: for stochastic policy
        - gate: [0, 1] where 1=safe (use base), 0=danger (apply residual)
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
            input_dim = NetParameters.RESIDUAL_INPUT_DIM  # 66 = radar(64) + base_action(2)
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
        self.gate_head = nn.Linear(hidden_dim, 1)  # Safety gate output
        
        # Initialize with small weights (start with small residuals)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        # Initialize gate to output ~1 (safe) initially
        nn.init.constant_(self.gate_head.weight, 0.0)
        nn.init.constant_(self.gate_head.bias, 2.0)  # sigmoid(2) ≈ 0.88, starts safe
        
    def forward(self, radar, base_action, velocity):
        """
        Args:
            radar: (batch, 64) - normalized radar readings
            base_action: (batch, 2) - base policy action
            velocity: (batch, 2) - [linear_vel, angular_vel] normalized
        Returns:
            mean: (batch, action_dim) - bounded in [-max_scale, max_scale]
            log_std: (batch, action_dim)
            gate: (batch, 1) - safety gate in [0, 1]
        """
        # Concatenate inputs: radar + base_action + velocity = 68 dim
        combined = torch.cat([radar, base_action, velocity], dim=-1)
        features = self.feature_net(combined)
        
        raw_mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        gate = torch.sigmoid(self.gate_head(features))  # [0, 1]
        
        # Bound mean output
        scaled_mean = torch.tanh(raw_mean) * self.max_scale
        
        # Clamp log_std to reasonable range
        log_std = torch.clamp(log_std, -20, 2)
        
        return scaled_mean, log_std, gate


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
    Gated Residual Policy Module:
    - Actor: Outputs bounded residual action + safety gate from radar + base_action
    - Critic: Estimates value from radar
    - Gate: Learned safety gate (1=safe, use base; 0=danger, apply residual)
    """
    def __init__(self):
        super().__init__()
        
        self.actor = ResidualActorNetwork()
        self.critic = ResidualCriticNetwork()
    
    @staticmethod
    def fuse_actions(base_action, residual_action, gate):
        """
        Gated fusion of base and residual actions.
        
        Args:
            base_action: (batch, action_dim) - frozen base policy output
            residual_action: (batch, action_dim) - residual correction
            gate: (batch, 1) - safety gate in [0, 1]
                  gate=1: fully trust base (safe)
                  gate=0: apply full residual correction (danger)
        
        Returns:
            fused_action: (batch, action_dim) - clamped to [-1, 1]
        """
        # corrected = base + residual, clamped
        corrected = torch.clamp(base_action + residual_action, -1.0, 1.0)
        # Interpolate: gate * base + (1 - gate) * corrected
        fused = gate * base_action + (1 - gate) * corrected
        return torch.clamp(fused, -1.0, 1.0)
