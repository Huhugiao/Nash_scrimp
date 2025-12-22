"""
Residual Actor-Critic Networks

残差网络用于学习 CBF base policy 的微调动作。
关键设计：
- 可配置的网络大小（比主网络更小）
- 输出残差动作时确保不超过物理限制
- tanh 限制残差范围，通过 RESIDUAL_SCALE 缩放
"""

import math
import torch
import torch.nn as nn
import numpy as np
from residual.alg_parameters_residual import ResidualNetParameters


class ResidualRadarEncoder(nn.Module):
    """雷达编码器（与主网络架构一致但独立参数）"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ResidualNetParameters.RADAR_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, ResidualNetParameters.RADAR_EMBED_DIM),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)


class ResidualActorNetwork(nn.Module):
    """
    残差 Actor 网络
    
    输出残差动作 δa ∈ [-scale, scale]
    最终动作 a = clip(a_base + δa, -1, 1)
    
    Args:
        hidden_dim: 隐藏层维度
        num_layers: 隐藏层数量
        residual_scale: 残差缩放系数 α
    """
    
    def __init__(self, 
                 hidden_dim=None, 
                 num_layers=None, 
                 residual_scale=None):
        super().__init__()
        
        # 从参数配置读取，允许覆盖
        self.hidden_dim = hidden_dim or ResidualNetParameters.HIDDEN_DIM
        self.num_layers = num_layers or ResidualNetParameters.NUM_LAYERS
        self.residual_scale = residual_scale or ResidualNetParameters.RESIDUAL_SCALE
        
        # 雷达编码器
        self.radar_encoder = ResidualRadarEncoder()
        
        # 计算输入维度：scalar + radar_embed
        input_dim = ResidualNetParameters.ACTOR_SCALAR_LEN + ResidualNetParameters.RADAR_EMBED_DIM
        
        # MLP backbone
        self.backbone = self._build_mlp(input_dim, self.hidden_dim, self.num_layers)
        
        # 策略输出头
        self.policy_mean = nn.Linear(self.hidden_dim, ResidualNetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(
            torch.ones(ResidualNetParameters.ACTION_DIM) * -1.0  # 初始 std ≈ 0.37
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 策略输出层使用更小的初始化（输出接近零）
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.constant_(self.policy_mean.bias, 0.0)
        
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, actor_obs):
        """
        前向传播
        
        Args:
            actor_obs: [Batch, 75] 原始观测
            
        Returns:
            mean: [Batch, 2] 残差动作均值（已 tanh 后缩放到 [-scale, scale]）
            log_std: [Batch, 2] 对数标准差
        """
        # 分离 scalar 和 radar
        actor_scalar = actor_obs[:, :ResidualNetParameters.ACTOR_SCALAR_LEN]
        actor_radar = actor_obs[:, ResidualNetParameters.ACTOR_SCALAR_LEN:]
        
        # 编码雷达
        radar_emb = self.radar_encoder(actor_radar)
        
        # 拼接输入
        actor_in = torch.cat([actor_scalar, radar_emb], dim=-1)
        
        # Backbone
        features = self.backbone(actor_in)
        
        # 输出：tanh 限制到 [-1, 1]，再乘以 scale
        raw_mean = self.policy_mean(features)
        mean = torch.tanh(raw_mean) * self.residual_scale
        
        log_std = self.log_std.expand_as(mean)
        
        return mean, log_std
    
    def forward_raw(self, actor_obs):
        """
        前向传播，返回 pre-tanh 的原始均值（用于正确采样）
        
        Args:
            actor_obs: [Batch, 75] 原始观测
            
        Returns:
            raw_mean: [Batch, 2] pre-tanh 均值
            log_std: [Batch, 2] 对数标准差
        """
        actor_scalar = actor_obs[:, :ResidualNetParameters.ACTOR_SCALAR_LEN]
        actor_radar = actor_obs[:, ResidualNetParameters.ACTOR_SCALAR_LEN:]
        
        radar_emb = self.radar_encoder(actor_radar)
        actor_in = torch.cat([actor_scalar, radar_emb], dim=-1)
        features = self.backbone(actor_in)
        raw_mean = self.policy_mean(features)
        log_std = self.log_std.expand_as(raw_mean)
        
        return raw_mean, log_std
    
    def get_residual_action(self, actor_obs, deterministic=False):
        """获取残差动作（不含 base action）"""
        raw_mean, log_std = self.forward_raw(actor_obs)
        
        if deterministic:
            return torch.tanh(raw_mean) * self.residual_scale
        else:
            std = torch.exp(log_std)
            noise = torch.randn_like(raw_mean)
            pre_tanh = raw_mean + std * noise
            action = torch.tanh(pre_tanh) * self.residual_scale
            # 裁剪到残差范围
            action = torch.clamp(action, -self.residual_scale, self.residual_scale)
            return action


class ResidualCriticNetwork(nn.Module):
    """
    残差 Critic 网络
    
    评估 combined action (base + residual) 的状态价值
    使用 CTDE：训练时看到完整状态
    """
    
    def __init__(self, hidden_dim=None, num_layers=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim or ResidualNetParameters.HIDDEN_DIM
        self.num_layers = num_layers or ResidualNetParameters.NUM_LAYERS
        
        # 雷达编码器
        self.radar_encoder = ResidualRadarEncoder()
        
        # 输入维度计算
        # Tracker: scalar(11) + radar_embed(8) = 19
        # Target: scalar(8) + radar_embed(8) = 16
        # Total: 35
        tracker_dim = ResidualNetParameters.ACTOR_SCALAR_LEN + ResidualNetParameters.RADAR_EMBED_DIM
        target_scalar_len = 8  # Privileged scalar
        target_dim = target_scalar_len + ResidualNetParameters.RADAR_EMBED_DIM
        input_dim = tracker_dim + target_dim
        
        # Backbone
        self.backbone = self._build_mlp(input_dim, self.hidden_dim, self.num_layers)
        
        # Value head
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, critic_obs):
        """
        前向传播
        
        Args:
            critic_obs: [Batch, 147] = Tracker(75) + Target(72)
            
        Returns:
            value: [Batch, 1]
        """
        tracker_end = ResidualNetParameters.ACTOR_RAW_LEN  # 75
        
        # Tracker 部分
        tracker_scalar = critic_obs[:, :ResidualNetParameters.ACTOR_SCALAR_LEN]
        tracker_radar = critic_obs[:, ResidualNetParameters.ACTOR_SCALAR_LEN:tracker_end]
        tracker_radar_emb = self.radar_encoder(tracker_radar)
        tracker_part = torch.cat([tracker_scalar, tracker_radar_emb], dim=-1)
        
        # Target 部分
        target_start = tracker_end
        target_scalar_len = 8
        target_scalar = critic_obs[:, target_start:target_start + target_scalar_len]
        target_radar = critic_obs[:, target_start + target_scalar_len:]
        target_radar_emb = self.radar_encoder(target_radar)
        target_part = torch.cat([target_scalar, target_radar_emb], dim=-1)
        
        # 合并
        critic_in = torch.cat([tracker_part, target_part], dim=-1)
        
        # 前向
        features = self.backbone(critic_in)
        value = self.value_head(features)
        
        return value


class ResidualPolicyNetwork(nn.Module):
    """
    完整的残差策略网络（Actor + Critic）
    """
    
    def __init__(self, hidden_dim=None, num_layers=None, residual_scale=None):
        super().__init__()
        self.actor = ResidualActorNetwork(hidden_dim, num_layers, residual_scale)
        self.critic = ResidualCriticNetwork(hidden_dim, num_layers)
        self.residual_scale = self.actor.residual_scale
        
    def forward(self, actor_obs, critic_obs):
        """
        前向传播
        
        Returns:
            mean: 残差动作均值
            value: 状态价值
            log_std: 对数标准差
        """
        mean, log_std = self.actor(actor_obs)
        value = self.critic(critic_obs)
        return mean, value, log_std
    
    def get_residual_action(self, actor_obs, deterministic=False):
        """
        采样或确定性获取残差动作
        
        Returns:
            action: 残差动作 (tanh * scale 后)
            pre_squash: pre-tanh 值（用于 log_prob 计算）
            raw_mean: pre-tanh 均值
            log_std: 对数标准差
        """
        raw_mean, log_std = self.actor.forward_raw(actor_obs)
        
        if deterministic:
            action = torch.tanh(raw_mean) * self.residual_scale
            return action, raw_mean, raw_mean, log_std
        else:
            std = torch.exp(log_std)
            noise = torch.randn_like(raw_mean)
            pre_squash = raw_mean + std * noise
            action = torch.tanh(pre_squash) * self.residual_scale
            action = torch.clamp(action, -self.residual_scale, self.residual_scale)
            return action, pre_squash, raw_mean, log_std
