import torch
import torch.nn as nn
import numpy as np
from star_hrl.alg_parameters_star import NetParameters, TrainingParameters

class RadarEncoder(nn.Module):
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

class StarNet(nn.Module):
    def __init__(self, device='cpu'):
        super(StarNet, self).__init__()
        self.device = device
        
        hidden_dim = NetParameters.HIDDEN_DIM
        num_layers = NetParameters.NUM_HIDDEN_LAYERS
        feature_dim = NetParameters.FEATURE_DIM
        
        self.radar_encoder = RadarEncoder()
        
        self.mlp_public = self._build_mlp(
            NetParameters.ACTOR_VECTOR_LEN,
            hidden_dim,
            feature_dim,
            num_layers
        )
        
        self.mlp_track = self._build_mlp(
            NetParameters.ACTOR_VECTOR_LEN,
            hidden_dim,
            feature_dim,
            num_layers
        )
        
        self.mlp_safe = self._build_mlp(
            NetParameters.ACTOR_VECTOR_LEN,
            hidden_dim,
            feature_dim,
            num_layers
        )
        
        self.track_action_head = nn.Linear(feature_dim * 2, NetParameters.ACTION_DIM)
        self.safe_action_head = nn.Linear(feature_dim * 2, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))
        
        self.critic_backbone = self._build_mlp(
            NetParameters.CRITIC_VECTOR_LEN,
            hidden_dim,
            hidden_dim,
            num_layers
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.output_dim = feature_dim * 2
        
    def _build_mlp(self, input_dim, hidden_dim, output_dim, num_layers):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def _encode_actor_obs(self, actor_obs):
        actor_scalar = actor_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        actor_radar = actor_obs[:, NetParameters.ACTOR_SCALAR_LEN:]
        actor_radar_emb = self.radar_encoder(actor_radar)
        return torch.cat([actor_scalar, actor_radar_emb], dim=-1)
    
    def _encode_critic_obs(self, critic_obs):
        tracker_end = NetParameters.ACTOR_RAW_LEN
        tracker_scalar = critic_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        tracker_radar = critic_obs[:, NetParameters.ACTOR_SCALAR_LEN:tracker_end]
        tracker_radar_emb = self.radar_encoder(tracker_radar)
        tracker_part = torch.cat([tracker_scalar, tracker_radar_emb], dim=-1)
        
        target_start = tracker_end
        target_scalar = critic_obs[:, target_start:target_start + NetParameters.PRIVILEGED_SCALAR_LEN]
        target_radar = critic_obs[:, target_start + NetParameters.PRIVILEGED_SCALAR_LEN:]
        target_radar_emb = self.radar_encoder(target_radar)
        target_part = torch.cat([target_scalar, target_radar_emb], dim=-1)
        
        return torch.cat([tracker_part, target_part], dim=-1)
    
    def forward_skill(self, actor_obs, skill_tag):
        encoded = self._encode_actor_obs(actor_obs)
        public_feature = self.mlp_public(encoded)
        
        if skill_tag == "track":
            skill_feature = self.mlp_track(encoded)
            combined = torch.cat([skill_feature, public_feature], dim=-1)
            action_mean = self.track_action_head(combined)
        elif skill_tag == "safe":
            skill_feature = self.mlp_safe(encoded)
            combined = torch.cat([skill_feature, public_feature], dim=-1)
            action_mean = self.safe_action_head(combined)
        else:
            raise ValueError(f"Unknown skill_tag: {skill_tag}")
        
        log_std = self.log_std.expand_as(action_mean)
        return action_mean, log_std, combined
    
    def forward_both_skills(self, actor_obs):
        encoded = self._encode_actor_obs(actor_obs)
        public_feature = self.mlp_public(encoded)
        
        track_feature = self.mlp_track(encoded)
        track_combined = torch.cat([track_feature, public_feature], dim=-1)
        track_mean = self.track_action_head(track_combined)
        
        safe_feature = self.mlp_safe(encoded)
        safe_combined = torch.cat([safe_feature, public_feature], dim=-1)
        safe_mean = self.safe_action_head(safe_combined)
        
        log_std = self.log_std.expand_as(track_mean)
        return track_mean, safe_mean, log_std
    
    def forward_value(self, critic_obs):
        encoded = self._encode_critic_obs(critic_obs)
        features = self.critic_backbone(encoded)
        value = self.value_head(features)
        return value

class HighLevelPolicy(nn.Module):
    def __init__(self, device='cpu'):
        super(HighLevelPolicy, self).__init__()
        self.device = device
        hidden_sizes = NetParameters.HIGH_LEVEL_HIDDEN
        
        self.radar_encoder = RadarEncoder()
        
        layers = []
        input_dim = NetParameters.ACTOR_VECTOR_LEN
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.weight_head = nn.Linear(input_dim, 2)
        self.value_head = nn.Linear(input_dim, 1)
        
    def _encode_obs(self, actor_obs):
        actor_scalar = actor_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        actor_radar = actor_obs[:, NetParameters.ACTOR_SCALAR_LEN:]
        actor_radar_emb = self.radar_encoder(actor_radar)
        return torch.cat([actor_scalar, actor_radar_emb], dim=-1)
    
    def forward(self, actor_obs):
        encoded = self._encode_obs(actor_obs)
        features = self.backbone(encoded)
        logits = self.weight_head(features)
        weights = torch.softmax(logits, dim=-1)
        value = self.value_head(features)
        return weights, logits, value

class StarNetCombined(nn.Module):
    def __init__(self, device='cpu'):
        super(StarNetCombined, self).__init__()
        self.device = device
        self.skill_net = StarNet(device)
        self.high_level = HighLevelPolicy(device)
        
    def forward(self, actor_obs, critic_obs):
        high_weights, high_logits, high_value = self.high_level(actor_obs)
        track_mean, safe_mean, log_std = self.skill_net.forward_both_skills(actor_obs)
        
        if getattr(TrainingParameters, "HARD_SKILL_SELECTION", False):
            skill_idx = torch.argmax(high_weights, dim=-1)
            skill_idx_expanded = skill_idx.unsqueeze(-1).expand_as(track_mean)
            blended_mean = torch.where(
                skill_idx_expanded == 0,
                track_mean,
                safe_mean,
            )
        else:
            w_track = high_weights[:, 0:1]
            w_safe = high_weights[:, 1:2]
            blended_mean = w_track * track_mean + w_safe * safe_mean
        
        skill_value = self.skill_net.forward_value(critic_obs)
        return blended_mean, skill_value, log_std, high_weights, track_mean, safe_mean, high_logits
    
    def freeze_skills(self):
        for param in self.skill_net.parameters():
            param.requires_grad = False
            
    def unfreeze_skills(self):
        for param in self.skill_net.parameters():
            param.requires_grad = True
