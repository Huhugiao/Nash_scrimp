import torch
import torch.nn as nn
import numpy as np
from mha.alg_parameters_mha import NetParameters

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CausalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Causal Mask: Upper triangular mask with -inf
        mask = torch.ones(n, n, device=x.device).tril()
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask == 0, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        # Pre-LayerNorm Architecture
        self.attn = PreNorm(dim, CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class ProtectingNetMHA(nn.Module):
    def __init__(self):
        super(ProtectingNetMHA, self).__init__()
        
        self.hidden_dim = NetParameters.HIDDEN_DIM
        self.context_window = NetParameters.CONTEXT_WINDOW
        
        # Feature Extractors (Separate Radar)
        # Actor: 27 dims (0-10 scalar, 11-26 radar)
        self.actor_scalar_embed = nn.Linear(11, self.hidden_dim // 2)
        self.actor_radar_embed = nn.Linear(16, self.hidden_dim // 2)
        
        # Critic: 24 dims (Target Obs)
        # Target: 8 scalar + 16 radar
        critic_scalar_dim = 8
        critic_radar_dim = 16
        self.critic_scalar_embed = nn.Linear(critic_scalar_dim, self.hidden_dim // 2)
        self.critic_radar_embed = nn.Linear(critic_radar_dim, self.hidden_dim // 2)

        # Learnable Positional Encoding
        # Fix: Initialize with small random noise instead of zeros
        self.pos_embed = nn.Parameter(torch.randn(1, self.context_window, self.hidden_dim) * 0.02)

        # Transformer Encoders
        self.actor_transformer = nn.ModuleList([
            TransformerBlock(self.hidden_dim, NetParameters.N_HEADS, self.hidden_dim // NetParameters.N_HEADS, self.hidden_dim * 4)
            for _ in range(NetParameters.N_LAYERS)
        ])
        
        self.critic_transformer = nn.ModuleList([
            TransformerBlock(self.hidden_dim, NetParameters.N_HEADS, self.hidden_dim // NetParameters.N_HEADS, self.hidden_dim * 4)
            for _ in range(NetParameters.N_LAYERS)
        ])

        # Heads
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, NetParameters.ACTION_DIM)
        )
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))
        
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
        
        # Prevent vanishing gradients:
        # 1. Initialize policy head with small gain (0.01) to keep actions centered/un-saturated initially
        # This prevents Tanh from saturating early, which kills gradients.
        nn.init.orthogonal_(self.policy_head[1].weight, gain=0.01)
        nn.init.constant_(self.policy_head[1].bias, 0.0)
        
        # 2. Initialize value head with gain 1.0
        nn.init.orthogonal_(self.value_head[1].weight, gain=1.0)
        nn.init.constant_(self.value_head[1].bias, 0.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use sqrt(2) for hidden layers (GELU/ReLU) to maintain variance
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, actor_obs, critic_obs):
        # actor_obs: (B, T, 27)
        # critic_obs: (B, T, 24) -> [Target(24)]
        
        B, T, _ = actor_obs.shape
        
        # --- Actor Embedding ---
        a_scalar = actor_obs[..., :11]
        a_radar = actor_obs[..., 11:27]
        a_embed = torch.cat([
            self.actor_scalar_embed(a_scalar),
            self.actor_radar_embed(a_radar)
        ], dim=-1) # (B, T, Hidden)
        
        # --- Critic Embedding ---
        # Target: 0-24 (Scalar 0-8, Radar 8-24)
        
        c_scalar = critic_obs[..., :8]
        c_radar = critic_obs[..., 8:24]

        c_embed = torch.cat([
            self.critic_scalar_embed(c_scalar),
            self.critic_radar_embed(c_radar)
        ], dim=-1)

        # --- Positional Encoding ---
        # If T < context_window, use first T pos embeddings
        # If T > context_window, we assume input is sliced to context_window
        pos = self.pos_embed[:, :T, :]
        
        x_actor = a_embed + pos
        x_critic = c_embed + pos
        
        # --- Transformer Pass ---
        for block in self.actor_transformer:
            x_actor = block(x_actor)
            
        for block in self.critic_transformer:
            x_critic = block(x_critic)
            
        # --- Heads ---
        # We return sequences. The caller (PPO) handles slicing or indexing.
        policy_mean = self.policy_head(x_actor)
        
        # Clamp log_std to prevent collapsing to 0 (vanishing) or exploding
        clamped_log_std = self.log_std.clamp(-20, 2)
        policy_log_std = clamped_log_std.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        
        value = self.value_head(x_critic)
        
        return policy_mean, value, policy_log_std
