import torch
import torch.nn as nn
import torch.nn.functional as F
from alg_parameters import EnvParameters, NetParameters


class ProtectingNetLSTM(nn.Module):
    """
    LSTM-based network architecture with asymmetric observations.
    Actor uses 23-dim observation with LSTM for temporal modeling.
    Critic uses (23-dim observation + opponent context) with LSTM.
    """

    def __init__(self, lstm_hidden_size=128, num_lstm_layers=1):
        super(ProtectingNetLSTM, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        # Actor LSTM and MLP
        self.actor_lstm = nn.LSTM(
            input_size=NetParameters.ACTOR_VECTOR_LEN,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        self.actor_encoder = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, getattr(NetParameters, 'ACTION_DIM', 2))
        )
        self.log_std = nn.Parameter(torch.zeros(getattr(NetParameters, 'ACTION_DIM', 2)))

        # Critic LSTM and MLP
        self.critic_lstm = nn.LSTM(
            input_size=NetParameters.CRITIC_VECTOR_LEN,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        self.critic_encoder = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for LSTM
        for name, param in self.actor_lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
                
        for name, param in self.critic_lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, actor_obs, critic_obs, actor_hidden=None, critic_hidden=None):
        """
        Forward pass with LSTM hidden state handling.
        
        Args:
            actor_obs: (batch_size, seq_len, actor_dim) or (batch_size, actor_dim)
            critic_obs: (batch_size, seq_len, critic_dim) or (batch_size, critic_dim)
            actor_hidden: tuple of (h, c) for actor LSTM
            critic_hidden: tuple of (h, c) for critic LSTM
        
        Returns:
            policy: action probabilities
            value: state value
            policy_logits: raw logits
            actor_hidden: updated actor LSTM state
            critic_hidden: updated critic LSTM state
        """
        # Handle input dimensions
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0).unsqueeze(0)  # Add batch and seq dim
        elif actor_obs.dim() == 2:
            actor_obs = actor_obs.unsqueeze(1)  # Add seq dim
            
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0).unsqueeze(0)  # Add batch and seq dim
        elif critic_obs.dim() == 2:
            critic_obs = critic_obs.unsqueeze(1)  # Add seq dim

        batch_size = actor_obs.size(0)
        
        # Initialize hidden states if not provided
        if actor_hidden is None:
            actor_hidden = self.init_hidden(batch_size, actor_obs.device)
        if critic_hidden is None:
            critic_hidden = self.init_hidden(batch_size, critic_obs.device)

        # Actor LSTM path
        actor_lstm_out, new_actor_hidden = self.actor_lstm(actor_obs, actor_hidden)
        # Take the last timestep output
        actor_features = actor_lstm_out[:, -1, :]
        actor_features = self.actor_encoder(actor_features)
        policy_mean = self.policy_head(actor_features)
        policy_log_std = self.log_std.unsqueeze(0).expand(policy_mean.size(0), -1)

        # Critic LSTM path
        critic_lstm_out, new_critic_hidden = self.critic_lstm(critic_obs, critic_hidden)
        # Take the last timestep output
        critic_features = critic_lstm_out[:, -1, :]
        critic_features = self.critic_encoder(critic_features)
        value = self.value_head(critic_features)
        
        return policy_mean, value, policy_log_std, new_actor_hidden, new_critic_hidden

    def init_hidden(self, batch_size, device):
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h, c)

    def get_actor_features(self, actor_obs, actor_hidden=None):
        """Extract actor features for auxiliary tasks."""
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0).unsqueeze(0)
        elif actor_obs.dim() == 2:
            actor_obs = actor_obs.unsqueeze(1)
            
        batch_size = actor_obs.size(0)
        
        if actor_hidden is None:
            actor_hidden = self.init_hidden(batch_size, actor_obs.device)
            
        actor_lstm_out, new_actor_hidden = self.actor_lstm(actor_obs, actor_hidden)
        actor_features = actor_lstm_out[:, -1, :]
        actor_features = self.actor_encoder(actor_features)
        
        return actor_features, new_actor_hidden

    def get_critic_features(self, critic_obs, critic_hidden=None):
        """Extract critic features for auxiliary tasks."""
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0).unsqueeze(0)
        elif critic_obs.dim() == 2:
            critic_obs = critic_obs.unsqueeze(1)
            
        batch_size = critic_obs.size(0)
        
        if critic_hidden is None:
            critic_hidden = self.init_hidden(batch_size, critic_obs.device)
            
        critic_lstm_out, new_critic_hidden = self.critic_lstm(critic_obs, critic_hidden)
        critic_features = critic_lstm_out[:, -1, :]
        critic_features = self.critic_encoder(critic_features)
        
        return critic_features, new_critic_hidden