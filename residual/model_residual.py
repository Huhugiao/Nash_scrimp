import numpy as np
import torch
from residual.nets_residual import ResidualPolicyNetwork
from residual.alg_parameters_residual import (
    TrainingParameters, 
    NetParameters,
    ResidualRLConfig
)


class ResidualModel:
    """
    Model wrapper for training ResidualPolicyNetwork using PPO.
    Gated Residual: Uses radar + base_action as input, learns safety gate.
    """
    def __init__(self, device, global_model=False):
        self.device = device
        self.network = ResidualPolicyNetwork().to(device)
        
        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=TrainingParameters.lr,
                eps=1e-5
            )
        else:
            self.net_optimizer = None
            
        self.network.train()
        self.current_lr = TrainingParameters.lr

    def get_weights(self):
        return {name: param.cpu() for name, param in self.network.state_dict().items()}

    def set_weights(self, weights):
        self.network.load_state_dict(weights)

    def _to_tensor(self, vector):
        if isinstance(vector, np.ndarray):
            input_vector = torch.from_numpy(vector).float().to(self.device)
        elif torch.is_tensor(vector):
            input_vector = vector.to(self.device).float()
        else:
            input_vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        return torch.nan_to_num(input_vector)

    @staticmethod
    def _log_prob_from_pre_tanh(pre_tanh, mean, log_std):
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        base_log_prob = dist.log_prob(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        return (base_log_prob - log_det_jac).sum(dim=-1)

    def train(self, radar_obs=None, base_actions=None, velocity_obs=None,
              returns=None, values=None, actions=None, old_log_probs=None, mask=None,
              writer=None, global_step=None, **kwargs):
        """
        PPO training step for Residual Network.
        Uses radar + base_action + velocity as actor input.
        """
        if self.net_optimizer is None:
            raise RuntimeError("Cannot train without optimizer (not a global model)")
        
        self.net_optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        policy_loss = 0.0
        entropy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clipfrac = 0.0
        grad_norm = 0.0
        adv_mean = 0.0
        adv_std = 0.0

        if radar_obs is not None:
            # Convert to tensors
            radar_obs = torch.as_tensor(radar_obs, dtype=torch.float32, device=self.device)
            returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)

            if radar_obs.dim() == 1: radar_obs = radar_obs.unsqueeze(0)
            if returns.dim() == 0: returns = returns.unsqueeze(0)
            if values.dim() == 0: values = values.unsqueeze(0)
            if old_log_probs.dim() == 0: old_log_probs = old_log_probs.unsqueeze(0)
            if actions.dim() == 1: actions = actions.unsqueeze(0)
            
            if mask is None:
                mask = torch.ones_like(returns, dtype=torch.float32, device=self.device)
            else:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
                if mask.dim() == 0: mask = mask.unsqueeze(0)

            # Convert base_actions and velocity to tensors
            base_actions_t = torch.as_tensor(base_actions, dtype=torch.float32, device=self.device)
            if base_actions_t.dim() == 1: base_actions_t = base_actions_t.unsqueeze(0)
            
            velocity_t = torch.as_tensor(velocity_obs, dtype=torch.float32, device=self.device)
            if velocity_t.dim() == 1: velocity_t = velocity_t.unsqueeze(0)

            # Forward pass through residual network (radar + base_action + velocity)
            mean, log_std = self.network.actor(radar_obs, base_actions_t, velocity_t)
            new_values = self.network.critic(radar_obs).squeeze(-1)
            
            # Compute new log probs
            new_log_probs = self._log_prob_from_pre_tanh(actions, mean, log_std)

            # Compute advantages
            raw_advantages = returns - values.squeeze(-1)
            valid_mask = mask > 0
            if valid_mask.sum() > 1:
                adv_std = float(raw_advantages[valid_mask].std().item())
                adv_mean = float(raw_advantages[valid_mask].mean().item())
                advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
            else:
                advantages = raw_advantages * 0.0

            advantages = advantages * mask

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss_t = -torch.min(surr1, surr2).sum() / mask.sum().clamp_min(1.0)
            policy_loss = policy_loss_t.item()

            # Entropy bonus
            entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum(dim=-1)
            entropy_loss_t = -(entropy * mask).sum() / mask.sum().clamp_min(1.0)
            entropy_loss = entropy_loss_t.item()

            # Value loss (clipped)
            value_clipped = values.squeeze(-1) + torch.clamp(
                new_values - values.squeeze(-1),
                -TrainingParameters.VALUE_CLIP_RANGE,
                TrainingParameters.VALUE_CLIP_RANGE
            )
            v_loss1 = (new_values - returns) ** 2
            v_loss2 = (value_clipped - returns) ** 2
            value_loss_t = (torch.max(v_loss1, v_loss2) * mask).sum() / mask.sum().clamp_min(1.0)
            value_loss = value_loss_t.item()

            # Total loss
            total_loss_t = (policy_loss_t + 
                           TrainingParameters.EX_VALUE_COEF * value_loss_t + 
                           TrainingParameters.ENTROPY_COEF * entropy_loss_t)
            
            total_loss_t.backward()
            total_loss = total_loss_t.item()

            # Gradient clipping and optimizer step
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 
                TrainingParameters.MAX_GRAD_NORM
            ).item())
            
            # Replace NaN grads
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad)

            self.net_optimizer.step()

            # Compute stats
            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
                clipfrac = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()

        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, 0.0, clipfrac, grad_norm, adv_mean]
        return {'losses': losses, 'il_loss': None, 'q_loss': None, 'il_filter_ratio': None}

    def update_learning_rate(self, new_lr):
        """Update learning rate."""
        if self.net_optimizer is None:
            return
        for group in self.net_optimizer.param_groups:
            group['lr'] = new_lr
        self.current_lr = new_lr
