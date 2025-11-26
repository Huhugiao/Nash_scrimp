import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from mlp.alg_parameters_mlp import NetParameters, TrainingParameters
from mlp.nets_mlp import ProtectingNetMLP
from lstm.model_lstm import Model as LSTMModel # Reuse static methods

class Model(object):
    # Reuse static helpers
    to_normalized_action = LSTMModel.to_normalized_action
    to_pre_tanh = LSTMModel.to_pre_tanh
    from_normalized = LSTMModel.from_normalized
    _angle_limit = LSTMModel._angle_limit

    def __init__(self, device, global_model=False):
        self.device = device
        self.network = ProtectingNetMLP().to(device)
        
        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=TrainingParameters.lr
            )
            self.net_scaler = GradScaler()
        else:
            self.net_optimizer = None
            self.net_scaler = None
            
        self.network.train()
        self.current_lr = TrainingParameters.lr

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

    @torch.no_grad()
    def step(self, actor_obs, critic_obs):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        
        mean, value, log_std = self.network(actor_tensor, critic_tensor)
        
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        
        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(), \
               float(value.item()), float(log_prob.item())

    @torch.no_grad()
    def evaluate(self, actor_obs, critic_obs, greedy=True):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        
        mean, value, log_std = self.network(actor_tensor, critic_tensor)
        
        pre_tanh = mean if greedy else mean + torch.exp(log_std) * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        
        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(), \
               float(value.item()), float(log_prob.item())

    def train(self, actor_obs, critic_obs, returns, values, actions, old_log_probs):
        self.net_optimizer.zero_grad(set_to_none=True)
        
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        
        with autocast():
            policy_mean, new_values, policy_log_std = self.network(actor_obs, critic_obs)
            new_values = new_values.squeeze(-1)
            
            new_action_log_probs = self._log_prob_from_pre_tanh(actions, policy_mean, policy_log_std)
            
            raw_advantages = returns - values
            if raw_advantages.numel() > 1:
                advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
            else:
                advantages = raw_advantages
            
            ratio = torch.exp(new_action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE, 
                                1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + policy_log_std).sum(dim=-1)
            entropy_loss = -entropy.mean()
            
            value_clipped = values + torch.clamp(new_values - values, 
                                                 -TrainingParameters.VALUE_CLIP_RANGE,
                                                 TrainingParameters.VALUE_CLIP_RANGE)
            v_loss1 = (new_values - returns) ** 2
            v_loss2 = (value_clipped - returns) ** 2
            value_loss = torch.max(v_loss1, v_loss2).mean()
            
            total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss + TrainingParameters.ENTROPY_COEF * entropy_loss
            
        self.net_scaler.scale(total_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()
        
        approx_kl = (old_log_probs - new_action_log_probs).mean()
        clipfrac = ((ratio - 1.0).abs() > TrainingParameters.CLIP_RANGE).float().mean()
        
        return [
            total_loss.item(), policy_loss.item(), entropy_loss.item(), value_loss.item(),
            0.0, approx_kl.item(), 0.0, clipfrac.item(), grad_norm.item(), 0.0
        ]

    def imitation_train(self, actor_obs, critic_obs, optimal_actions):
        self.net_optimizer.zero_grad(set_to_none=True)
        
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.float32, device=self.device)
        
        with autocast():
            mean, _, log_std = self.network(actor_obs, critic_obs)
            log_prob = self._log_prob_from_pre_tanh(optimal_actions, mean, log_std)
            loss = -log_prob.mean()
            
        self.net_scaler.scale(loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()
        
        return [loss.item(), grad_norm.item()]

    def set_weights(self, weights):
        self.network.load_state_dict(weights)
        
    def get_weights(self):
        return {k: v.cpu() for k, v in self.network.state_dict().items()}
        
    def update_learning_rate(self, new_lr):
        for group in self.net_optimizer.param_groups:
            group['lr'] = new_lr
