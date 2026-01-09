import numpy as np
import torch
from star_hrl.alg_parameters_star import NetParameters, TrainingParameters
from star_hrl.nets_star import StarNetCombined

class StarModel:
    def __init__(self, device, global_model=False):
        self.device = device
        self.network = StarNetCombined(device).to(device)
        
        if global_model:
            if TrainingParameters.FREEZE_SKILLS:
                self.network.freeze_skills()
                self.net_optimizer = torch.optim.Adam(
                    self.network.high_level.parameters(),
                    lr=TrainingParameters.HIGH_LEVEL_LR
                )
            else:
                self.net_optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=TrainingParameters.lr
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
    
    @torch.no_grad()
    def step(self, actor_obs, critic_obs):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        
        blended_mean, value, log_std, high_weights, _, _, _ = self.network(
            actor_tensor, critic_tensor
        )
        
        std = torch.exp(log_std)
        eps = torch.randn_like(blended_mean)
        pre_tanh = blended_mean + eps * std
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, blended_mean, log_std)
        
        return (
            action[0].cpu().numpy(),
            pre_tanh[0].cpu().numpy(),
            float(value.item()),
            float(log_prob.item()),
            high_weights[0].cpu().numpy()
        )
    
    @torch.no_grad()
    def evaluate(self, actor_obs, critic_obs, greedy=True):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        
        blended_mean, value, log_std, high_weights, _, _, _ = self.network(
            actor_tensor, critic_tensor
        )
        
        if greedy:
            pre_tanh = blended_mean
        else:
            std = torch.exp(log_std)
            pre_tanh = blended_mean + std * torch.randn_like(blended_mean)
            
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, blended_mean, log_std)
        
        return (
            action[0].cpu().numpy(),
            pre_tanh[0].cpu().numpy(),
            float(value.item()),
            float(log_prob.item()),
            high_weights[0].cpu().numpy()
        )
    
    def train(self, actor_obs, critic_obs, returns, values, actions, 
              old_log_probs, mask=None, writer=None, global_step=None):
        self.net_optimizer.zero_grad(set_to_none=True)
        
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        
        if actor_obs.dim() == 1: actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1: critic_obs = critic_obs.unsqueeze(0)
        if returns.dim() == 0: returns = returns.unsqueeze(0)
        if values.dim() == 0: values = values.unsqueeze(0)
        if old_log_probs.dim() == 0: old_log_probs = old_log_probs.unsqueeze(0)
        if actions.dim() == 1: actions = actions.unsqueeze(0)
        
        if mask is None:
            mask = torch.ones_like(returns, dtype=torch.float32, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            if mask.dim() == 0: mask = mask.unsqueeze(0)
        
        blended_mean, new_values, log_std, high_weights, track_mean, safe_mean, high_logits = \
            self.network(actor_obs, critic_obs)
        
        new_values = new_values.squeeze(-1)
        new_action_log_probs = self._log_prob_from_pre_tanh(actions, blended_mean, log_std)
        
        # Advantage
        raw_advantages = returns - values.squeeze(-1)
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_std = float(raw_advantages[valid_mask].std().item())
            adv_mean = float(raw_advantages[valid_mask].mean().item())
            advantages = (raw_advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = raw_advantages * 0.0
            adv_std = 0.0
            adv_mean = 0.0
        advantages = advantages * mask
        
        ratio = torch.exp(new_action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                            1.0 + TrainingParameters.CLIP_RANGE) * advantages
        policy_loss_t = -torch.min(surr1, surr2).sum() / mask.sum().clamp_min(1.0)
        policy_loss = policy_loss_t.item()
        
        entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum(dim=-1)
        entropy_loss_t = -(entropy * mask).sum() / mask.sum().clamp_min(1.0)
        entropy_loss = entropy_loss_t.item()
        
        value_clipped = values.squeeze(-1) + torch.clamp(
            new_values - values.squeeze(-1),
            -TrainingParameters.VALUE_CLIP_RANGE,
            TrainingParameters.VALUE_CLIP_RANGE
        )
        v_loss1 = (new_values - returns) ** 2
        v_loss2 = (value_clipped - returns) ** 2
        value_loss_t = (torch.max(v_loss1, v_loss2) * mask).sum() / mask.sum().clamp_min(1.0)
        value_loss = value_loss_t.item()
        
        total_loss_t = (policy_loss_t + 
                        TrainingParameters.EX_VALUE_COEF * value_loss_t + 
                        TrainingParameters.ENTROPY_COEF * entropy_loss_t)
        
        total_loss_t.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), TrainingParameters.MAX_GRAD_NORM
        ).item())
        
        self.net_optimizer.step()
        
        with torch.no_grad():
            approx_kl = (old_log_probs - new_action_log_probs).mean().item()
            clipfrac = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()
        
        high_weight_mean = high_weights.detach().mean(dim=0).cpu().numpy()
        
        losses = [
            total_loss_t.item(), policy_loss, entropy_loss, value_loss,
            adv_std, approx_kl, 0.0, clipfrac, grad_norm, adv_mean,
        ]
        
        return {
            'losses': losses,
            'high_weights': high_weight_mean
        }
    
    def load_skill_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if 'skill_net' in checkpoint:
            self.network.skill_net.load_state_dict(checkpoint['skill_net'])
        else:
            self.network.skill_net.load_state_dict(checkpoint)
        print(f"Loaded skill weights from {path}")
        
    def save_checkpoint(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'skill_net': self.network.skill_net.state_dict(),
            'high_level': self.network.high_level.state_dict(),
            'optimizer': self.net_optimizer.state_dict() if self.net_optimizer else None,
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        if self.net_optimizer and checkpoint.get('optimizer'):
            self.net_optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {path}")
    
    def update_learning_rate(self, new_lr):
        for group in self.net_optimizer.param_groups:
            group['lr'] = new_lr
        self.current_lr = new_lr
