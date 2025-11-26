import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from typing import Optional, Tuple, List

from mha.alg_parameters_mha import NetParameters, TrainingParameters
from mha.nets_mha import ProtectingNetMHA
from lstm.model_lstm import Model as LSTMModel # Reuse static methods

def get_grad_projection(g1, g2):
    """
    Project gradient g1 (RL) onto the normal plane of g2 (IL) if they conflict.
    Mimics CoFeDIRL logic.
    """
    gradient_dot = torch.dot(g1.view(-1), g2.view(-1))
    g2_norm = torch.norm(g2)
    
    if gradient_dot < 0:
        # Project g1 onto g2 direction
        g1_projection = (gradient_dot / (g2_norm**2 + 1e-8)) * g2
        # Remove the conflicting component
        g1_projection_normal = g1 - g1_projection
        return g1_projection_normal
    else:
        return g1

class Model(object):
    # Reuse static helpers
    to_normalized_action = LSTMModel.to_normalized_action
    to_pre_tanh = LSTMModel.to_pre_tanh
    from_normalized = LSTMModel.from_normalized
    _angle_limit = LSTMModel._angle_limit

    def __init__(self, device, global_model=False):
        self.device = device
        self.network = ProtectingNetMHA().to(device)
        
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
        # Ensure (Batch, Seq, Dim)
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0).unsqueeze(0)
        elif input_vector.dim() == 2:
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
    def step(self, actor_obs, critic_obs, history_state=None):
        """
        Inference step.
        history_state: Tuple(actor_history_buffer, critic_history_buffer)
        Buffers are numpy arrays of shape (T, Dim)
        """
        if history_state is None:
            # Initialize empty history
            actor_hist = np.zeros((0, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32)
            critic_hist = np.zeros((0, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32)
        else:
            actor_hist, critic_hist = history_state

        # Append current obs
        actor_hist = np.concatenate([actor_hist, actor_obs.reshape(1, -1)], axis=0)
        critic_hist = np.concatenate([critic_hist, critic_obs.reshape(1, -1)], axis=0)
        
        # Clip to context window
        win_size = NetParameters.CONTEXT_WINDOW
        if actor_hist.shape[0] > win_size:
            actor_hist = actor_hist[-win_size:]
            critic_hist = critic_hist[-win_size:]
            
        # Prepare input (1, T, D)
        actor_tensor = self._to_tensor(actor_hist)
        critic_tensor = self._to_tensor(critic_hist)
        
        mean, value, log_std = self.network(actor_tensor, critic_tensor)
        
        # Take last timestep
        mean = mean[:, -1, :]
        value = value[:, -1, :]
        log_std = log_std[:, -1, :]
        
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        
        new_history = (actor_hist, critic_hist)
        
        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(), \
               new_history, float(value.item()), float(log_prob.item())

    @torch.no_grad()
    def evaluate(self, actor_obs, critic_obs, history_state=None, greedy=True):
        if history_state is None:
            actor_hist = np.zeros((0, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32)
            critic_hist = np.zeros((0, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32)
        else:
            actor_hist, critic_hist = history_state

        actor_hist = np.concatenate([actor_hist, actor_obs.reshape(1, -1)], axis=0)
        critic_hist = np.concatenate([critic_hist, critic_obs.reshape(1, -1)], axis=0)
        
        win_size = NetParameters.CONTEXT_WINDOW
        if actor_hist.shape[0] > win_size:
            actor_hist = actor_hist[-win_size:]
            critic_hist = critic_hist[-win_size:]
            
        actor_tensor = self._to_tensor(actor_hist)
        critic_tensor = self._to_tensor(critic_hist)
        
        mean, value, log_std = self.network(actor_tensor, critic_tensor)
        
        mean = mean[:, -1, :]
        value = value[:, -1, :]
        log_std = log_std[:, -1, :]
        
        pre_tanh = mean if greedy else mean + torch.exp(log_std) * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        
        new_history = (actor_hist, critic_hist)
        
        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(), \
               new_history, float(value.item()), float(log_prob.item())

    def train(self, actor_obs, critic_obs, returns, values, actions, old_log_probs, mask=None, il_batch=None):
        # Inputs are (Batch, Seq, Dim)
        self.net_optimizer.zero_grad(set_to_none=True)
        
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        
        if mask is None:
            mask = torch.ones_like(returns)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            
        valid_steps = mask.sum().clamp_min(1.0)
        
        # 1. Compute RL Gradients
        with autocast():
            policy_mean, new_values, policy_log_std = self.network(actor_obs, critic_obs)
            new_values = new_values.squeeze(-1)
            
            # Calculate losses
            new_action_log_probs = self._log_prob_from_pre_tanh(actions, policy_mean, policy_log_std)
            
            raw_advantages = returns - values
            # Normalize advantages
            valid_mask = mask > 0
            if valid_mask.sum() > 1:
                adv_mean = raw_advantages[valid_mask].mean()
                adv_std = raw_advantages[valid_mask].std()
                advantages = (raw_advantages - adv_mean) / (adv_std + 1e-8)
            else:
                advantages = raw_advantages
            advantages = advantages * mask
            
            ratio = torch.exp(new_action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE, 
                                1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss = -torch.min(surr1, surr2).sum() / valid_steps
            
            entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + policy_log_std).sum(dim=-1)
            entropy_loss = -entropy.sum() / valid_steps
            
            value_clipped = values + torch.clamp(new_values - values, 
                                                 -TrainingParameters.VALUE_CLIP_RANGE,
                                                 TrainingParameters.VALUE_CLIP_RANGE)
            v_loss1 = (new_values - returns) ** 2
            v_loss2 = (value_clipped - returns) ** 2
            value_loss = torch.max(v_loss1, v_loss2).sum() / valid_steps
            
            total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss + TrainingParameters.ENTROPY_COEF * entropy_loss
            
        self.net_scaler.scale(total_loss).backward()
        
        # 2. Gradient Projection (if IL data provided)
        if il_batch is not None:
            self.net_scaler.unscale_(self.net_optimizer)
            
            # Store RL gradients
            rl_grads = {}
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    rl_grads[name] = param.grad.clone()
            
            # Zero grads to compute IL gradients
            self.net_optimizer.zero_grad(set_to_none=True)
            
            # Prepare IL data
            il_actor = torch.as_tensor(il_batch['actor_obs'], dtype=torch.float32, device=self.device)
            il_critic = torch.as_tensor(il_batch['critic_obs'], dtype=torch.float32, device=self.device)
            il_actions = torch.as_tensor(il_batch['actions'], dtype=torch.float32, device=self.device)
            il_mask = torch.as_tensor(il_batch['mask'], dtype=torch.float32, device=self.device) if 'mask' in il_batch else None
            
            with autocast():
                mean, _, log_std = self.network(il_actor, il_critic)
                il_log_prob = self._log_prob_from_pre_tanh(il_actions, mean, log_std)
                if il_mask is not None:
                    il_loss = -(il_log_prob * il_mask).sum() / il_mask.sum().clamp_min(1.0)
                else:
                    il_loss = -il_log_prob.mean()
            
            self.net_scaler.scale(il_loss).backward()
            
            # Manually unscale IL gradients because unscale_() can only be called once per step
            current_scale = self.net_scaler.get_scale()
            inv_scale = 1.0 / current_scale
            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
            
            # Project RL gradients onto IL gradients
            for name, param in self.network.named_parameters():
                if param.grad is not None and name in rl_grads:
                    g_il = param.grad
                    g_rl = rl_grads[name]
                    # Project RL gradient
                    param.grad = get_grad_projection(g_rl, g_il)
                elif name in rl_grads:
                    # If no IL gradient, keep RL gradient
                    param.grad = rl_grads[name]
        else:
            self.net_scaler.unscale_(self.net_optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()
        
        approx_kl = ((old_log_probs - new_action_log_probs) * mask).sum() / valid_steps
        clipfrac = ((ratio - 1.0).abs() > TrainingParameters.CLIP_RANGE).float().sum() / valid_steps
        
        return [
            total_loss.item(), policy_loss.item(), entropy_loss.item(), value_loss.item(),
            0.0, approx_kl.item(), 0.0, clipfrac.item(), grad_norm.item(), 0.0
        ]

    def imitation_train(self, actor_obs, critic_obs, optimal_actions):
        self.net_optimizer.zero_grad(set_to_none=True)
        
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.float32, device=self.device)
        
        if actor_obs.dim() == 2:
            actor_obs = actor_obs.unsqueeze(1)
            critic_obs = critic_obs.unsqueeze(1)
            
        with autocast():
            mean, _, log_std = self.network(actor_obs, critic_obs)
            # mean is (B, 1, ActionDim)
            mean = mean.squeeze(1)
            log_std = log_std.squeeze(1)
            
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
