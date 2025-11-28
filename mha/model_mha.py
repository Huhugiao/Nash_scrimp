import numpy as np
import torch
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
            self.net_optimizer = torch.optim.Adam(self.network.parameters(), lr=TrainingParameters.lr)
        else:
            self.net_optimizer = None
            
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

        # --- 1. Compute IL Gradients (if available) ---
        il_grads = {}
        il_loss_value = None
        if il_batch is not None:
            il_actor = torch.as_tensor(il_batch['actor_obs'], dtype=torch.float32, device=self.device)
            il_critic = torch.as_tensor(il_batch['critic_obs'], dtype=torch.float32, device=self.device)
            il_actions = torch.as_tensor(il_batch['actions'], dtype=torch.float32, device=self.device)
            il_mask = torch.as_tensor(il_batch['mask'], dtype=torch.float32, device=self.device) if 'mask' in il_batch else None

            mean_il, _, log_std_il = self.network(il_actor, il_critic)
            
            # 修复：使用 MSE Loss
            # mean_il 是 pre_tanh，expert_actions 是 tanh 后的值
            # 所以比较 tanh(mean_il) 和 expert_actions
            pred_actions = torch.tanh(mean_il)  # 预测动作
            
            # MSE Loss
            mse = ((pred_actions - il_actions) ** 2).sum(dim=-1)  # (B, T)
            
            if il_mask is not None:
                il_loss = (mse * il_mask).sum() / il_mask.sum().clamp_min(1.0)
            else:
                il_loss = mse.mean()

            il_loss_value = float(il_loss.item())
            
            # 只有 loss 合理时才计算梯度
            if torch.isfinite(il_loss) and il_loss_value > 0:
                il_loss.backward()
                
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        il_grads[name] = param.grad.clone()
                
                self.net_optimizer.zero_grad(set_to_none=True)
        
        # --- 2. Compute RL Gradients ---
        mean, value, log_std = self.network(actor_obs, critic_obs)
        new_values = value.squeeze(-1) # (Batch, Seq)
        
        # Recover pre_tanh from actions
        action_pre_tanh = torch.atanh(torch.clamp(actions, -1.0 + 1e-7, 1.0 - 1e-7))
        new_action_log_probs = self._log_prob_from_pre_tanh(action_pre_tanh, mean, log_std)
        
        raw_advantages = returns - values
        # Normalize advantages
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_mean = raw_advantages[valid_mask].mean()
            adv_std = raw_advantages[valid_mask].std()
            advantages = (raw_advantages - adv_mean) / (adv_std + 1e-5)
        else:
            advantages = raw_advantages
        advantages = advantages * mask
        
        ratio = torch.exp(new_action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE, 
                            1.0 + TrainingParameters.CLIP_RANGE) * advantages
        policy_loss = -torch.min(surr1, surr2).sum() / valid_steps
        
        entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum(dim=-1)
        entropy_loss = -(entropy * mask).sum() / valid_steps
        
        value_clipped = values + torch.clamp(new_values - values, 
                                             -TrainingParameters.VALUE_CLIP_RANGE,
                                             TrainingParameters.VALUE_CLIP_RANGE)
        v_loss1 = (new_values - returns) ** 2
        v_loss2 = (value_clipped - returns) ** 2
        value_loss = (torch.max(v_loss1, v_loss2) * mask).sum() / valid_steps
        
        total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss + TrainingParameters.ENTROPY_COEF * entropy_loss

        # Check for invalid loss before backward
        if not torch.isfinite(total_loss):
            return {'losses': [float('nan')] * 10, 'il_loss': il_loss_value}

        total_loss.backward() 
        
        # Gradient Clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        # --- 3. Gradient Projection (RL projected onto IL) ---
        for name, param in self.network.named_parameters():
            g_rl = param.grad
            g_il = il_grads.get(name)
            
            if g_rl is None and g_il is None:
                continue
            elif g_il is None:
                pass 
            elif g_rl is None:
                param.grad = g_il.clone()
            else:
                projected = get_grad_projection(g_rl, g_il)
                param.grad = projected

        self.net_optimizer.step()
        
        # Calculate metrics for logging
        with torch.no_grad():
            log_ratio = new_action_log_probs - old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
            clipfrac = ((ratio - 1.0).abs() > TrainingParameters.CLIP_RANGE).float().mean()

        losses = [total_loss.item(), policy_loss.item(), entropy_loss.item(), value_loss.item(),
                0.0, approx_kl.item(), 0.0, clipfrac.item(), float(grad_norm), 0.0]
        
        return {'losses': losses, 'il_loss': il_loss_value}

    def imitation_train(self, actor_obs, critic_obs, optimal_actions):
        """单独的 IL 训练，使用 MSE loss"""
        self.net_optimizer.zero_grad(set_to_none=True)
        
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.float32, device=self.device)
        
        if actor_obs.dim() == 2:
            actor_obs = actor_obs.unsqueeze(1)
            critic_obs = critic_obs.unsqueeze(1)
            optimal_actions = optimal_actions.unsqueeze(1)
            
        mean, _, log_std = self.network(actor_obs, critic_obs)
        
        # MSE: 比较 tanh(mean) 和 expert_actions
        pred_actions = torch.tanh(mean)
        il_loss = ((pred_actions - optimal_actions) ** 2).mean()
            
        if torch.isfinite(il_loss):
            il_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
            self.net_optimizer.step()
        else:
            grad_norm = 0.0
        
        return [float(il_loss.item()), float(grad_norm)]

    def set_weights(self, weights):
        self.network.load_state_dict(weights)
        
    def get_weights(self):
        return {k: v.cpu() for k, v in self.network.state_dict().items()}
        
    def update_learning_rate(self, new_lr):
        for group in self.net_optimizer.param_groups:
            group['lr'] = new_lr
