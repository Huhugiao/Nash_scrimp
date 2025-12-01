import importlib
import numpy as np
import torch
from mlp.alg_parameters_mlp import NetParameters, TrainingParameters
from mlp.nets_mlp import ProtectingNetMLP
from util import write_to_tensorboard

def get_grad_projection(g1, g2):
    gradient_dot = torch.dot(g1.view(-1), g2.view(-1))
    g2_norm = torch.norm(g2)
    if gradient_dot < 0:
        g1_projection = (gradient_dot / (g2_norm**2 + 1e-8)) * g2
        return g1 - g1_projection
    else:
        return g1

class Model(object):
    @staticmethod
    def _angle_limit() -> float:
        try:
            map_cfg = importlib.import_module('map_config')
            limit = float(getattr(map_cfg, 'max_turn_deg', 45.0))
        except Exception:
            limit = 45.0
        return max(1.0, limit)

    @staticmethod
    def to_normalized_action(pair):
        max_turn = Model._angle_limit()
        angle_norm = float(np.clip(pair[0] / max_turn, -1.0, 1.0))
        speed_norm = float(np.clip(pair[1], 0.0, 1.0) * 2.0 - 1.0)
        return np.array([angle_norm, speed_norm], dtype=np.float32)

    @staticmethod
    def to_pre_tanh(action_normalized):
        clipped = np.clip(action_normalized, -0.999999, 0.999999)
        return np.arctanh(clipped).astype(np.float32)

    @staticmethod
    def from_normalized(action_normalized):
        max_turn = Model._angle_limit()
        angle = float(np.clip(action_normalized[0], -1.0, 1.0) * max_turn)
        speed = float(np.clip((action_normalized[1] + 1.0) * 0.5, 0.0, 1.0))
        return angle, speed

    def __init__(self, device, global_model=False):
        self.device = device
        self.network = ProtectingNetMLP().to(device)
        
        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=TrainingParameters.lr
            )
        else:
            self.net_optimizer = None
            self.net_scaler = None
            
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

    def train(self, actor_obs, critic_obs, returns, values, actions, old_log_probs, mask=None, il_batch=None,
              writer=None, global_step=None, perf_dict=None):
        """
        Simplified MLP trainer — inputs must be sample-flattened:
          actor_obs: (N, actor_dim)
          critic_obs: (N, critic_dim)
          returns, values, old_log_probs, mask: (N,)
          actions: (N, action_dim)  (pre-tanh)
        il_batch (optional) should be a dict with flattened arrays of same sample-first shapes.
        """
        self.net_optimizer.zero_grad(set_to_none=True)

        # Convert to tensors — expect flattened per-sample inputs
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)

        # Ensure dimensions are sample-first (N, dim) or (N,)
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if old_log_probs.dim() == 0:
            old_log_probs = old_log_probs.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if mask is None:
            mask = torch.ones_like(returns, dtype=torch.float32, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)

        # final shapes:
        # actor_obs: (N, actor_dim)
        # critic_obs: (N, critic_dim)
        # returns, values, old_log_probs, mask: (N,)
        # actions: (N, action_dim)
        N = actor_obs.shape[0]

        # 1. Compute IL Gradients
        il_grads = None
        il_loss_value = None
        il_filter_ratio = None
        
        if il_batch is not None:
            il_actor = torch.as_tensor(il_batch['actor_obs'], dtype=torch.float32, device=self.device)
            il_critic = torch.as_tensor(il_batch['critic_obs'], dtype=torch.float32, device=self.device)
            il_actions = torch.as_tensor(il_batch['actions'], dtype=torch.float32, device=self.device)
            il_mask = torch.as_tensor(il_batch.get('mask', None), dtype=torch.float32, device=self.device) if 'mask' in il_batch else None

            if il_actor.dim() == 1: il_actor = il_actor.unsqueeze(0)
            if il_critic.dim() == 1: il_critic = il_critic.unsqueeze(0)
            if il_actions.dim() == 1: il_actions = il_actions.unsqueeze(0)
            
            if il_mask is None:
                gating_mask = torch.ones(il_actor.size(0), device=self.device)
            else:
                gating_mask = il_mask.reshape(-1)

            if TrainingParameters.USE_Q_CRITIC:
                with torch.no_grad():
                    _, il_value_flat, _ = self.network(il_actor, il_critic)
                    q_expert_flat = self.network.forward_q(il_critic, il_actions).squeeze(-1)
                    conf_mask = (q_expert_flat - il_value_flat.squeeze(-1) > TrainingParameters.IL_FILTER_THRESHOLD).float()
                il_filter_ratio = float(conf_mask.mean().item())
                gating_mask = gating_mask * conf_mask
            else:
                il_filter_ratio = 1.0

            if gating_mask.sum() > 0:
                mean_il, _, _ = self.network(il_actor, il_critic)
                pred_actions = torch.tanh(mean_il)
                mse = ((pred_actions - il_actions) ** 2).sum(dim=-1)
                il_loss = (mse * gating_mask).sum() / gating_mask.sum()
                il_loss_value = float(il_loss.item())
                
                self.net_optimizer.zero_grad(set_to_none=True)
                il_loss.backward()
                
                # Save IL gradients
                il_grads = []
                for param in self.network.parameters():
                    if param.grad is not None:
                        il_grads.append(param.grad.clone())
                    else:
                        il_grads.append(torch.zeros_like(param))
            
            # Clear grads for RL step
            self.net_optimizer.zero_grad(set_to_none=True)

        # 2. Compute RL Gradients
        # PPO forward + loss computation (flattened)
        mean_flat, value_flat, log_std_flat = self.network(actor_obs, critic_obs)
        new_values = value_flat.squeeze(-1)  # (N,)
        new_action_log_probs = self._log_prob_from_pre_tanh(actions, mean_flat, log_std_flat)  # (N,)

        raw_advantages = returns - values.squeeze(-1)
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            advantages = ((raw_advantages - raw_advantages[valid_mask].mean()) /
                          (raw_advantages[valid_mask].std() + 1e-8))
        else:
            advantages = raw_advantages * 0.0

        advantages = advantages * mask

        ratio = torch.exp(new_action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                            1.0 + TrainingParameters.CLIP_RANGE) * advantages
        policy_loss = -torch.min(surr1, surr2).sum() / mask.sum().clamp_min(1.0)

        entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std_flat).sum(dim=-1)
        entropy_loss = -(entropy * mask).sum() / mask.sum().clamp_min(1.0)

        value_clipped = values.squeeze(-1) + torch.clamp(new_values - values.squeeze(-1),
                                             -TrainingParameters.VALUE_CLIP_RANGE,
                                             TrainingParameters.VALUE_CLIP_RANGE)
        v_loss1 = (new_values - returns) ** 2
        v_loss2 = (value_clipped - returns) ** 2
        value_loss = (torch.max(v_loss1, v_loss2) * mask).sum() / mask.sum().clamp_min(1.0)

        total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss + TrainingParameters.ENTROPY_COEF * entropy_loss
        q_loss_value = None
        if TrainingParameters.USE_Q_CRITIC:
            agent_actions_flat = torch.tanh(actions).reshape(-1, actions.shape[-1])
            returns_flat = returns.reshape(-1)
            q_pred_flat = self.network.forward_q(critic_obs, agent_actions_flat).squeeze(-1)
            q_loss = ((q_pred_flat - returns_flat) ** 2 * mask.reshape(-1)).sum() / mask.reshape(-1).sum().clamp_min(1.0)
            total_loss = total_loss + TrainingParameters.Q_LOSS_COEF * q_loss
            q_loss_value = float(q_loss.item())

        total_loss.backward()
        
        # 3. Gradient Projection
        if il_grads is not None:
            # Flatten gradients for projection
            g_il_flat = torch.cat([g.view(-1) for g in il_grads])
            
            rl_grads = []
            for param in self.network.parameters():
                if param.grad is not None:
                    rl_grads.append(param.grad)
                else:
                    rl_grads.append(torch.zeros_like(param))
            g_rl_flat = torch.cat([g.view(-1) for g in rl_grads])
            
            # Project RL gradient onto orthogonal complement of IL gradient
            # if they conflict (dot product < 0)
            dot_prod = torch.dot(g_rl_flat, g_il_flat)
            if dot_prod < 0:
                il_norm_sq = torch.dot(g_il_flat, g_il_flat)
                if il_norm_sq > 1e-8:
                    proj = (dot_prod / il_norm_sq) * g_il_flat
                    g_rl_flat = g_rl_flat - proj
            
            # Combine gradients: g_final = g_il + g_rl_projected
            g_final_flat = g_il_flat + g_rl_flat
            
            # Unflatten and set gradients
            idx = 0
            for param in self.network.parameters():
                numel = param.numel()
                param.grad = g_final_flat[idx:idx+numel].view(param.shape)
                idx += numel

        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        for name, param in self.network.named_parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad)

        self.net_optimizer.step()

        # logging metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_action_log_probs).mean()
            clipfrac = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean()

        losses = [total_loss.item(), policy_loss.item(), entropy_loss.item(), value_loss.item(),
                  float(raw_advantages[valid_mask].std().item()) if valid_mask.any() else 0.0,
                  approx_kl.item(), 0.0, clipfrac.item(), float(grad_norm),
                  float(raw_advantages[valid_mask].mean().item()) if valid_mask.any() else 0.0]
        return {'losses': losses, 'il_loss': il_loss_value, 'q_loss': q_loss_value, 'il_filter_ratio': il_filter_ratio}

    def imitation_train(self, actor_obs, critic_obs, optimal_actions, writer=None, global_step=None):
        # Expect flattened per-sample inputs: actor_obs (N, actor_dim), critic_obs (N, critic_dim), optimal_actions (N, action_dim)
        self.net_optimizer.zero_grad(set_to_none=True)
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.float32, device=self.device)

        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        if optimal_actions.dim() == 1:
            optimal_actions = optimal_actions.unsqueeze(0)

        gate_mask = torch.ones(optimal_actions.shape[0], device=self.device)
        gate_ratio = 1.0
        if TrainingParameters.USE_Q_CRITIC:
            with torch.no_grad():
                _, value_flat, _ = self.network(actor_obs, critic_obs)
                q_expert_flat = self.network.forward_q(critic_obs, optimal_actions).squeeze(-1)
                adv_flat = q_expert_flat - value_flat.squeeze(-1)
                gate_mask = (adv_flat > TrainingParameters.IL_FILTER_THRESHOLD).float()
            gate_ratio = float(gate_mask.mean().item())
        if gate_mask.sum() <= 0:
            if writer is not None and global_step is not None:
                writer.add_scalar('IL/filter_ratio', gate_ratio, int(global_step))
            return [None, 0.0]
        mean, _, _ = self.network(actor_obs, critic_obs)
        pred_actions = torch.tanh(mean)
        il_loss = (((pred_actions - optimal_actions) ** 2).sum(dim=-1) * gate_mask).sum() / gate_mask.sum()
        if torch.isfinite(il_loss):
            il_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
            self.net_optimizer.step()
        else:
            grad_norm = 0.0
        return [float(il_loss.item()), float(grad_norm)]

    def update_learning_rate(self, new_lr):
        for group in self.net_optimizer.param_groups:
            group['lr'] = new_lr
        self.current_lr = new_lr