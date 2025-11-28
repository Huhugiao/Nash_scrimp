import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from mlp.alg_parameters_mlp import NetParameters, TrainingParameters
from mlp.nets_mlp import ProtectingNetMLP
from lstm.model_lstm import Model as LSTMModel # Reuse static methods

def get_grad_projection(g1, g2):
	gradient_dot = torch.dot(g1.view(-1), g2.view(-1))
	g2_norm = torch.norm(g2)
	if gradient_dot < 0:
		g1_projection = (gradient_dot / (g2_norm**2 + 1e-8)) * g2
		return g1 - g1_projection
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

    def train(self, actor_obs, critic_obs, returns, values, actions, old_log_probs, mask=None, il_batch=None):
        """
        actor_obs etc are expected as (Batch, Seq, Dim) to follow MHA flow.
        Internally we flatten (B*T, D) to run through MLP network, then reshape.
        Supports optional il_batch (dict) for IL gradient computation + gradient projection.
        """
        self.net_optimizer.zero_grad(set_to_none=True)

        # Convert & ensure (B, T, D)
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)

        if actor_obs.dim() == 2:
            actor_obs = actor_obs.unsqueeze(1)
            critic_obs = critic_obs.unsqueeze(1)
            actions = actions.unsqueeze(1)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
            values = values.unsqueeze(1)
            old_log_probs = old_log_probs.unsqueeze(1)
        if mask is None:
            mask = torch.ones_like(returns, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)

        B, T = actor_obs.shape[0], actor_obs.shape[1]
        actor_flat = actor_obs.reshape(-1, actor_obs.shape[-1])
        critic_flat = critic_obs.reshape(-1, critic_obs.shape[-1])

        il_grads = {}
        il_loss_value = None
        if il_batch is not None:
            il_actor = torch.as_tensor(il_batch['actor_obs'], dtype=torch.float32, device=self.device)
            il_critic = torch.as_tensor(il_batch['critic_obs'], dtype=torch.float32, device=self.device)
            il_actions = torch.as_tensor(il_batch['actions'], dtype=torch.float32, device=self.device)
            il_mask = torch.as_tensor(il_batch['mask'], dtype=torch.float32, device=self.device) if 'mask' in il_batch else None
            if il_actor.dim() == 2:
                il_actor = il_actor.unsqueeze(1)
                il_critic = il_critic.unsqueeze(1)
                il_actions = il_actions.unsqueeze(1)
                if il_mask is not None and il_mask.dim() == 1:
                    il_mask = il_mask.unsqueeze(1)

            mean_il, _, _ = self.network(il_actor.reshape(-1, il_actor.shape[-1]),
                                         il_critic.reshape(-1, il_critic.shape[-1]))
            mean_il = mean_il.reshape(il_actor.shape[0], il_actor.shape[1], -1)
            pred_actions = torch.tanh(mean_il)
            mse = ((pred_actions - il_actions) ** 2).sum(dim=-1)
            if il_mask is not None:
                il_loss = (mse * il_mask).sum() / il_mask.sum().clamp_min(1.0)
            else:
                il_loss = mse.mean()
            il_loss_value = float(il_loss.item())
            if torch.isfinite(il_loss) and il_loss_value > 0:
                il_loss.backward()
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        il_grads[name] = param.grad.clone()
                self.net_optimizer.zero_grad(set_to_none=True)

        mean_flat, value_flat, log_std_flat = self.network(actor_flat, critic_flat)
        mean = mean_flat.reshape(B, T, -1)
        value = value_flat.reshape(B, T, -1)
        log_std = log_std_flat.reshape(B, T, -1)
        new_values = value.squeeze(-1)

        new_action_log_probs = self._log_prob_from_pre_tanh(actions, mean, log_std)
        raw_advantages = returns - values
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_mean = (raw_advantages * mask).sum() / mask.sum()
            adv_std = torch.sqrt(((raw_advantages - adv_mean) ** 2 * mask).sum() / mask.sum() + 1e-8)
            advantages = (raw_advantages - adv_mean) / (adv_std + 1e-5)
        else:
            advantages = raw_advantages
        advantages = advantages * mask

        ratio = torch.exp(new_action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                            1.0 + TrainingParameters.CLIP_RANGE) * advantages
        policy_loss = -torch.min(surr1, surr2).sum() / mask.sum().clamp_min(1.0)

        entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std).sum(dim=-1)
        entropy_loss = -(entropy * mask).sum() / mask.sum().clamp_min(1.0)

        value_clipped = values + torch.clamp(new_values - values,
                                             -TrainingParameters.VALUE_CLIP_RANGE,
                                             TrainingParameters.VALUE_CLIP_RANGE)
        v_loss1 = (new_values - returns) ** 2
        v_loss2 = (value_clipped - returns) ** 2
        value_loss = (torch.max(v_loss1, v_loss2) * mask).sum() / mask.sum().clamp_min(1.0)

        total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss + TrainingParameters.ENTROPY_COEF * entropy_loss
        if not torch.isfinite(total_loss):
            return {'losses': [float('nan')] * 10, 'il_loss': il_loss_value}

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        for name, param in self.network.named_parameters():
            g_rl = param.grad
            g_il = il_grads.get(name)
            if g_rl is None and g_il is None:
                continue
            elif g_il is None:
                continue
            elif g_rl is None:
                param.grad = g_il.clone()
            else:
                param.grad = get_grad_projection(g_rl, g_il)

        self.net_optimizer.step()

        # logging metrics
        with torch.no_grad():
            log_ratio = new_action_log_probs - old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
            clipfrac = ((ratio - 1.0).abs() > TrainingParameters.CLIP_RANGE).float().mean()

        losses = [total_loss.item(), policy_loss.item(), entropy_loss.item(), value_loss.item(),
                  0.0, approx_kl.item(), 0.0, clipfrac.item(), float(grad_norm), 0.0]

        return {'losses': losses, 'il_loss': il_loss_value}

    def imitation_train(self, actor_obs, critic_obs, optimal_actions):
        # Use MSE between tanh(mean) and expert normalized actions; support (B, T, D)
        self.net_optimizer.zero_grad(set_to_none=True)
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.float32, device=self.device)
        # flatten
        if actor_obs.dim() == 3:
            actor_flat = actor_obs.reshape(-1, actor_obs.shape[-1])
            critic_flat = critic_obs.reshape(-1, critic_obs.shape[-1])
        else:
            actor_flat = actor_obs
            critic_flat = critic_obs

        mean, _, _ = self.network(actor_flat, critic_flat)
        # reshape to original sequence dims if needed
        if optimal_actions.dim() == 3:
            mean = mean.reshape(optimal_actions.shape[0], optimal_actions.shape[1], -1)
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
