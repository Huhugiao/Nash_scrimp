import importlib
from typing import Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from alg_parameters import NetParameters, TrainingParameters
from nets_lstm import ProtectingNetLSTM


class Model(object):
    """PPO model with LSTM for Protecting environment - single agent with Asymmetric Actor-Critic"""

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

    def __init__(self, device, global_model=False, lstm_hidden_size=128, num_lstm_layers=1):
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        self.network = ProtectingNetLSTM(
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers
        ).to(device)

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

    def _process_hidden_state(self, hidden_state: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                                 Tuple[torch.Tensor, torch.Tensor]]]):
        if hidden_state is None:
            return (None, None)

        if isinstance(hidden_state, tuple) and len(hidden_state) == 2:
            actor_hidden, critic_hidden = hidden_state
            return actor_hidden, critic_hidden
        return (None, None)

    def _ensure_single_hidden(self, hidden, batch_size):
        if hidden is None:
            return self.network.init_hidden(batch_size, self.device)
        h, c = hidden
        if isinstance(h, np.ndarray):
            h = torch.from_numpy(h)
        if isinstance(c, np.ndarray):
            c = torch.from_numpy(c)
        h = h.to(self.device).float()
        c = c.to(self.device).float()
        if h.dim() == 2:
            h = h.unsqueeze(1)
        if c.dim() == 2:
            c = c.unsqueeze(1)
        return (h.contiguous(), c.contiguous())

    def _detach_hidden(self, hidden):
        if hidden is None:
            return None
        h, c = hidden
        return (h.detach(), c.detach())

    @staticmethod
    def _log_prob_from_pre_tanh(pre_tanh, mean, log_std):
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        base_log_prob = dist.log_prob(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        return (base_log_prob - log_det_jac).sum(dim=-1)

    def prepare_hidden(self, hidden_state, batch_size=1):
        actor_hidden, critic_hidden = self._process_hidden_state(hidden_state)
        actor_hidden = self._ensure_single_hidden(actor_hidden, batch_size)
        critic_hidden = self._ensure_single_hidden(critic_hidden, batch_size)
        return actor_hidden, critic_hidden

    def _format_hidden_batch(self, hidden_states, batch_size):
        if hidden_states is None:
            return self.network.init_hidden(batch_size, self.device)
        h_np, c_np = hidden_states
        h = torch.as_tensor(h_np, dtype=torch.float32, device=self.device).permute(1, 0, 2).contiguous()
        c = torch.as_tensor(c_np, dtype=torch.float32, device=self.device).permute(1, 0, 2).contiguous()
        return (h, c)

    @torch.no_grad()
    def step(self, actor_obs, critic_obs, hidden_state=None):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        actor_hidden, critic_hidden = self.prepare_hidden(hidden_state, actor_tensor.shape[0])

        mean, value, log_std, new_actor_hidden, new_critic_hidden = self.network(
            actor_tensor, critic_tensor, actor_hidden, critic_hidden
        )
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        new_hidden_state = (
            self._detach_hidden(new_actor_hidden),
            self._detach_hidden(new_critic_hidden)
        )
        return action[0].cpu().numpy().astype(np.float32), pre_tanh[0].cpu().numpy().astype(np.float32), \
            new_hidden_state, float(torch.nan_to_num(value).squeeze().cpu().numpy()), float(log_prob.item())

    @torch.no_grad()
    def evaluate(self, actor_obs, critic_obs, hidden_state=None, greedy=True):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        actor_hidden, critic_hidden = self.prepare_hidden(hidden_state, actor_tensor.shape[0])

        mean, value, log_std, new_actor_hidden, new_critic_hidden = self.network(
            actor_tensor, critic_tensor, actor_hidden, critic_hidden
        )
        pre_tanh = mean if greedy else mean + torch.exp(log_std) * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        new_hidden_state = (
            self._detach_hidden(new_actor_hidden),
            self._detach_hidden(new_critic_hidden)
        )
        return action[0].cpu().numpy().astype(np.float32), pre_tanh[0].cpu().numpy().astype(np.float32), \
            new_hidden_state, float(torch.nan_to_num(value).squeeze().cpu().numpy()), float(log_prob.item())

    def train(self, actor_obs, critic_obs, returns, values, actions, old_log_probs,
              actor_hidden_states=None, critic_hidden_states=None, mask=None, episode_starts=None,
              train_valid=None, blocking=None, message=None):
        if self.net_optimizer is None or self.net_scaler is None:
            raise RuntimeError("Global model required for training.")

        self.net_optimizer.zero_grad(set_to_none=True)

        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)

        if actor_obs.dim() != 3:
            actor_obs = actor_obs.unsqueeze(0)
            critic_obs = critic_obs.unsqueeze(0)
            returns = returns.unsqueeze(0)
        if values.dim() != 3:
            values = values.unsqueeze(0)
        if actions.dim() != 2:
            actions = actions.unsqueeze(0)
        if old_log_probs.dim() != 2:
            old_log_probs = old_log_probs.unsqueeze(0)

        batch_size, seq_len, _ = actor_obs.shape

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)

        if episode_starts is None:
            episode_starts = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        else:
            episode_starts = torch.as_tensor(episode_starts, dtype=torch.bool, device=self.device)

        returns = returns.reshape(batch_size, seq_len)
        values = values.reshape(batch_size, seq_len)
        actions = actions.reshape(batch_size, seq_len, actions.shape[-1])
        old_log_probs = old_log_probs.reshape(batch_size, seq_len)
        mask = mask.reshape(batch_size, seq_len)
        episode_starts = episode_starts.reshape(batch_size, seq_len)

        valid_steps = mask.sum().clamp_min(1.0)

        actor_hidden = self._format_hidden_batch(actor_hidden_states, batch_size)
        critic_hidden = self._format_hidden_batch(critic_hidden_states, batch_size)

        with autocast():
            actor_h, actor_c = actor_hidden
            critic_h, critic_c = critic_hidden
            mean_outputs = []
            logstd_outputs = []
            value_outputs = []
            for t in range(seq_len):
                reset_mask = episode_starts[:, t]
                if reset_mask.any():
                    idx = reset_mask.nonzero(as_tuple=True)[0]
                    actor_h[:, idx, :] = 0.0
                    actor_c[:, idx, :] = 0.0
                    critic_h[:, idx, :] = 0.0
                    critic_c[:, idx, :] = 0.0

                inp_actor = actor_obs[:, t, :]
                inp_critic = critic_obs[:, t, :]

                mean, value, log_std, new_actor_hidden, new_critic_hidden = self.network(
                    inp_actor, inp_critic, (actor_h, actor_c), (critic_h, critic_c)
                )
                actor_h, actor_c = new_actor_hidden
                critic_h, critic_c = new_critic_hidden
                mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
                value = torch.nan_to_num(value.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
                mean_outputs.append(mean)
                logstd_outputs.append(log_std)
                value_outputs.append(value)
            policy_mean = torch.stack(mean_outputs, dim=1)
            policy_log_std = torch.stack(logstd_outputs, dim=1)
            new_values = torch.stack(value_outputs, dim=1)
            raw_advantages = returns - values
            raw_advantages = torch.nan_to_num(raw_advantages, nan=0.0, posinf=0.0, neginf=0.0)
            valid_mask = mask > 0.0
            raw_adv_valid = raw_advantages[valid_mask]
            raw_adv_mean = raw_adv_valid.mean() if raw_adv_valid.numel() > 0 else torch.tensor(0.0, device=self.device)
            if raw_adv_valid.numel() > 1:
                raw_adv_std = raw_adv_valid.std(unbiased=False)
            else:
                raw_adv_std = torch.tensor(0.0, device=self.device)
            advantages = raw_advantages.clone()
            if raw_adv_valid.numel() > 1 and raw_adv_std > 1e-6:
                advantages[valid_mask] = (raw_advantages[valid_mask] - raw_adv_mean) / (raw_adv_std + 1e-8)
            else:
                advantages[valid_mask] = raw_advantages[valid_mask] - raw_adv_mean
            advantages = advantages * mask

            new_action_log_probs = self._log_prob_from_pre_tanh(actions, policy_mean, policy_log_std)
            ratio = torch.exp(new_action_log_probs - old_log_probs)
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
            ratio_raw = ratio.clone()
            ratio = ratio.clamp(0.0, getattr(TrainingParameters, 'RATIO_CLAMP_MAX', 4.0))

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                 1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss = -torch.sum(torch.min(surr1, surr2) * mask) / valid_steps

            entropy_terms = (0.5 * (1.0 + np.log(2 * np.pi)) + policy_log_std).sum(dim=-1)
            entropy = torch.sum(entropy_terms * mask) / valid_steps
            value_clipped = values + torch.clamp(new_values - values,
                                                  -TrainingParameters.VALUE_CLIP_RANGE,
                                                  TrainingParameters.VALUE_CLIP_RANGE)
            value_losses = (returns - new_values) ** 2
            value_losses_clipped = (returns - value_clipped) ** 2
            value_loss = torch.sum(torch.max(value_losses, value_losses_clipped) * mask) / valid_steps
            value_clip_fraction = torch.sum((value_losses_clipped > value_losses).float() * mask) / valid_steps

            total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss - TrainingParameters.ENTROPY_COEF * entropy

            approx_kl = torch.sum((old_log_probs - new_action_log_probs) * mask) / valid_steps
        approx_kl_value = torch.nan_to_num(approx_kl, nan=0.0, posinf=0.0, neginf=0.0)
        value_clip_fraction = torch.nan_to_num(value_clip_fraction, nan=0.0, posinf=0.0, neginf=0.0)
        clipfrac = torch.sum((torch.abs(ratio_raw - 1.0) > TrainingParameters.CLIP_RANGE).float() * mask) / valid_steps

        self.net_scaler.scale(total_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        return [float(total_loss.detach().cpu().numpy()),
                float(policy_loss.detach().cpu().numpy()),
                float(entropy.detach().cpu().numpy()),
                float(value_loss.detach().cpu().numpy()),
                float(raw_adv_std.detach().cpu().numpy()),
                float(approx_kl_value.detach().cpu().numpy()),
                float(value_clip_fraction.detach().cpu().numpy()),
                float(torch.nan_to_num(clipfrac).detach().cpu().numpy()),
                float(torch.nan_to_num(grad_norm).detach().cpu().numpy()),
                float(raw_adv_mean.detach().cpu().numpy())]
    def imitation_train(self, actor_obs, critic_obs, optimal_actions):
        if self.net_optimizer is None or self.net_scaler is None:
            raise RuntimeError("Global model required for imitation training.")

        self.net_optimizer.zero_grad(set_to_none=True)

        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.float32, device=self.device)

        actor_obs = torch.nan_to_num(actor_obs)
        critic_obs = torch.nan_to_num(critic_obs)

        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
            critic_obs = critic_obs.unsqueeze(0)
        if optimal_actions.dim() == 0:
            optimal_actions = optimal_actions.unsqueeze(0)

        with autocast():
            mean, _, log_std, _, _ = self.network(actor_obs, critic_obs, None, None)
            mean = torch.nan_to_num(mean)
            log_std = log_std.expand_as(mean)
            log_prob = self._log_prob_from_pre_tanh(optimal_actions, mean, log_std)
            imitation_loss = -log_prob.mean()

        if not torch.isfinite(imitation_loss):
            return [float('nan'), float('nan')]

        self.net_scaler.scale(imitation_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        return [float(imitation_loss.detach().cpu().numpy()),
                float(torch.nan_to_num(grad_norm).detach().cpu().numpy())]
    @torch.no_grad()
    def set_weights(self, weights):
        if weights is None:
            return
        self.network.load_state_dict(weights, strict=True)

    @torch.no_grad()
    def get_weights(self):
        return {k: v.cpu() for k, v in self.network.state_dict().items()}

    @torch.no_grad()
    def value(self, critic_obs, hidden_state=None):
        critic_tensor = self._to_tensor(critic_obs)
        actor_tensor = torch.zeros(critic_tensor.shape[0], NetParameters.ACTOR_VECTOR_LEN, device=self.device)

        actor_hidden, critic_hidden = self.prepare_hidden(hidden_state, critic_tensor.shape[0])
        _, value, _, _, _ = self.network(actor_tensor, critic_tensor, actor_hidden, critic_hidden)
        value = torch.nan_to_num(value)
        return float(value.squeeze().cpu().numpy())

    @torch.no_grad()
    def reset_hidden_state(self, batch_size=1):
        actor_hidden = self.network.init_hidden(batch_size, self.device)
        critic_hidden = self.network.init_hidden(batch_size, self.device)
        return (self._detach_hidden(actor_hidden), self._detach_hidden(critic_hidden))

    def update_learning_rate(self, new_lr):
        if self.net_optimizer is None:
            return
        for group in self.net_optimizer.param_groups:
            group['lr'] = new_lr
        self.current_lr = new_lr