import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib
from targetmaker.target_alg_parameters import TargetTrainingParameters, TargetNetParameters
from targetmaker.target_nets import TargetPPOActorCritic

class TargetPPO(object):
    @staticmethod
    def _angle_limit() -> float:
        try:
            map_cfg = importlib.import_module('map_config')
            limit = float(getattr(map_cfg, 'target_max_angular_speed', 12.0))
        except Exception:
            limit = 12.0
        return max(1.0, limit)
        
    @staticmethod
    def _normalize_action(raw_action):
        limit = TargetPPO._angle_limit()
        return np.clip(raw_action / limit, -1.0, 1.0) # Simplify: Assume raw output is already normalized or mapped? PPO output is usually raw, treated as normalized.

    def __init__(self, device):
        self.device = device
        self.net = TargetPPOActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=TargetTrainingParameters.LEARNING_RATE)
        
    def get_weights(self):
        return self.net.state_dict()

    def set_weights(self, weights):
        self.net.load_state_dict(weights)

    def step(self, actor_obs, critic_obs):
        # Single step for rollout
        with torch.no_grad():
            t_ac = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            t_cr = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            mean, std, value = self.net(t_ac, t_cr)
            
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Tanh Squash usually done in Env wrapper or just output raw. 
            # Tracker PPO outputs raw Gaussian and clips/scales in Env.
            # We will perform Tanh here if we want bounded action [-1, 1], but PPO standard is Gaussian.
            # Let's apply Tanh for consistency with previous Target behavior, OR just output raw and let Env clip.
            # SAC used Tanh. PPO usually doesn't, it just clips.
            # Let's use Tanh to strictly bound to [-1, 1] which matches normalizing logic.
            # WAIT: PPO Gaussian distribution is over (-inf, inf).
            # If we apply Tanh, we must account for it in log_prob (Jacobian). 
            # Tracker code (mlp/model_mlp.py) uses Normal(mean, std) and then clips. It does NOT use Tanh.
            # Let's stick to simple Normal and clip in Runner/Env.
            
            return action.cpu().numpy()[0], value.item(), log_prob.item()

    def evaluate(self, actor_obs):
        with torch.no_grad():
            t_ac = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, _, _ = self.net(t_ac, None)
            return mean.cpu().numpy()[0]

    def train(self, data):
        # data: dict of numpy arrays (flattened)
        obs = torch.as_tensor(data['actor_obs'], dtype=torch.float32, device=self.device)
        c_obs = torch.as_tensor(data['critic_obs'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(data['actions'], dtype=torch.float32, device=self.device)
        old_logpac = torch.as_tensor(data['logp'], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(data['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(data['adv'], dtype=torch.float32, device=self.device)
        
        # Normalize Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = obs.shape[0]
        batch_size = TargetTrainingParameters.BATCH_SIZE
        
        metrics = {'pg_loss': [], 'vf_loss': [], 'entropy': [], 'approx_kl': []}
        
        for _ in range(TargetTrainingParameters.EPOCHS):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                b_obs = obs[idx]
                b_c_obs = c_obs[idx]
                b_act = actions[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]
                b_old_log = old_logpac[idx]
                
                mean, std, value = self.net(b_obs, b_c_obs)
                dist = torch.distributions.Normal(mean, std)
                new_log_prob = dist.log_prob(b_act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Ratio
                ratio = torch.exp(new_log_prob - b_old_log)
                
                # Surreal Loss / Clip Loss
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - TargetTrainingParameters.CLIP_RANGE, 1.0 + TargetTrainingParameters.CLIP_RANGE) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                # Clip value?
                # v_pred_clipped = b_old_val + torch.clamp(value.squeeze(-1) - b_old_val, -clip, clip)
                # For simplicity, standard MSE
                vf_loss = F.mse_loss(value.squeeze(-1), b_ret)
                
                loss = pg_loss + TargetTrainingParameters.VF_COEF * vf_loss - TargetTrainingParameters.ENT_COEF * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), TargetTrainingParameters.MAX_GRAD_NORM)
                self.optimizer.step()
                
                with torch.no_grad():
                     approx_kl = (b_old_log - new_log_prob).mean().item()
                
                metrics['pg_loss'].append(pg_loss.item())
                metrics['vf_loss'].append(vf_loss.item())
                metrics['entropy'].append(entropy.item())
                metrics['approx_kl'].append(approx_kl)
                
        return {k: np.mean(v) for k, v in metrics.items()}
