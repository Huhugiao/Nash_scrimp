import numpy as np
import torch
import ray

from mlp.alg_parameters_mlp import SetupParameters, NetParameters
from mlp.model_mlp import Model as BaseModel  # 仅用于归一化/对手
from mlp.util_mlp import set_global_seeds
from mlp.policymanager_mlp import PolicyManager

from residual.alg_parameters_residual import (
    ResidualTrainingParameters as TP,
    ResidualNetParameters,
    ResidualRLParameters,
)
from residual.nets_residual import ResidualPolicyNetwork
from cbf_controller import CBFTracker
from env import TrackingEnv


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TP.N_ENVS + 1), 1))
class ResidualRunner(object):
    '''
    Residual RL 采样器：
    1) CBF 基础动作（安全、无梯度）
    2) 残差动作（可训练，tanh 限幅后乘 scale）
    3) 融合：final = clip(base + residual, [-1, 1])，避免超出物理/动作空间
    '''

    def __init__(self, env_id):
        self.ID = env_id
        set_global_seeds(env_id * 123 + 1000)
        # 禁用环境 Safety Layer，因为 CBF 基础策略已提供安全保障
        self.env = TrackingEnv(safety_layer_enabled=TP.SAFETY_LAYER_ENABLED)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')

        # --- Base policy: CBF (frozen, 无参数优化) ---
        self.cbf_base = CBFTracker(env=self.env)

        # --- Residual policy (trainable) ---
        self.residual_model = ResidualPolicyNetwork().to(self.local_device)

        # 对手（可选）
        self.opponent_model = BaseModel(self.local_device) if TP.OPPONENT_TYPE == "policy" else None
        self.policy_manager = PolicyManager() if TP.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
        self.current_opponent_policy = None
        self.current_opponent_id = -1

        self.reset_env()

    def reset_env(self):
        obs_tuple = self.env.reset()
        if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
            try:
                self.tracker_obs, self.target_obs = obs_tuple[0]
            except Exception:
                self.tracker_obs = obs_tuple[0]
                self.target_obs = obs_tuple[0]
        else:
            self.tracker_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            self.target_obs = self.tracker_obs

        self.tracker_obs = np.asarray(self.tracker_obs, dtype=np.float32)
        self.target_obs = np.asarray(self.target_obs, dtype=np.float32)

    def _get_opponent_action(self, target_obs, tracker_obs):
        if TP.OPPONENT_TYPE == "policy":
            critic_obs = np.concatenate([target_obs, tracker_obs])
            opp_action, _, _, _ = self.opponent_model.evaluate(target_obs, critic_obs, greedy=True)
            return opp_action
        if TP.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            if self.policy_manager and self.current_opponent_policy:
                return self.policy_manager.get_action(self.current_opponent_policy, target_obs)
            return np.zeros(2, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def run(self, residual_weights, opponent_weights, total_steps, policy_manager_state=None):
        with torch.no_grad():
            # 同步权重
            self.residual_model.load_state_dict(residual_weights)
            if opponent_weights and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)

            if self.policy_manager and policy_manager_state:
                for name, history in policy_manager_state.items():
                    if name in self.policy_manager.win_history:
                        self.policy_manager.win_history[name] = list(history)

            n_steps = TP.N_STEPS

            data = {
                'actor_obs': np.zeros((n_steps, NetParameters.ACTOR_RAW_LEN), dtype=np.float32),
                'critic_obs': np.zeros((n_steps, NetParameters.CRITIC_RAW_LEN), dtype=np.float32),
                'rewards': np.zeros(n_steps, dtype=np.float32),
                'values': np.zeros(n_steps, dtype=np.float32),
                'actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32),  # residual pre-tanh
                'logp': np.zeros(n_steps, dtype=np.float32),
                'dones': np.zeros(n_steps, dtype=np.bool_),
                'episode_starts': np.zeros(n_steps, dtype=np.bool_),
                'base_actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32),
                'episode_success': [],
            }

            performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}

            if self.current_opponent_policy is None and self.policy_manager:
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")

            episode_reward = 0.0
            ep_len = 0
            episodes = 0
            episode_start = True
            current_episode_idx = 0
            completed_opponents = []

            for i in range(n_steps):
                critic_obs_full = np.concatenate([self.tracker_obs, self.target_obs])

                # --- 1. Base action via CBF ---
                # Get privileged state for full CBF-QP control (like evaluate_mlp.py)
                privileged_state = self.env.get_privileged_state() if hasattr(self.env, 'get_privileged_state') else None
                
                # CBF returns [delta_deg, speed_frac] which is exactly what env.step expects
                # Do NOT normalize - env_lib.agent_move uses raw degrees
                cbf_action = self.cbf_base.get_action(self.tracker_obs, privileged_state=privileged_state)
                base_action = np.asarray(cbf_action, dtype=np.float32)

                # --- 2. Residual action (trainable) ---
                obs_tensor = torch.from_numpy(self.tracker_obs).float().to(self.local_device).unsqueeze(0)
                
                # 使用 forward_raw 获取 pre-tanh 均值
                raw_mean, log_std = self.residual_model.actor.forward_raw(obs_tensor)
                
                std = torch.exp(log_std)
                noise = torch.randn_like(raw_mean)
                residual_pre_tanh = raw_mean + std * noise  # pre-tanh sample
                
                # 应用 tanh 和 scale
                residual_scale = self.residual_model.actor.residual_scale
                residual_action = torch.tanh(residual_pre_tanh) * residual_scale

                residual_pre_tanh_np = residual_pre_tanh.cpu().numpy()[0]
                residual_action_np = residual_action.cpu().numpy()[0]

                # log prob (for PPO replay) - 标准 tanh squashed Gaussian
                log_prob = -0.5 * (((residual_pre_tanh - raw_mean) / std).pow(2) + 2 * log_std + np.log(2 * np.pi)).sum(-1)
                log_prob -= (2 * (np.log(2) - residual_pre_tanh - torch.nn.functional.softplus(-2 * residual_pre_tanh))).sum(-1)
                log_prob_np = log_prob.cpu().numpy()[0]

                critic_tensor = torch.from_numpy(critic_obs_full).float().to(self.local_device).unsqueeze(0)
                value_pred = self.residual_model.critic(critic_tensor).item()

                # --- 3. Action fusion ---
                # base_action is [angle_deg, speed_frac] from CBF
                # residual_action is also in same space (scaled by residual_scale)
                # NOTE: residual_scale should be small (e.g., 1.5 for angle, 0.3 for speed)
                # Current residual is tanh * 0.3 which is too small for angle adjustment
                
                # Since CBF outputs raw degrees, we need residual to be in same scale
                # For now, multiply angle residual appropriately
                import map_config
                max_turn = float(getattr(map_config, 'tracker_max_turn_deg', 5.0))
                
                # Scale residual: angle component needs to match degree scale, speed is fine
                residual_scaled = np.array([
                    residual_action_np[0] * max_turn,  # scale [-0.3, 0.3] to [-1.5, 1.5] degrees
                    residual_action_np[1]              # speed residual stays as-is
                ], dtype=np.float32)
                
                fused = base_action + residual_scaled
                
                # Clip to valid physical ranges
                final_action = np.array([
                    np.clip(fused[0], -max_turn, max_turn),  # angle in degrees
                    np.clip(fused[1], 0.0, 1.0)              # speed fraction
                ], dtype=np.float32)

                # --- 4. Env step ---
                target_action = self._get_opponent_action(self.target_obs, self.tracker_obs)
                obs_result, reward, terminated, truncated, info = self.env.step(
                    (final_action, target_action),
                    residual_action=residual_action_np,
                    action_penalty_coef=TP.ACTION_PENALTY_COEF,
                )

                done = terminated or truncated

                data['actor_obs'][i] = self.tracker_obs
                data['critic_obs'][i] = critic_obs_full
                data['values'][i] = value_pred
                data['actions'][i] = residual_pre_tanh_np
                data['logp'][i] = log_prob_np
                data['rewards'][i] = reward
                data['dones'][i] = done
                data['episode_starts'][i] = episode_start
                data['base_actions'][i] = base_action

                episode_start = False
                episode_reward += float(reward)
                ep_len += 1

                if isinstance(obs_result, tuple) and len(obs_result) == 2:
                    self.tracker_obs, self.target_obs = obs_result
                else:
                    self.tracker_obs = obs_result
                    self.target_obs = obs_result

                if TP.OBS_NOISE_STD > 0:
                    self.tracker_obs = self.tracker_obs + np.random.randn(*self.tracker_obs.shape).astype(np.float32) * TP.OBS_NOISE_STD
                    self.tracker_obs = np.clip(self.tracker_obs, -1.0, 1.0)

                if done:
                    win = 1 if info.get('reason') == 'tracker_caught_target' else 0
                    performance_dict['per_r'].append(episode_reward)
                    performance_dict['per_episode_len'].append(ep_len)
                    performance_dict['win'].append(win)

                    completed_opponents.append(self.current_opponent_policy)

                    if self.policy_manager:
                        self.policy_manager.update_win_rate(self.current_opponent_policy, win)
                        self.policy_manager.reset()
                        self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")

                    data['episode_success'].append({
                        'start_idx': current_episode_idx,
                        'end_idx': i + 1,
                        'success': bool(win),
                        'reward': episode_reward,
                        'length': ep_len,
                        'opponent': completed_opponents[-1],
                    })

                    episodes += 1
                    episode_reward = 0.0
                    ep_len = 0
                    episode_start = True
                    current_episode_idx = i + 1
                    self.reset_env()

            critic_obs_last = np.concatenate([self.tracker_obs, self.target_obs])
            critic_tensor_last = torch.from_numpy(critic_obs_last).float().to(self.local_device).unsqueeze(0)
            last_value = self.residual_model.critic(critic_tensor_last).item()

            advantages = np.zeros_like(data['rewards'])
            lastgaelam = 0.0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_non_terminal = 1.0 - data['dones'][t]
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - data['dones'][t]  # 使用当前步的 done 状态
                    next_value = data['values'][t + 1]
                delta = data['rewards'][t] + TP.GAMMA * next_value * next_non_terminal - data['values'][t]
                lastgaelam = delta + TP.GAMMA * TP.LAM * next_non_terminal * lastgaelam
                advantages[t] = lastgaelam
            data['returns'] = advantages + data['values']

            pm_state = {k: list(v) for k, v in self.policy_manager.win_history.items()} if self.policy_manager else None
            return {
                'data': data,
                'performance': performance_dict,
                'episodes': episodes,
                'policy_manager_state': pm_state,
                'completed_opponents': completed_opponents,
            }
