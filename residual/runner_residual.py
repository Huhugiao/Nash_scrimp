import numpy as np
import torch
import ray
from mlp.alg_parameters_mlp import *
from mlp.model_mlp import Model as BaseModel  # For loading base policy
from mlp.nets_residual import ResidualPolicyNetwork
from mlp.util_mlp import set_global_seeds
from env import TrackingEnv
from mlp.policymanager_mlp import PolicyManager

@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TrainingParameters.N_ENVS + 1), 1))
class ResidualRunner(object):
    """
    Runner for Residual RL Training.
    
    Manages interaction:
    1. Get Base Action (Frozen RL)
    2. Get Residual Action (Trainable)
    3. Fuse: action = clip(base + residual)
    4. Step Environment
    """
    def __init__(self, env_id):
        self.ID = env_id
        set_global_seeds(env_id * 123 + 1000) # Offset seed to avoid exact same eps as base training
        self.env = TrackingEnv()
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        
        # --- Load Frozen Base Model ---
        print(f"ResidualRunner {env_id}: Loading base model from {ResidualRLParameters.BASE_MODEL_PATH}")
        self.base_model = BaseModel(self.local_device)
        checkpoint = torch.load(ResidualRLParameters.BASE_MODEL_PATH, map_location=self.local_device)
        # Handle cases where checkpoint structure might differ
        if 'model_weights' in checkpoint:
            self.base_model.set_weights(checkpoint['model_weights'])
        elif 'model' in checkpoint:
            self.base_model.set_weights(checkpoint['model'])
        else:
            # Assume strict state dict if not wrapped
            self.base_model.network.load_state_dict(checkpoint)
        
        # We don't need optimizer for base model, just inference
        self.base_model.network.eval()
        
        # --- Initialize Residual Model (for local inference/sync) ---
        self.residual_model = ResidualPolicyNetwork().to(self.local_device)
        
        # Opponent handling (standard)
        self.opponent_model = BaseModel(self.local_device) if TrainingParameters.OPPONENT_TYPE == "policy" else None
        self.policy_manager = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
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
        # Same logic as main runner
        if TrainingParameters.OPPONENT_TYPE == "policy":
            critic_obs = np.concatenate([target_obs, tracker_obs])
            opp_action, _, _, _ = self.opponent_model.evaluate(target_obs, critic_obs, greedy=True)
            return opp_action
        elif TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            if self.policy_manager and self.current_opponent_policy:
                return self.policy_manager.get_action(self.current_opponent_policy, target_obs)
            return np.zeros(2, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def run(self, residual_weights, opponent_weights, total_steps, policy_manager_state=None):
        """
        Execute rollout with Residual Policy.
        """
        with torch.no_grad():
            # Sync weights
            self.residual_model.load_state_dict(residual_weights)
            if opponent_weights and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)
            
            if self.policy_manager and policy_manager_state:
                for name, history in policy_manager_state.items():
                    if name in self.policy_manager.win_history:
                        self.policy_manager.win_history[name] = list(history)

            n_steps = TrainingParameters.N_STEPS
            
            # Data buffers
            data = {
                'actor_obs': np.zeros((n_steps, NetParameters.ACTOR_RAW_LEN), dtype=np.float32),
                'critic_obs': np.zeros((n_steps, NetParameters.CRITIC_RAW_LEN), dtype=np.float32),
                'rewards': np.zeros(n_steps, dtype=np.float32),
                'values': np.zeros(n_steps, dtype=np.float32),
                'actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32), # This stores residual action
                'logp': np.zeros(n_steps, dtype=np.float32),
                'dones': np.zeros(n_steps, dtype=np.bool_),
                'episode_starts': np.zeros(n_steps, dtype=np.bool_),
                'base_actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32), # Store base action for analysis
                'episode_success': []
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
                
                # --- 1. Get Base Action (Frozen) ---
                # We use evaluate() which returns deterministic/greedy action usually, but step() gives exploration.
                # Since base is trained and frozen, we likely want its "best" action, so evaluate(greedy=True).
                # But if we want it to react naturally, maybe regular step? Let's use evaluate(greedy=True) for stability.
                base_action, _, _, _ = self.base_model.evaluate(self.tracker_obs, critic_obs_full, greedy=True)
                
                # --- 2. Get Residual Action (Trainable) ---
                # Convert obs to tensor for residual model
                obs_tensor = torch.from_numpy(self.tracker_obs).float().to(self.local_device).unsqueeze(0)
                mean, log_std = self.residual_model.actor(obs_tensor)
                
                # Sample residual action (exploration)
                std = torch.exp(log_std)
                noise = torch.randn_like(mean)
                residual_pre_tanh = mean + std * noise 
                
                residual_action_tanh = torch.tanh(residual_pre_tanh)
                residual_pre_tanh_np = residual_pre_tanh.cpu().numpy()[0]
                residual_action_np = residual_action_tanh.cpu().numpy()[0]
                
                # Calculate log prob (for verification, though PPO recomputes it)
                # Note: We need log_prob of the *residual action* from the *residual distribution*.
                log_prob = -0.5 * (((residual_pre_tanh - mean) / std).pow(2) + 2 * log_std + np.log(2 * np.pi)).sum(-1)
                log_prob -= (2 * (np.log(2) - residual_pre_tanh - torch.nn.functional.softplus(-2 * residual_pre_tanh))).sum(-1)
                log_prob_np = log_prob.cpu().numpy()[0]
                
                # Get value estimate
                critic_tensor = torch.from_numpy(critic_obs_full).float().to(self.local_device).unsqueeze(0)
                value_pred = self.residual_model.critic(critic_tensor).item()
                
                # --- 3. Action Fusion ---
                # Final Action = clip(Base + Residual, -1, 1)
                final_action = np.clip(base_action + residual_action_np, -1.0, 1.0)
                
                # --- 4. Step Environment ---
                target_action = self._get_opponent_action(self.target_obs, self.tracker_obs)
                
                # Update env step call with kwarg (will be implemented in env.py later)
                # IMPORTANT: We rely on env.step accepting residual_action
                obs_result, reward, terminated, truncated, info = self.env.step(
                    (final_action, target_action),
                    residual_action=residual_action_np,
                    action_penalty_coef=ResidualRLParameters.ACTION_PENALTY_COEF
                )
                
                done = terminated or truncated

                # Store data
                data['actor_obs'][i] = self.tracker_obs
                data['critic_obs'][i] = critic_obs_full
                data['values'][i] = value_pred
                data['actions'][i] = residual_pre_tanh_np # Store PRE-TANH for PPO Training!
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

                # Obs Noise
                if TrainingParameters.OBS_NOISE_STD > 0:
                    self.tracker_obs = self.tracker_obs + np.random.randn(*self.tracker_obs.shape).astype(np.float32) * TrainingParameters.OBS_NOISE_STD
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
                        'opponent': completed_opponents[-1]
                    })
                    
                    episodes += 1
                    episode_reward = 0.0
                    ep_len = 0
                    episode_start = True
                    current_episode_idx = i + 1
                    self.reset_env()

            # GAE Calculation (Standard)
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
                    next_non_terminal = 1.0 - data['dones'][t + 1]
                    next_value = data['values'][t + 1]
                delta = data['rewards'][t] + TrainingParameters.GAMMA * next_value * next_non_terminal - data['values'][t]
                lastgaelam = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * next_non_terminal * lastgaelam
                advantages[t] = lastgaelam
            data['returns'] = advantages + data['values']

            pm_state = {k: list(v) for k, v in self.policy_manager.win_history.items()} if self.policy_manager else None
            return {
                'data': data,
                'performance': performance_dict,
                'episodes': episodes,
                'policy_manager_state': pm_state,
                'completed_opponents': completed_opponents
            }
