import numpy as np
import torch
import ray
import map_config

from star_hrl.alg_parameters_star import (
    SetupParameters, TrainingParameters, NetParameters, RecordingParameters
)
from star_hrl.model_star import StarModel
from mlp.util_mlp import set_global_seeds
from env import TrackingEnv
from mlp.policymanager_mlp import PolicyManager

@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TrainingParameters.N_ENVS + 1), 1))
class StarRunner:
    def __init__(self, env_id):
        self.ID = env_id
        map_config.set_obstacle_density(SetupParameters.OBSTACLE_DENSITY)
        set_global_seeds(env_id * 123)
        
        enable_safety = getattr(RecordingParameters, 'ENABLE_SAFETY_LAYER', True)
        self.env = TrackingEnv(enable_safety_layer=enable_safety)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.agent_model = StarModel(self.local_device)
        
        self.policy_manager = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
        self.current_opponent_policy = None
        self.current_opponent_id = -1
        
        self.reset_env()
        
    def reset_env(self):
        obs_tuple = self.env.reset()
        if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
            self.tracker_obs, self.target_obs = obs_tuple[0], obs_tuple[0]
            if isinstance(obs_tuple[0], (tuple, list)) and len(obs_tuple[0]) == 2:
                self.tracker_obs, self.target_obs = obs_tuple[0]
        else:
            self.tracker_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            self.target_obs = self.tracker_obs
        self.tracker_obs = np.asarray(self.tracker_obs, dtype=np.float32)
        self.target_obs = np.asarray(self.target_obs, dtype=np.float32)
        
    def _get_opponent_action(self, target_obs, tracker_obs):
        if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            if self.policy_manager and self.current_opponent_policy:
                return self.policy_manager.get_action(self.current_opponent_policy, target_obs)
            return np.zeros(2, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)
    
    def run(self, model_weights, opponent_weights, total_steps, policy_manager_state=None):
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            
            if self.policy_manager and policy_manager_state:
                for name, history in policy_manager_state.items():
                    if name in self.policy_manager.win_history:
                        self.policy_manager.win_history[name] = list(history)
            
            n_steps = TrainingParameters.N_STEPS
            data = {
                'actor_obs': np.zeros((n_steps, NetParameters.ACTOR_RAW_LEN), dtype=np.float32),
                'critic_obs': np.zeros((n_steps, NetParameters.CRITIC_RAW_LEN), dtype=np.float32),
                'rewards': np.zeros(n_steps, dtype=np.float32),
                'values': np.zeros(n_steps, dtype=np.float32),
                'actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32),
                'logp': np.zeros(n_steps, dtype=np.float32),
                'dones': np.zeros(n_steps, dtype=np.bool_),
                'episode_starts': np.zeros(n_steps, dtype=np.bool_),
                'high_weights': np.zeros((n_steps, 2), dtype=np.float32),
            }
            performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}
            
            if self.current_opponent_policy is None and self.policy_manager:
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")
            
            episode_reward = 0.0
            ep_len = 0
            episodes = 0
            episode_start = True
            completed_opponents = []
            
            for i in range(n_steps):
                critic_obs_full = np.concatenate([self.tracker_obs, self.target_obs])
                
                agent_action, agent_pre_tanh, v_pred, log_prob, high_weights = \
                    self.agent_model.step(self.tracker_obs, critic_obs_full)
                
                target_action = self._get_opponent_action(self.target_obs, self.tracker_obs)
                
                obs_result, reward, terminated, truncated, info = self.env.step((agent_action, target_action))
                done = terminated or truncated
                
                data['actor_obs'][i] = self.tracker_obs
                data['critic_obs'][i] = critic_obs_full
                data['values'][i] = v_pred
                data['actions'][i] = agent_pre_tanh
                data['logp'][i] = log_prob
                data['rewards'][i] = reward
                data['dones'][i] = done
                data['episode_starts'][i] = episode_start
                data['high_weights'][i] = high_weights
                
                episode_start = False
                episode_reward += float(reward)
                ep_len += 1
                
                if isinstance(obs_result, tuple) and len(obs_result) == 2:
                    self.tracker_obs, self.target_obs = obs_result
                else:
                    self.tracker_obs = obs_result
                    self.target_obs = obs_result
                    
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
                    
                    episodes += 1
                    episode_reward = 0.0
                    ep_len = 0
                    episode_start = True
                    self.reset_env()
            
            critic_obs_last = np.concatenate([self.tracker_obs, self.target_obs])
            _, _, last_value, _, _ = self.agent_model.evaluate(self.tracker_obs, critic_obs_last)
            
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
