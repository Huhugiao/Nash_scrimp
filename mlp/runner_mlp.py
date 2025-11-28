import numpy as np
import torch
import ray
from mlp.alg_parameters_mlp import *
from mlp.model_mlp import Model
from util import set_global_seeds, get_opponent_id_one_hot
from env import TrackingEnv
from policymanager import PolicyManager
from cbf_controller import CBFTracker

@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TrainingParameters.N_ENVS + 1), 1))
class Runner(object):
    def __init__(self, env_id):
        self.ID = env_id
        set_global_seeds(env_id * 123)
        self.env = TrackingEnv()
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        
        self.agent_model = Model(self.local_device)
        self.opponent_model = Model(self.local_device) if TrainingParameters.OPPONENT_TYPE == "policy" else None
        
        self.policy_manager = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
        self.current_opponent_policy = None
        self.current_opponent_id = -1
        self.cbf_teacher = CBFTracker()
        
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
        # ensure arrays are numpy float32
        self.tracker_obs = np.asarray(self.tracker_obs, dtype=np.float32)
        self.target_obs = np.asarray(self.target_obs, dtype=np.float32)

    def _get_opponent_action(self, target_obs, tracker_obs):
        if TrainingParameters.OPPONENT_TYPE == "policy":
            dummy_context = np.zeros(NetParameters.CONTEXT_LEN, dtype=np.float32)
            critic_obs = np.concatenate([target_obs, dummy_context])
            opp_action, _, _, _ = self.opponent_model.evaluate(target_obs, critic_obs, greedy=True)
            return opp_action
        elif TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            if self.policy_manager and self.current_opponent_policy:
                return self.policy_manager.get_action(self.current_opponent_policy, target_obs)
            return np.zeros(2, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def run(self, model_weights, opponent_weights, total_steps, policy_manager_state=None, il_prob=0.0):
        """
        Match MHA Runner output structure and collect expert labels (DAgger-style).
        Returns dict with 'data' and performance metadata.
        """
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            if opponent_weights and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)
            if self.policy_manager and policy_manager_state:
                for name, history in policy_manager_state.items():
                    if name in self.policy_manager.win_history:
                        self.policy_manager.win_history[name] = list(history)

            n_steps = TrainingParameters.N_STEPS
            data = {
                'actor_obs': np.zeros((n_steps, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32),
                'critic_obs': np.zeros((n_steps, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32),
                'rewards': np.zeros(n_steps, dtype=np.float32),
                'values': np.zeros(n_steps, dtype=np.float32),
                'actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32),
                'logp': np.zeros(n_steps, dtype=np.float32),
                'dones': np.zeros(n_steps, dtype=np.bool_),
                'episode_starts': np.zeros(n_steps, dtype=np.bool_),
                # IL 标签相关
                'expert_actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32),
                'is_expert_step': np.zeros(n_steps, dtype=np.bool_),
                'episode_indices': np.zeros(n_steps, dtype=np.int32),
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

            for i in range(n_steps):
                data['episode_indices'][i] = episodes
                critic_obs_full = np.concatenate([self.tracker_obs, get_opponent_id_one_hot(self.current_opponent_id)])
                
                agent_action, agent_pre_tanh, v_pred, log_prob = self.agent_model.step(self.tracker_obs, critic_obs_full)
                
                # DAgger-style: decide whether to record expert label
                record_expert = (np.random.rand() < il_prob)
                if record_expert:
                    privileged = self.env.get_privileged_state() if hasattr(self.env, "get_privileged_state") else None
                    expert_pair = self.cbf_teacher.get_action(self.tracker_obs, privileged_state=privileged)
                    normalized = np.asarray(expert_pair, dtype=np.float32)
                    pre_tanh = Model.to_pre_tanh(normalized)
                    data['expert_actions'][i] = normalized
                    data['is_expert_step'][i] = True
                else:
                    data['is_expert_step'][i] = False

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
                        'use_expert': False
                    })
                    
                    episodes += 1
                    episode_reward = 0.0
                    ep_len = 0
                    episode_start = True
                    current_episode_idx = i + 1
                    self.reset_env()

            # Last value for GAE
            critic_obs_last = np.concatenate([self.tracker_obs, get_opponent_id_one_hot(self.current_opponent_id)])
            last_value = self.agent_model.evaluate(self.tracker_obs, critic_obs_last)[2]
            
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
            return {'data': data, 'performance': performance_dict, 'episodes': episodes, 'policy_manager_state': pm_state}

    def imitation(self, model_weights, opponent_weights, total_steps):
        # keep imitation simpler: return arrays for IL pretraining
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            if opponent_weights and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)
            n_steps = TrainingParameters.N_STEPS
            actor_obs_arr = np.zeros((n_steps, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32)
            critic_obs_arr = np.zeros((n_steps, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32)
            actions = np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32)
            performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}
            if self.current_opponent_policy is None and self.policy_manager:
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")
            self.reset_env()
            if hasattr(self.cbf_teacher, "reset"): self.cbf_teacher.reset()
            episodes = 0
            episode_reward = 0.0
            ep_len = 0
            for i in range(n_steps):
                critic_full = np.concatenate([self.tracker_obs, get_opponent_id_one_hot(self.current_opponent_id)])
                privileged = self.env.get_privileged_state() if hasattr(self.env, "get_privileged_state") else None
                expert_pair = self.cbf_teacher.get_action(self.tracker_obs, privileged_state=privileged)
                normalized = np.asarray(expert_pair, dtype=np.float32)
                pre_tanh = Model.to_pre_tanh(normalized)
                target_action = self._get_opponent_action(self.target_obs, self.tracker_obs)
                next_obs, reward, terminated, truncated, info = self.env.step((normalized, target_action))
                done = terminated or truncated
                actor_obs_arr[i] = self.tracker_obs
                critic_obs_arr[i] = critic_full
                actions[i] = pre_tanh
                episode_reward += float(reward)
                ep_len += 1
                if isinstance(next_obs, tuple) and len(next_obs) == 2:
                    self.tracker_obs, self.target_obs = next_obs
                else:
                    self.tracker_obs = next_obs
                    self.target_obs = next_obs
                if done:
                    performance_dict['per_r'].append(episode_reward)
                    performance_dict['per_episode_len'].append(ep_len)
                    win = 1 if info.get('reason') == 'tracker_caught_target' else 0
                    performance_dict['win'].append(win)
                    if self.policy_manager:
                        self.policy_manager.update_win_rate(self.current_opponent_policy, win)
                        self.policy_manager.reset()
                        self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")
                    self.reset_env()
                    if hasattr(self.cbf_teacher, "reset"): self.cbf_teacher.reset()
                    episode_reward = 0.0
                    ep_len = 0
                    episodes += 1
            return {
                'actor_obs': actor_obs_arr,
                'critic_obs': critic_obs_arr,
                'actions': actions,
                'performance': performance_dict,
                'episodes': episodes
            }
