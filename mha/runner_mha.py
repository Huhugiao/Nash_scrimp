import numpy as np
import torch
import ray
from collections import deque
from mha.alg_parameters_mha import *
from mha.model_mha import Model
from util import set_global_seeds
from env import TrackingEnv
from policymanager import PolicyManager
from rule_policies import CBFTracker


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
        # 用 rule_policies.CBFTracker
        self.cbf_teacher = CBFTracker()
        
        self.tracker_obs = None
        self.target_obs = None
        self.agent_history = None
        self.opponent_history = None
        
        self.reset_env()

    def reset_env(self):
        obs_result, info = self.env.reset()
        self._parse_obs(obs_result)
        self.agent_history = None
        self.opponent_history = None

    def _parse_obs(self, obs_result):
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            self.tracker_obs = np.asarray(obs_result[0], dtype=np.float32)
            self.target_obs = np.asarray(obs_result[1], dtype=np.float32)
        else:
            obs = np.asarray(obs_result, dtype=np.float32)
            self.tracker_obs = obs
            self.target_obs = obs[:NetParameters.CRITIC_VECTOR_LEN]

    def _get_opponent_action(self):
        if TrainingParameters.OPPONENT_TYPE == "policy":
            opp_action, _, new_hist, _, _ = self.opponent_model.evaluate(
                self.target_obs, self.target_obs, self.opponent_history, greedy=True
            )
            self.opponent_history = new_hist
            return opp_action
        elif TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            if self.policy_manager and self.current_opponent_policy:
                return self.policy_manager.get_action(self.current_opponent_policy, self.target_obs)
        return np.zeros(NetParameters.ACTION_DIM, dtype=np.float32)

    def _get_expert_action(self):
        """使用与 evaluate.py 相同接口的 CBFTracker 规则策略"""
        try:
            # 构建与 evaluate 一致的 actor_obs 和 privileged_state
            actor_obs = np.asarray(self.tracker_obs, dtype=np.float32)
            privileged_state = self.env.get_privileged_state() if hasattr(self.env, "get_privileged_state") else None

            # rule_policies.CBFTracker.get_action(actor_obs, privileged_state)
            if hasattr(self.cbf_teacher, "get_action"):
                expert_action = self.cbf_teacher.get_action(actor_obs, privileged_state)
            else:
                expert_action = self.cbf_teacher(actor_obs, privileged_state)

            expert_action = np.asarray(expert_action, dtype=np.float32).reshape(NetParameters.ACTION_DIM)
            # 归一化到 [-1, 1]
            return np.clip(expert_action, -1.0, 1.0).astype(np.float32)
        except Exception:
            return np.zeros(NetParameters.ACTION_DIM, dtype=np.float32)

    def run(self, model_weights, opponent_weights, total_steps, policy_manager_state=None, il_prob=0.0):
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            if opponent_weights and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)
            if self.policy_manager and policy_manager_state:
                for name, history in policy_manager_state.items():
                    if name in self.policy_manager.win_history:
                        self.policy_manager.win_history[name] = deque(
                            history, maxlen=TrainingParameters.ADAPTIVE_SAMPLING_WINDOW
                        )

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
                # expert 只作标签，不控制环境
                'expert_actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32),
                'is_expert_step': np.zeros(n_steps, dtype=np.bool_),
                'episode_indices': np.zeros(n_steps, dtype=np.int32),
                'episode_success': [],
            }
            
            performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}

            if self.current_opponent_policy is None and self.policy_manager:
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")

            episode_reward = 0.0
            ep_len = 0
            episodes = 0
            episode_start = True
            current_episode_start_idx = 0

            for i in range(n_steps):
                data['episode_indices'][i] = episodes
                critic_obs = self.target_obs
                
                # 1) agent 决策（始终用于控制环境）
                agent_action, agent_pre_tanh, new_hist, v_pred, log_prob = self.agent_model.step(
                    self.tracker_obs, critic_obs, self.agent_history
                )
                self.agent_history = new_hist

                # 2) expert 动作，只作为 IL 标签（DAgger）：是否记录由 il_prob 决定
                record_expert = (np.random.rand() < il_prob)
                if record_expert:
                    expert_action = self._get_expert_action()
                    data['expert_actions'][i] = expert_action
                    data['is_expert_step'][i] = True
                else:
                    data['is_expert_step'][i] = False

                execute_action = agent_action  # 环境永远执行 agent_action

                target_action = self._get_opponent_action()
                
                obs_result, reward, terminated, truncated, info = self.env.step((execute_action, target_action))
                done = terminated or truncated
                
                data['actor_obs'][i] = self.tracker_obs
                data['critic_obs'][i] = critic_obs
                data['values'][i] = v_pred
                data['actions'][i] = agent_pre_tanh
                data['logp'][i] = log_prob
                data['rewards'][i] = reward
                data['dones'][i] = done
                data['episode_starts'][i] = episode_start
                
                episode_start = False
                episode_reward += reward
                ep_len += 1
                
                self._parse_obs(obs_result)
                
                if done:
                    win = info.get('reason') == 'tracker_caught_target' if isinstance(info, dict) else False
                    
                    data['episode_success'].append({
                        'start_idx': current_episode_start_idx,
                        'end_idx': i + 1,
                        'success': win,
                        'reward': episode_reward,
                        'length': ep_len,
                        # DAgger 模式下不再有“整条 expert episode”的概念，这里统一设 False，仅保留字段以兼容
                        'use_expert': False,
                    })
                    
                    performance_dict['per_r'].append(episode_reward)
                    performance_dict['per_episode_len'].append(ep_len)
                    performance_dict['win'].append(1 if win else 0)
                    
                    if self.policy_manager and self.current_opponent_policy:
                        self.policy_manager.update_win_rate(self.current_opponent_policy, win)
                        self.current_opponent_policy, self.current_opponent_id = \
                            self.policy_manager.sample_policy("target")
                    
                    episodes += 1
                    episode_reward = 0.0
                    ep_len = 0
                    episode_start = True
                    current_episode_start_idx = i + 1
                    
                    self.reset_env()
                    if hasattr(self.cbf_teacher, "reset"):
                        self.cbf_teacher.reset()

            # 3) GAE 计算
            last_value = self.agent_model.evaluate(
                self.tracker_obs, self.target_obs, self.agent_history
            )[3]
            
            advantages = np.zeros_like(data['rewards'])
            lastgaelam = 0.0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - float(data['dones'][t])
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - float(data['dones'][t])
                    nextvalues = data['values'][t + 1]
                delta = data['rewards'][t] + TrainingParameters.GAMMA * nextvalues * nextnonterminal - data['values'][t]
                advantages[t] = lastgaelam = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * nextnonterminal * lastgaelam
            
            data['returns'] = advantages + data['values']
            
            pm_state = {k: list(v) for k, v in self.policy_manager.win_history.items()} if self.policy_manager else None
            
            return {
                'data': data,
                'performance': performance_dict,
                'episodes': episodes,
                'policy_manager_state': pm_state
            }
