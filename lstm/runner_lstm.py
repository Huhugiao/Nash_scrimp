import numpy as np
import torch
import ray

from map_config import EnvParameters
from lstm.alg_parameters import *  # 修改导入
from lstm.model_lstm import Model
from util import set_global_seeds, update_perf, get_opponent_id_one_hot
from env import TrackingEnv
from rule_policies import TRACKER_POLICY_REGISTRY, TARGET_POLICY_REGISTRY
from policymanager import PolicyManager


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TrainingParameters.N_ENVS + 1), 1))
class Runner(object):
    """Runner for training tracker with LSTM-based Asymmetric Actor-Critic."""

    def __init__(self, env_id):
        self.ID = env_id
        set_global_seeds(env_id * 123)

        # 固定为tracker训练
        self.env = TrackingEnv()
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')

        self.agent_model = Model(self.local_device)
        self.opponent_model = Model(self.local_device) if TrainingParameters.OPPONENT_TYPE == "policy" else None

        self.policy_manager = None
        self.current_opponent_policy = None
        self.current_opponent_id = -1
        
        if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            self.policy_manager = PolicyManager()

        # 初始化观测
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

        self.agent_hidden = None
        self.opponent_hidden = None

    def _get_opponent_action(self, target_obs, privileged_state=None):
        """获取对手(target)动作"""
        if TrainingParameters.OPPONENT_TYPE == "policy":
            dummy_context = np.zeros(NetParameters.CONTEXT_LEN, dtype=np.float32)
            critic_obs = np.concatenate([target_obs, dummy_context])
            
            opp_action, _, new_hidden, _, _ = self.opponent_model.evaluate(
                target_obs, critic_obs, self.opponent_hidden, greedy=True
            )
            self.opponent_hidden = new_hidden
            return opp_action
            
        elif TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            if self.policy_manager and self.current_opponent_policy:
                return self.policy_manager.get_action(
                    self.current_opponent_policy, 
                    target_obs, 
                    privileged_state
                )
            return np.zeros(2, dtype=np.float32)
            
        else:
            raise ValueError(f"Unsupported OPPONENT_TYPE: {TrainingParameters.OPPONENT_TYPE}")

    def _compute_gae_returns(self, data, last_value):
        n_steps = data['rewards'].shape[0]
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
        data['advantages'] = advantages
        return data

    def run(self, model_weights, opponent_weights, total_steps, policy_manager_state=None):
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            if opponent_weights and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)
            if self.policy_manager and policy_manager_state:
                for name, history in policy_manager_state.items():
                    if name in self.policy_manager.win_history:
                        self.policy_manager.win_history[name].clear()
                        self.policy_manager.win_history[name].extend(history)

            n_steps = TrainingParameters.N_STEPS
            hidden_size = self.agent_model.lstm_hidden_size
            num_layers = self.agent_model.num_lstm_layers
            data = {
                'actor_obs': np.zeros((n_steps, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32),
                'critic_obs': np.zeros((n_steps, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32),
                'rewards': np.zeros(n_steps, dtype=np.float32),
                'values': np.zeros(n_steps, dtype=np.float32),
                'actions': np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32),
                'logp': np.zeros(n_steps, dtype=np.float32),
                'dones': np.zeros(n_steps, dtype=np.bool_),
                'episode_starts': np.zeros(n_steps, dtype=np.bool_),
                'actor_hidden_h': np.zeros((n_steps, num_layers, hidden_size), dtype=np.float32),
                'actor_hidden_c': np.zeros((n_steps, num_layers, hidden_size), dtype=np.float32),
                'critic_hidden_h': np.zeros((n_steps, num_layers, hidden_size), dtype=np.float32),
                'critic_hidden_c': np.zeros((n_steps, num_layers, hidden_size), dtype=np.float32)
            }
            performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}

            if self.current_opponent_policy is None and self.policy_manager:
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")

            episode_reward = 0.0
            ep_len = 0
            episodes = 0
            episode_start = True

            for i in range(n_steps):
                # Tracker观测(27) + 对手ID context
                critic_obs_full = np.concatenate([self.tracker_obs, get_opponent_id_one_hot(self.current_opponent_id)])

                actor_hidden_pre, critic_hidden_pre = self.agent_model.prepare_hidden(self.agent_hidden, batch_size=1)
                data['actor_hidden_h'][i] = actor_hidden_pre[0].detach().cpu().numpy()[:, 0, :]
                data['actor_hidden_c'][i] = actor_hidden_pre[1].detach().cpu().numpy()[:, 0, :]
                data['critic_hidden_h'][i] = critic_hidden_pre[0].detach().cpu().numpy()[:, 0, :]
                data['critic_hidden_c'][i] = critic_hidden_pre[1].detach().cpu().numpy()[:, 0, :]

                # Agent(tracker)动作
                agent_action, agent_pre_tanh, new_hidden, v_pred, log_prob = self.agent_model.step(
                    self.tracker_obs, critic_obs_full, (actor_hidden_pre, critic_hidden_pre)
                )
                self.agent_hidden = new_hidden

                # 对手(target)动作
                target_action = self._get_opponent_action(self.target_obs, self.tracker_obs)

                obs_result, reward, terminated, truncated, info = self.env.step((agent_action, target_action))
                done = terminated or truncated

                # 解析观测
                if isinstance(obs_result, tuple) and len(obs_result) == 2:
                    self.tracker_obs, self.target_obs = obs_result
                else:
                    self.tracker_obs = obs_result
                    self.target_obs = obs_result

                # 记录数据
                data['actor_obs'][i] = self.tracker_obs
                data['critic_obs'][i] = critic_obs_full
                data['values'][i] = v_pred
                data['actions'][i] = agent_pre_tanh
                data['logp'][i] = log_prob
                data['rewards'][i] = reward
                data['dones'][i] = done
                data['episode_starts'][i] = episode_start

                episode_reward += float(reward)
                ep_len += 1
                episode_start = False

                if done:
                    performance_dict['per_r'].append(episode_reward)
                    performance_dict['per_episode_len'].append(ep_len)
                    win = 1 if info.get('reason') == 'tracker_caught_target' else 0
                    performance_dict['win'].append(win)
                    if self.policy_manager:
                        self.policy_manager.update_win_rate(self.current_opponent_policy, win)
                        self.policy_manager.reset()
                        self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")

                    episode_reward = 0.0
                    ep_len = 0
                    episodes += 1
                    episode_start = True
                    
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
                    self.agent_hidden = None
                    self.opponent_hidden = None

            # 计算最终价值
            critic_obs_last = np.concatenate([self.tracker_obs, get_opponent_id_one_hot(self.current_opponent_id)])
            last_value = self.agent_model.value(critic_obs_last, self.agent_hidden)
            data = self._compute_gae_returns(data, last_value)

            new_pm_state = {k: list(v) for k, v in self.policy_manager.win_history.items()} if self.policy_manager else None
            return data, n_steps, episodes, performance_dict, new_pm_state

    def imitation(self, model_weights, opponent_weights, total_steps):
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            if opponent_weights and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)

            n_steps = TrainingParameters.N_STEPS
            actor_obs_arr = np.zeros((n_steps, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32)
            critic_obs_arr = np.zeros((n_steps, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32)
            actions = np.zeros((n_steps, NetParameters.ACTION_DIM), dtype=np.float32)
            performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}

            obs_tuple = self.env.reset()
            if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
                try:
                    tracker_obs, target_obs = obs_tuple[0]
                except Exception:
                    tracker_obs = obs_tuple[0]
                    target_obs = obs_tuple[0]
            else:
                tracker_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
                target_obs = tracker_obs

            if self.current_opponent_policy is None and self.policy_manager:
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")

            episode_reward = 0.0
            ep_len = 0
            episodes = 0

            for i in range(n_steps):
                # 使用APF专家策略生成tracker动作
                critic_full = np.concatenate([tracker_obs, get_opponent_id_one_hot(self.current_opponent_id)])
                teacher_name = "APF" if "APF" in TRACKER_POLICY_REGISTRY else sorted(TRACKER_POLICY_REGISTRY.keys())[0]
                policy_fn = TRACKER_POLICY_REGISTRY.get(teacher_name)
                if not policy_fn:
                    raise ValueError("No tracker policy found (APF or default).")
                expert_pair = policy_fn(tracker_obs)
                normalized = np.asarray(expert_pair, dtype=np.float32)
                pre_tanh = Model.to_pre_tanh(normalized)
                
                # 对手动作
                target_action = self._get_opponent_action(target_obs, tracker_obs)

                next_obs, reward, terminated, truncated, info = self.env.step((normalized, target_action))
                done = terminated or truncated

                # 记录IL数据
                actor_obs_arr[i] = tracker_obs
                critic_obs_arr[i] = critic_full
                actions[i] = pre_tanh

                episode_reward += float(reward)
                ep_len += 1

                # 解析观测
                if isinstance(next_obs, tuple) and len(next_obs) == 2:
                    try:
                        tracker_obs, target_obs = next_obs
                    except Exception:
                        tracker_obs = next_obs
                        target_obs = next_obs
                else:
                    tracker_obs = next_obs
                    target_obs = next_obs

                if done:
                    performance_dict['per_r'].append(episode_reward)
                    performance_dict['per_episode_len'].append(ep_len)
                    win = 1 if info.get('reason') == 'tracker_caught_target' else 0
                    performance_dict['win'].append(win)
                    if self.policy_manager:
                        self.policy_manager.update_win_rate(self.current_opponent_policy, win)
                        self.policy_manager.reset()
                        self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy("target")
                    obs_tuple = self.env.reset()
                    if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
                        try:
                            tracker_obs, target_obs = obs_tuple[0]
                        except Exception:
                            tracker_obs = obs_tuple[0]
                            target_obs = obs_tuple[0]
                    else:
                        tracker_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
                        target_obs = tracker_obs
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