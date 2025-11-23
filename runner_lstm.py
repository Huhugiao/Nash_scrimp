import numpy as np
import torch
import ray

from alg_parameters import *
from model_lstm import Model
from util import set_global_seeds, update_perf, get_opponent_id_one_hot
from env import TrackingEnv
from rule_policies import TRACKER_POLICY_REGISTRY, TARGET_POLICY_REGISTRY
from policymanager import PolicyManager


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TrainingParameters.N_ENVS + 1), 1))
class Runner(object):
    """Runner with LSTM-based Asymmetric Actor-Critic observations and adaptive opponent sampling."""

    def __init__(self, env_id, mission):
        self.ID = env_id
        self.mission = mission
        set_global_seeds(env_id * 123)

        self.env = TrackingEnv(mission=mission)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')

        self.agent_model = Model(self.local_device)
        self.opponent_model = Model(self.local_device) if TrainingParameters.OPPONENT_TYPE == "policy" else None

        self.policy_manager = None
        self.current_opponent_policy = None
        self.current_opponent_id = -1
        self.opponent_role = "target" if self.mission == 0 else "tracker"
        if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            self.policy_manager = PolicyManager()

        # Maintain role-specific observations
        self.tracker_obs, self.target_obs = None, None
        obs_tuple = self.env.reset()
        if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
            # env.reset() -> ( (tracker_obs, target_obs), info )
            try:
                self.tracker_obs, self.target_obs = obs_tuple[0]
            except Exception:
                # fallback: duplicate
                self.tracker_obs = obs_tuple[0]
                self.target_obs = obs_tuple[0]
        else:
            # 兼容单一观测
            base_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            self.tracker_obs = base_obs
            self.target_obs = base_obs

        self.agent_hidden = None
        self.opponent_hidden = None

    def _get_opponent_action(self, actor_obs, critic_obs):
        if TrainingParameters.OPPONENT_TYPE == "policy":
            opp_action, _, new_hidden, _, _ = self.opponent_model.evaluate(
                actor_obs, critic_obs, self.opponent_hidden, greedy=True
            )
            self.opponent_hidden = new_hidden
            return opp_action
        elif TrainingParameters.OPPONENT_TYPE == "expert":
            # 使用新的策略接口
            if self.mission == 0:  # tracker训练，对手是target
                default_target = sorted(TARGET_POLICY_REGISTRY.keys())[0]
                policy_cls = TARGET_POLICY_REGISTRY.get(default_target)
                if policy_cls:
                    policy_obj = policy_cls()
                    return policy_obj.get_action(actor_obs)
            else:  # target训练，对手是tracker
                default_tracker = sorted(TRACKER_POLICY_REGISTRY.keys())[0]
                policy_fn = TRACKER_POLICY_REGISTRY.get(default_tracker)
                if policy_fn:
                    return policy_fn(actor_obs)
            raise ValueError("No expert policy found")
        elif TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
            return self.policy_manager.get_action(self.current_opponent_policy, actor_obs)
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
                'actions': np.zeros((n_steps, getattr(NetParameters, 'ACTION_DIM', 2)), dtype=np.float32),
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
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy(self.opponent_role)

            episode_reward = 0.0
            ep_len = 0
            episodes = 0
            episode_start = True

            for i in range(n_steps):
                # === 根据训练角色选择正确的观测 ===
                if self.mission == 0:  # 训练tracker
                    agent_actor_obs = self.tracker_obs  # tracker观测(27)
                    opp_actor_obs = self.target_obs      # target观测(24)
                else:  # 训练target（不建议在当前观测维度配置下）
                    agent_actor_obs = self.target_obs
                    opp_actor_obs = self.tracker_obs

                critic_obs_full = np.concatenate([agent_actor_obs, get_opponent_id_one_hot(self.current_opponent_id)])

                actor_hidden_pre, critic_hidden_pre = self.agent_model.prepare_hidden(self.agent_hidden, batch_size=1)
                data['actor_hidden_h'][i] = actor_hidden_pre[0].detach().cpu().numpy()[:, 0, :]
                data['actor_hidden_c'][i] = actor_hidden_pre[1].detach().cpu().numpy()[:, 0, :]
                data['critic_hidden_h'][i] = critic_hidden_pre[0].detach().cpu().numpy()[:, 0, :]
                data['critic_hidden_c'][i] = critic_hidden_pre[1].detach().cpu().numpy()[:, 0, :]

                agent_action, agent_pre_tanh, new_hidden, v_pred, log_prob = self.agent_model.step(
                    agent_actor_obs, critic_obs_full, (actor_hidden_pre, critic_hidden_pre)
                )
                self.agent_hidden = new_hidden

                # === 对手使用正确的观测 ===
                opp_pair = self._get_opponent_action(opp_actor_obs, agent_actor_obs)

                tracker_action, target_action = (agent_action, opp_pair) if self.mission == 0 else (opp_pair, agent_action)
                obs_result, reward, terminated, truncated, info = self.env.step((tracker_action, target_action))
                done = terminated or truncated

                # 解析观测
                if isinstance(obs_result, tuple) and len(obs_result) == 2:
                    self.tracker_obs, self.target_obs = obs_result
                else:
                    self.tracker_obs = obs_result
                    self.target_obs = obs_result

                # 记录当前agent的actor_obs与critic_obs_full
                data['actor_obs'][i] = agent_actor_obs
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
                        self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy(self.opponent_role)
                    update_perf({'episode_reward': episode_reward, 'num_step': ep_len}, performance_dict)

                    episode_reward = 0.0
                    ep_len = 0
                    episodes += 1
                    episode_start = True
                    # reset env and hidden states
                    obs_tuple = self.env.reset()
                    if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
                        try:
                            self.tracker_obs, self.target_obs = obs_tuple[0]
                        except Exception:
                            self.tracker_obs = obs_tuple[0]
                            self.target_obs = obs_tuple[0]
                    else:
                        base_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
                        self.tracker_obs = base_obs
                        self.target_obs = base_obs
                    self.agent_hidden = None
                    self.opponent_hidden = None
                    if self.policy_manager and TrainingParameters.ADAPTIVE_SAMPLING:
                        self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy(self.opponent_role)

            # 终值价值
            last_base = self.tracker_obs if self.mission == 0 else self.target_obs
            critic_obs_last = np.concatenate([last_base, get_opponent_id_one_hot(self.current_opponent_id)])
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
            actions = np.zeros((n_steps, getattr(NetParameters, 'ACTION_DIM', 2)), dtype=np.float32)
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

            # ensure opponent sampling if needed
            if self.current_opponent_policy is None and self.policy_manager:
                self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy(self.opponent_role)

            episode_reward = 0.0
            ep_len = 0
            episodes = 0

            for i in range(n_steps):
                if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                    # Build critic context using tracker obs
                    critic_full = np.concatenate([tracker_obs, get_opponent_id_one_hot(self.current_opponent_id)])
                    # expert tracker action from APF (explicit)
                    teacher_name = "APF" if "APF" in TRACKER_POLICY_REGISTRY else sorted(TRACKER_POLICY_REGISTRY.keys())[0]
                    policy_fn = TRACKER_POLICY_REGISTRY.get(teacher_name)
                    if not policy_fn:
                        raise ValueError("No tracker policy found (APF or default).")
                    expert_pair = policy_fn(tracker_obs)
                    normalized = np.asarray(expert_pair, dtype=np.float32)
                    pre_tanh = Model.to_pre_tanh(normalized)
                    # opponent target action
                    opp_pair = self._get_opponent_action(target_obs, tracker_obs)
                    tracker_action, target_action = normalized, opp_pair
                else:
                    # target training path (kept for completeness; note ACTOR_VECTOR_LEN mismatch)
                    critic_full = np.concatenate([target_obs, get_opponent_id_one_hot(self.current_opponent_id)])
                    default_target = sorted(TARGET_POLICY_REGISTRY.keys())[0]
                    policy_cls = TARGET_POLICY_REGISTRY.get(default_target)
                    if not policy_cls:
                        raise ValueError("No target policy found")
                    expert_pair = policy_cls().get_action(target_obs)
                    normalized = np.asarray(expert_pair, dtype=np.float32)
                    pre_tanh = Model.to_pre_tanh(normalized)
                    opp_pair = self._get_opponent_action(tracker_obs, target_obs)
                    tracker_action, target_action = opp_pair, normalized

                next_obs, reward, terminated, truncated, info = self.env.step((tracker_action, target_action))
                done = terminated or truncated

                # record IL data
                if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                    actor_obs_arr[i] = tracker_obs
                else:
                    actor_obs_arr[i] = target_obs
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
                        self.current_opponent_policy, self.current_opponent_id = self.policy_manager.sample_policy(self.opponent_role)
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