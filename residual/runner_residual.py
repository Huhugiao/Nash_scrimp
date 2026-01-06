import numpy as np
import torch
import ray
import map_config

from residual.alg_parameters_residual import SetupParameters, TrainingParameters, NetParameters, ResidualRLConfig
from mlp.model_mlp import Model as BaseModel
from residual.nets_residual import ResidualPolicyNetwork
from mlp.util_mlp import set_global_seeds
from env import TrackingEnv
from mlp.policymanager_mlp import PolicyManager


def _log_prob_from_pre_tanh(pre_tanh: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """
    Log prob of a Tanh-squashed Gaussian action, computed from pre-tanh sample u.
    action = tanh(u) * scale  (scale does not affect log_det_jac of tanh; constant scale ignored here)
    """
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    base_log_prob = dist.log_prob(pre_tanh)  # [B, action_dim]
    log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)  # [B, action_dim]
    return (base_log_prob - log_det_jac).sum(dim=-1)  # [B]


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TrainingParameters.N_ENVS + 1), 1))
class ResidualRunner(object):
    def __init__(self, env_id: int):
        self.ID = int(env_id)
        set_global_seeds(self.ID * 123 + 1000)

        # Ensure Ray worker uses obstacle density from alg_parameters_residual.py
        map_config.set_obstacle_density(SetupParameters.OBSTACLE_DENSITY)

        # Safety layer MUST be OFF during residual training (your requirement)
        self.env = TrackingEnv(enable_safety_layer=False)

        self.local_device = torch.device("cuda") if SetupParameters.USE_GPU_LOCAL else torch.device("cpu")

        # Load frozen base model
        self.base_model = BaseModel(self.local_device)
        ckpt = torch.load(ResidualRLConfig.BASE_MODEL_PATH, map_location=self.local_device)
        if isinstance(ckpt, dict) and "model_weights" in ckpt:
            self.base_model.set_weights(ckpt["model_weights"])
        elif isinstance(ckpt, dict) and "model" in ckpt:
            self.base_model.set_weights(ckpt["model"])
        else:
            self.base_model.network.load_state_dict(ckpt)
        self.base_model.network.eval()

        # Residual policy (synced from learner each rollout)
        self.residual_model = ResidualPolicyNetwork().to(self.local_device)
        self.residual_model.eval()

        # Opponent policy manager (optional)
        self.policy_manager = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
        self.current_opponent_policy = None

        self.reset_env()

    def reset_env(self):
        obs, _info = self.env.reset()
        # env returns (tracker_obs, target_obs)
        self.tracker_obs, self.target_obs = obs
        self.tracker_obs = np.asarray(self.tracker_obs, dtype=np.float32)
        self.target_obs = np.asarray(self.target_obs, dtype=np.float32)

    def _get_opponent_action(self, target_obs: np.ndarray) -> np.ndarray:
        # TrainingParameters.OPPONENT_TYPE == "random" in your config; keep robust behavior
        if self.policy_manager is not None:
            if self.current_opponent_policy is None:
                # PolicyManager API may vary; try best-effort
                try:
                    self.current_opponent_policy, _ = self.policy_manager.sample_policy("target")
                except Exception:
                    self.current_opponent_policy = "Greedy"
            try:
                return np.asarray(self.policy_manager.get_action(self.current_opponent_policy, target_obs), dtype=np.float32)
            except Exception:
                return np.zeros(2, dtype=np.float32)

        # Fallback: stationary target (should not happen with your current config)
        return np.zeros(2, dtype=np.float32)

    def run(self, residual_weights, opponent_weights, total_steps, policy_manager_state=None):
        # Sync residual weights
        self.residual_model.load_state_dict(residual_weights)
        self.residual_model.eval()

        # Restore policy_manager state if provided
        if self.policy_manager is not None and policy_manager_state:
            try:
                for name, history in policy_manager_state.items():
                    if name in self.policy_manager.win_history:
                        self.policy_manager.win_history[name] = list(history)
            except Exception:
                pass

        n_steps = int(TrainingParameters.N_STEPS)

        # Determine obs sizes dynamically (avoid relying on missing NetParameters constants)
        actor_dim = int(self.tracker_obs.size)
        target_dim = int(self.target_obs.size)
        critic_dim = actor_dim + target_dim
        action_dim = 2

        data = {
            "actor_obs": np.zeros((n_steps, actor_dim), dtype=np.float32),
            "critic_obs": np.zeros((n_steps, critic_dim), dtype=np.float32),
            "radar_obs": np.zeros((n_steps, NetParameters.RADAR_DIM), dtype=np.float32),
            "velocity_obs": np.zeros((n_steps, 2), dtype=np.float32),
            "base_actions": np.zeros((n_steps, action_dim), dtype=np.float32),
            # Store PRE-TANH action u for PPO logp consistency
            "actions": np.zeros((n_steps, action_dim), dtype=np.float32),
            "logp": np.zeros((n_steps,), dtype=np.float32),
            "values": np.zeros((n_steps,), dtype=np.float32),
            "rewards": np.zeros((n_steps,), dtype=np.float32),
            "dones": np.zeros((n_steps,), dtype=np.bool_),
            "episode_starts": np.zeros((n_steps,), dtype=np.bool_),
        }

        performance = {"per_r": [], "per_episode_len": [], "win": []}
        completed_opponents = []

        # Safety shaping (adds emphasis on "not too close", on top of env_reward)
        SAFE_EDGE_PX = 18.0      # below this edge distance, penalize strongly
        SAFE_W = 0.15            # safety emphasis weight (keep modest; env_reward already has shaping)
        EXTRA_COLLISION_PEN = 5.0  # extra penalty when tracker_collision happens (env_reward already penalizes)
        action_pen_coef = float(getattr(ResidualRLConfig, "ACTION_PENALTY_COEF", 0.0))

        # Residual scale (best-effort from network)
        max_scale = 1.0
        try:
            max_scale = float(getattr(self.residual_model.actor, "max_scale", 1.0))
        except Exception:
            max_scale = 1.0

        episode_reward = 0.0
        ep_len = 0
        episode_start = True

        residual_l2_sum = 0.0
        collision_steps = 0
        min_edge_sum = 0.0
        min_edge_count = 0

        for t in range(n_steps):
            tracker_obs = self.tracker_obs
            target_obs = self.target_obs
            critic_obs = np.concatenate([tracker_obs, target_obs]).astype(np.float32)

            # Base action (frozen)
            base_action, _, _, _ = self.base_model.evaluate(tracker_obs, critic_obs, greedy=True)
            base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
            base_action_t = torch.from_numpy(base_action).float().to(self.local_device).unsqueeze(0)

            # Residual inputs
            radar_obs = np.asarray(tracker_obs[NetParameters.ACTOR_SCALAR_LEN:], dtype=np.float32)
            vel_obs = np.asarray(tracker_obs[0:2], dtype=np.float32)

            radar_t = torch.from_numpy(radar_obs).float().to(self.local_device).unsqueeze(0)
            vel_t = torch.from_numpy(vel_obs).float().to(self.local_device).unsqueeze(0)

            with torch.no_grad():
                mean, log_std = self.residual_model.actor(radar_t, base_action_t, vel_t)
                std = torch.exp(log_std)
                u = mean + std * torch.randn_like(mean)  # pre-tanh
                residual_used = torch.tanh(u) * max_scale  # bounded executed residual
                logp = _log_prob_from_pre_tanh(u, mean, log_std)
                value = self.residual_model.critic(radar_t).squeeze(-1)

                fused = ResidualPolicyNetwork.fuse_actions(base_action_t, residual_used)
                tracker_action = fused.squeeze(0).cpu().numpy().astype(np.float32)

            # Opponent action
            target_action = self._get_opponent_action(target_obs)
            obs_next, env_reward, terminated, truncated, info = self.env.step((tracker_action, target_action))
            done = bool(terminated or truncated)

            if bool(info.get("tracker_collision", False)):
                collision_steps += 1

            # --- Reward: base env reward + extra safety emphasis - residual penalty ---
            r = float(env_reward)

            if bool(info.get("tracker_collision", False)):
                r -= EXTRA_COLLISION_PEN

            # Absolute closeness penalty using info['min_edge_distance'] (px). Smaller => more dangerous.
            min_edge = info.get("min_edge_distance", None)
            if min_edge is not None:
                min_edge_sum += float(min_edge)
                min_edge_count += 1
                d = float(min_edge)
                if d < SAFE_EDGE_PX:
                    x = (SAFE_EDGE_PX - d) / (SAFE_EDGE_PX + 1e-6)
                    r -= SAFE_W * (x * x)

            # Penalize magnitude of executed residual (bounded)
            residual_np = residual_used.squeeze(0).cpu().numpy().astype(np.float32)
            residual_l2_sum += float(np.linalg.norm(residual_np))
            r -= action_pen_coef * float(np.sum(residual_np * residual_np))

            # Store rollout
            data["actor_obs"][t] = tracker_obs
            data["critic_obs"][t] = critic_obs
            data["radar_obs"][t] = radar_obs
            data["velocity_obs"][t] = vel_obs
            data["base_actions"][t] = base_action
            data["actions"][t] = u.squeeze(0).cpu().numpy().astype(np.float32)  # pre-tanh
            data["logp"][t] = float(logp.item())
            data["values"][t] = float(value.item())
            data["rewards"][t] = float(r)
            data["dones"][t] = done
            data["episode_starts"][t] = episode_start

            episode_start = False
            episode_reward += float(r)
            ep_len += 1

            # Unpack next obs
            (self.tracker_obs, self.target_obs) = obs_next
            self.tracker_obs = np.asarray(self.tracker_obs, dtype=np.float32)
            self.target_obs = np.asarray(self.target_obs, dtype=np.float32)

            if done:
                win = 1 if info.get("reason") == "tracker_caught_target" else 0
                performance["per_r"].append(float(episode_reward))
                performance["per_episode_len"].append(int(ep_len))
                performance["win"].append(int(win))

                if self.policy_manager is not None:
                    try:
                        self.policy_manager.update_win_rate(self.current_opponent_policy, win)
                        completed_opponents.append(self.current_opponent_policy)
                        self.current_opponent_policy, _ = self.policy_manager.sample_policy("target")
                    except Exception:
                        pass

                episode_reward = 0.0
                ep_len = 0
                episode_start = True
                self.reset_env()

        # Compute GAE returns
        with torch.no_grad():
            radar_last = np.asarray(self.tracker_obs[NetParameters.ACTOR_SCALAR_LEN:], dtype=np.float32)
            radar_last_t = torch.from_numpy(radar_last).float().to(self.local_device).unsqueeze(0)
            last_value = float(self.residual_model.critic(radar_last_t).squeeze(-1).item())

        advantages = np.zeros((n_steps,), dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - float(data["dones"][t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(data["dones"][t + 1])
                next_value = float(data["values"][t + 1])
            delta = float(data["rewards"][t]) + float(TrainingParameters.GAMMA) * next_value * next_non_terminal - float(data["values"][t])
            lastgaelam = delta + float(TrainingParameters.GAMMA) * float(TrainingParameters.LAM) * next_non_terminal * lastgaelam
            advantages[t] = float(lastgaelam)

        data["returns"] = advantages + data["values"]

        pm_state = None
        if self.policy_manager is not None:
            try:
                pm_state = {k: list(v) for k, v in self.policy_manager.win_history.items()}
            except Exception:
                pm_state = None


        extra_stats = {
            "residual_l2_mean": float(residual_l2_sum) / float(n_steps),
            "collision_rate": float(collision_steps) / float(n_steps),
            "min_edge_distance_mean": (float(min_edge_sum) / float(min_edge_count)) if min_edge_count > 0 else float("nan"),
        }

        return {
            "data": data,
            "performance": performance,
            "episodes": len(performance["per_r"]),
            "policy_manager_state": pm_state,
            "completed_opponents": completed_opponents,
            "extra_stats": extra_stats,
        }
