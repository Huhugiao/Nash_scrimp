import numpy as np
from collections import deque
from rule_policies import (
    TRACKER_POLICY_REGISTRY,
    TARGET_POLICY_REGISTRY
)
# 使用 MLP 专用参数
from mlp.alg_parameters_mlp import TrainingParameters
from map_config import EnvParameters

class PolicyManager:
    """
    Manages target policies with support for weighted and adaptive random selection.
    专门用于tracker训练时的对手(target)策略管理 (MLP版本)
    """
    def __init__(self, training_params=None):
        """
        Args:
            training_params: 可选的训练参数类，默认使用alg_parameters_mlp.TrainingParameters
        """
        if training_params is None:
            training_params = TrainingParameters
            
        self.exclude_expert = getattr(training_params, 'OPPONENT_TYPE', 'random') == "random_nonexpert"
        
        # 只管理target策略
        default_target = sorted(TARGET_POLICY_REGISTRY.keys())[0]
        target_policies = {name: cls() for name, cls in TARGET_POLICY_REGISTRY.items()}
        self._policies = target_policies
        
        # 获取权重配置
        raw_weights = getattr(training_params, 'RANDOM_OPPONENT_WEIGHTS', {}).get("target", {})
        
        if self.exclude_expert:
            raw_weights = {k: v for k, v in raw_weights.items() if k != default_target}
        
        # 如果没有配置权重，则使用默认均匀权重
        self._base_weights = raw_weights or {name: 1.0 for name in TARGET_POLICY_REGISTRY}
        self._policy_ids = {name: idx for idx, name in enumerate(sorted(TARGET_POLICY_REGISTRY))}
        
        self.min_history = getattr(training_params, 'ADAPTIVE_SAMPLING_MIN_GAMES', 32)
        self.adaptive_sampling = getattr(training_params, 'ADAPTIVE_SAMPLING', True)
        self.adaptive_strength = getattr(training_params, 'ADAPTIVE_SAMPLING_STRENGTH', 1.8)
        
        self.win_history = {
            name: deque(maxlen=getattr(training_params, 'ADAPTIVE_SAMPLING_WINDOW', 400))
            for name in self._base_weights
        }

    def update_win_rate(self, policy_name, win):
        """Record the outcome of an episode against a specific policy."""
        if self.adaptive_sampling and policy_name in self.win_history:
            self.win_history[policy_name].append(1 if win else 0)

    def sample_policy(self, role="target"):
        """随机采样一个target策略"""
        if role != "target":
            raise ValueError(f"PolicyManager only manages target policies, got role={role}")

        weights = {}
        for name, base in self._base_weights.items():
            if base <= 0:
                weights[name] = 0.0
                continue
            
            adjusted = max(base, 1e-6)
            if self.adaptive_sampling:
                history = self.win_history[name]
                if len(history) >= self.min_history:
                    win_rate = float(np.mean(history))
                    adjusted *= max(1.0 - win_rate, 0.05) ** self.adaptive_strength
            weights[name] = adjusted
            
        policies = list(weights.keys())
        total = float(sum(weights.values()))
        probs = [weights[p] / total for p in policies] if total > 0.0 else [1.0 / len(policies)] * len(policies)
        policy_name = np.random.choice(policies, p=probs)
        policy_id = self._policy_ids.get(policy_name, -1)
        return policy_name, policy_id
    
    def reset(self):
        for policy in self._policies.values():
            if hasattr(policy, 'reset'):
                policy.reset()
    
    def get_action(self, policy_name, observation, privileged_state=None):
        """获取target策略动作"""
        if policy_name not in self._policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        policy = self._policies[policy_name]
        
        if hasattr(policy, 'get_action'):
            import inspect
            sig = inspect.signature(policy.get_action)
            if 'privileged_state' in sig.parameters:
                return policy.get_action(observation, privileged_state)
            else:
                return policy.get_action(observation)
        elif callable(policy):
            return policy(observation)
        else:
            raise ValueError(f"Policy {policy_name} is not callable")

    def get_policy_id(self, policy_name):
        return self._policy_ids.get(policy_name, -1)
