import numpy as np
from collections import deque
from rule_policies import (
    TRACKER_POLICY_REGISTRY,
    TARGET_POLICY_REGISTRY
)
from alg_parameters import TrainingParameters
from map_config import EnvParameters

class PolicyManager:
    """
    Manages policies with support for weighted and adaptive random selection.
    """
    def __init__(self):
        self.exclude_expert = getattr(TrainingParameters, 'OPPONENT_TYPE', 'random') == "random_nonexpert"
        
        # Get default policies (first in alphabetical order)
        default_tracker = sorted(TRACKER_POLICY_REGISTRY.keys())[0]
        default_target = sorted(TARGET_POLICY_REGISTRY.keys())[0]
        
        tracker_cls = TRACKER_POLICY_REGISTRY.get(default_tracker)
        if isinstance(tracker_cls, type):
            tracker_policy = tracker_cls()
        else:
            tracker_policy = tracker_cls
            
        target_policies = {name: cls() for name, cls in TARGET_POLICY_REGISTRY.items()}
        self._policies = {default_tracker: tracker_policy, **target_policies}
        raw_weights = getattr(TrainingParameters, 'RANDOM_OPPONENT_WEIGHTS', {}).get("target", {})
        if self.exclude_expert:
            raw_weights = {k: v for k, v in raw_weights.items() if k != default_target}
        self._base_weights = raw_weights or {name: 1.0 for name in TARGET_POLICY_REGISTRY}
        self._policy_ids = {name: idx for idx, name in enumerate(sorted(TARGET_POLICY_REGISTRY))}
        self.min_history = getattr(TrainingParameters, 'ADAPTIVE_SAMPLING_MIN_GAMES', 32)
        self.adaptive_sampling = TrainingParameters.ADAPTIVE_SAMPLING
        self.win_history = {
            name: deque(maxlen=TrainingParameters.ADAPTIVE_SAMPLING_WINDOW)
            for name in self._base_weights
        }
        
    def get_policies_by_role(self, role):
        if role == "tracker":
            return sorted(TRACKER_POLICY_REGISTRY.keys())
        if role == "target":
            return list(self._base_weights.keys())
        raise ValueError(f"Unknown role: {role}")

    def update_win_rate(self, policy_name, win):
        """Record the outcome of an episode against a specific policy."""
        if self.adaptive_sampling and policy_name in self.win_history:
            self.win_history[policy_name].append(1 if win else 0)

    def sample_policy(self, role):
        """Randomly sample a policy name based on weights, possibly adaptive."""
        if role != "target":
            policies = self.get_policies_by_role(role)
            return np.random.choice(policies), -1

        weights = {}
        for name, base in self._base_weights.items():
            adjusted = max(base, 1e-6)
            if self.adaptive_sampling:
                history = self.win_history[name]
                if len(history) >= self.min_history:
                    win_rate = float(np.mean(history))
                    adjusted *= max(1.0 - win_rate, 0.05) ** TrainingParameters.ADAPTIVE_SAMPLING_STRENGTH
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
        if policy_name not in self._policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        policy = self._policies[policy_name]
        
        # 如果提供了特权状态，设置到全局变量
        if privileged_state is not None:
            from rule_policies import set_privileged_state
            set_privileged_state(privileged_state)
        
        if callable(policy) and not hasattr(policy, 'get_action'):
            return policy(observation)
        return policy.get_action(observation)

    def get_policy_id(self, policy_name):
        return self._policy_ids.get(policy_name, -1)