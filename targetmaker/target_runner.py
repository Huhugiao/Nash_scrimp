import numpy as np
import torch
import ray
import os
import random

# Set SDL to dummy for headless rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"

from targetmaker.target_alg_parameters import TargetTrainingParameters, TargetNetParameters, TargetSetupParameters
from targetmaker.target_model import TargetPPO
from mlp.util_mlp import set_global_seeds
from env import TrackingEnv
from cbf_controller import CBFTracker
from targetmaker.tracker_policies import PurePursuitTracker
from targetmaker.reward_shaping import TargetRewardShaper
import math
import numpy as np

@ray.remote(num_cpus=1, num_gpus=TargetSetupParameters.NUM_GPU / max((TargetTrainingParameters.N_ENVS + 1), 1))
class TargetRunner(object):
    def __init__(self, env_id):
        self.ID = env_id
        set_global_seeds(env_id * 321)
        self.env = TrackingEnv()
        self.local_device = torch.device('cuda') if TargetSetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        
        self.agent = TargetPPO(self.local_device)
        self.opponent_policy = PurePursuitTracker()
        self.reward_shaper = TargetRewardShaper()
        self.last_dist = 0.0
        
        # Init buffers
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

    def run_rollout(self, weights, style_name):
        self.agent.set_weights(weights)
        
        n_steps = TargetTrainingParameters.N_STEPS
        
        # PPO Data Structure
        data = {
            'actor_obs': np.zeros((n_steps, TargetNetParameters.TARGET_RAW_LEN), dtype=np.float32),
            'critic_obs': np.zeros((n_steps, TargetNetParameters.STATE_DIM), dtype=np.float32),
            'actions': np.zeros((n_steps, TargetNetParameters.ACTION_DIM), dtype=np.float32),
            'rewards': np.zeros(n_steps, dtype=np.float32),
            'values': np.zeros(n_steps, dtype=np.float32),
            'logp': np.zeros(n_steps, dtype=np.float32),
            'dones': np.zeros(n_steps, dtype=np.float32)
        }
        
        ep_rewards = []
        ep_success = []
        curr_ep_r = 0
        
        for i in range(n_steps):
            critic_obs = np.concatenate([self.target_obs, self.tracker_obs])
            
            action, value, log_prob = self.agent.step(self.target_obs, critic_obs)
            
            # Clip Action for Env
            # Note: PPO outputs raw gaussian, we clip to [-1, 1] then scale in env? 
            # Env expects [-1, 1].
            action_clamped = np.clip(action, -1.0, 1.0)
            
            try:
                privileged = self.env.get_privileged_state() if hasattr(self.env, "get_privileged_state") else None
                tracker_action = self.opponent_policy.get_action(self.tracker_obs, privileged_state=privileged)
            except:
                tracker_action = self.opponent_policy.get_action(self.tracker_obs)
            
            obs_result, raw_reward, terminated, truncated, info = self.env.step((tracker_action, action_clamped))
            done = terminated or truncated
            
            # Distance Delta Calculation
            try:
                dx = self.env.tracker['x'] - self.env.target['x']
                dy = self.env.tracker['y'] - self.env.target['y']
                curr_dist = math.hypot(dx, dy)
            except:
                curr_dist = self.last_dist
            
            dist_delta = curr_dist - self.last_dist
            self.last_dist = curr_dist
            
            # Reward Shaping
            r = self.reward_shaper.compute_reward(style_name, self.target_obs, self.tracker_obs, info, done, raw_tracker_reward=raw_reward, dist_delta=dist_delta)
            
            data['actor_obs'][i] = self.target_obs
            data['critic_obs'][i] = critic_obs
            data['actions'][i] = action # Store RAW action for PPO update distribution logging/prob
            data['values'][i] = value
            data['logp'][i] = log_prob
            data['rewards'][i] = r
            data['dones'][i] = float(done)
            
            curr_ep_r += r
            
            # Next Obs
            if isinstance(obs_result, tuple) and len(obs_result) == 2:
                 self.tracker_obs, self.target_obs = obs_result
            else:
                 self.tracker_obs = obs_result
                 self.target_obs = obs_result
            self.tracker_obs = np.asarray(self.tracker_obs, dtype=np.float32)
            self.target_obs = np.asarray(self.target_obs, dtype=np.float32)
            
            if done:
                 is_caught = (info.get('reason') == 'tracker_caught_target')
                 success = 1 if not is_caught else 0
                 ep_rewards.append(curr_ep_r)
                 ep_success.append(success)
                 curr_ep_r = 0
                 self.reset_env()
                 if hasattr(self.opponent_policy, 'reset'):
                      self.opponent_policy.reset()

        # GAE Calculation
        next_val = 0
        with torch.no_grad():
             c_obs = np.concatenate([self.target_obs, self.tracker_obs])
             _, next_val, _ = self.agent.step(self.target_obs, c_obs)
             
        advantages = np.zeros_like(data['rewards'])
        lastgaelam = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - data['dones'][t]
                nextvalues = next_val
            else:
                nextnonterminal = 1.0 - data['dones'][t+1]
                nextvalues = data['values'][t+1]
            delta = data['rewards'][t] + TargetTrainingParameters.GAMMA * nextvalues * nextnonterminal - data['values'][t]
            advantages[t] = lastgaelam = delta + TargetTrainingParameters.GAMMA * TargetTrainingParameters.LAM * nextnonterminal * lastgaelam
        
        return {
            'data': data,
            'adv': advantages,
            'returns': advantages + data['values'],
            'metrics': {'r': ep_rewards, 'succ': ep_success}
        }
        
    def evaluate_policy(self, weights, episodes, style="survival", record_gif=False):
        # ... logic similar to before but for PPO ...
        self.agent.set_weights(weights)
        eval_r = []
        eval_succ = []
        frames = []
        
        for ep_idx in range(episodes):
            self.reset_env()
            curr_ep_r = 0
            done = False
            
            # Record first episode only
            should_record = record_gif and (ep_idx == 0)
            
            if should_record:
                try:
                    f = self.env.render(mode='rgb_array')
                    if f is not None: frames.append(f)
                except Exception as e:
                    print(f"Render Error: {e}")

            while not done:
                 mean = self.agent.evaluate(self.target_obs)
                 action = np.clip(mean, -1.0, 1.0) # Proper clip
                 try:
                    privileged = self.env.get_privileged_state() if hasattr(self.env, "get_privileged_state") else None
                    tracker_action = self.opponent_policy.get_action(self.tracker_obs, privileged_state=privileged)
                 except:
                    tracker_action = self.opponent_policy.get_action(self.tracker_obs)
                 
                 obs_result, raw_reward, terminated, truncated, info = self.env.step((tracker_action, action))
                 done = terminated or truncated
                 
                 # Distance Delta
                 try:
                    dx = self.env.tracker['x'] - self.env.target['x']
                    dy = self.env.tracker['y'] - self.env.target['y']
                    curr_dist = math.hypot(dx, dy)
                 except:
                    curr_dist = self.last_dist
                
                 dist_delta = curr_dist - self.last_dist
                 self.last_dist = curr_dist

                 # Basic Metric Calculation (Raw Reward Inverse)
                 # Note: evaluate usually tracks raw success, but let's track shaped reward for consistency with train metric
                 shaped_r = self.reward_shaper.compute_reward(style, self.target_obs, self.tracker_obs, info, done, raw_tracker_reward=raw_reward, dist_delta=dist_delta) 
                 curr_ep_r += shaped_r
                 
                 if should_record:
                     try:
                         f = self.env.render(mode='rgb_array')
                         if f is not None: frames.append(f)
                     except:
                         pass
                 
                 if isinstance(obs_result, tuple): self.tracker_obs, self.target_obs = obs_result
                 else: self.tracker_obs = obs_result; self.target_obs = obs_result
                 self.tracker_obs = np.asarray(self.tracker_obs, dtype=np.float32)
                 self.target_obs = np.asarray(self.target_obs, dtype=np.float32)

            is_caught = (info.get('reason') == 'tracker_caught_target')
            eval_succ.append(1 if not is_caught else 0)
            eval_r.append(curr_ep_r)
            
        return {'r': np.mean(eval_r), 'succ': np.mean(eval_succ), 'frames': frames}
