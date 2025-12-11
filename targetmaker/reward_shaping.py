import numpy as np
from map_config import EnvParameters

class TargetRewardShaper:
    """
    Handles reward shaping for different target styles.
    """
    def __init__(self):
        # Common
        self.SURVIVAL_REWARD = 0.1
        self.DEATH_PENALTY = -10.0
        
        # Survival
        self.SURVIVAL_DIST_SCALE = 0.05  # Bonus for keeping distance
        
        # Stealth
        self.STEALTH_HIDDEN_BONUS = 0.1
        self.STEALTH_VISIBLE_PENALTY = -0.1

    def compute_reward(self, style, target_obs, tracker_obs, info, done, raw_tracker_reward=0.0, dist_delta=0.0):
        """
        Compute shaped reward based on style and observation.
        
        Args:
            style (str): 'survival', 'stealth', 'taunt'
            target_obs (np.array): Target's observation
            tracker_obs (np.array): Tracker's observation (from TargetRunner)
            info (dict): Step info from env
            done (bool): Whether episode is done
            raw_tracker_reward (float): The raw reward the tracker received from env.
            dist_delta (float): curr_dist - prev_dist (Positive = Moving Away)
            
        Returns:
            float: Shaped reward
        """
        # Parse common features from Tracker Obs (which contains relative info)
        # Structure from env._get_tracker_observation in env.py:
        # 0: self_vel (tracker)
        # 1: self_angular_vel (tracker)
        # 2: self_heading
        # 3: target_distance_norm
        # 8: in_fov
        
        tr_obs = np.asarray(tracker_obs, dtype=np.float32).reshape(-1)
        
        tracker_w_norm = float(tr_obs[1])
        dist_norm = float(tr_obs[3]) # [-1, 1], -1=close, 1=far (max range)
        in_fov = float(tr_obs[8])
        
        reward = 0.0
        
        # 1. Base Survival Reward (Per Step)
        if not done:
            reward += self.SURVIVAL_REWARD
            
        # 2. Death Penalty (Terminal)
        if done and info.get('reason') == 'tracker_caught_target':
            reward += self.DEATH_PENALTY
            return reward # Terminal, return immediately usually? Or add step reward? usually terminal overrides.
        
        # 3. Collision Penalty
        if info.get('target_collided', False):
            reward -= 2.0
            
        # 4. Style Specific Rewards
        if style == 'survival':
            # Differential Distance Reward (Match Tracker's principle)
            # Tracker gets positive for closing distance; Target gets positive for opening it.
            reward += 0.05 * dist_delta
            
        elif style == 'stealth':
            # 1. Base Survival Component (Keep Distance)
            reward += 0.05 * dist_delta

            # 2. Stealth Component (Visibility)
            check_occluded = float(tr_obs[9])
            is_visible = (in_fov > 0.5) and (check_occluded < 0.5)
            
            if not is_visible:
                reward += 0.4  # Hidden bonus
            else:
                # Visible: Penalty + Steering
                # Problem: Agent doesn't know *how* to hide.
                # Solution: If visible, punish being "in the open". Encourage hugging walls.
                
                # Parse Target Radar (Indices 8:72 in target_obs)
                # Radar values are [-1, 1], where -1 is 0 distance, 1 is max range.
                tg_obs = np.asarray(target_obs, dtype=np.float32).reshape(-1)
                radar_readings = tg_obs[8:72] 
                
                # Find nearest obstacle distance (normalized [-1, 1])
                # -1.0 means touching wall, 1.0 means no wall in range
                min_radar_val = np.min(radar_readings) 
                
                # Convert to [0, 1] scale where 0 is touching, 1 is max range
                dist_to_wall_frac = (min_radar_val + 1.0) / 2.0
                
                # Base Penalty
                reward -= 0.2
                
                # "Exposure" Penalty: Punish being far from obstacles when visible
                # Max penalty (open field) = -0.2
                # Min penalty (hugging wall) = 0.0
                reward -= 0.2 * dist_to_wall_frac
                
        elif style == 'taunt':
            # 1. Base Survival Component (Keep Distance)
            reward += 0.01 * dist_delta
            
            # 2. Reward for Tracker Spinning
            tg_obs = np.asarray(target_obs, dtype=np.float32).reshape(-1)
            
            spin_reward = abs(tracker_w_norm) * 0.05
            reward += spin_reward

            
        return reward
