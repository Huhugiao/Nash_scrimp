import numpy as np
import math
import sys
import os

from env_lib import apply_hard_mask
from map_config import EnvParameters
import map_config
from lstm.model_lstm import Model

class PurePursuitTracker:
    """
    Pure Pursuit Tracker Policy with Hard Mask.
    - Pursues the target by turning towards it.
    - Uses hard mask (ray-casting check) to avoid immediate collisions.
    """
    def __init__(self):
        self.kp_heading = 4.0
        self.max_speed = float(getattr(map_config, 'tracker_speed', 2.4))
        self.max_turn_deg = float(getattr(map_config, "tracker_max_angular_speed", 10.0))
        
        # Acceleration limits
        self.max_acc = float(getattr(map_config, 'tracker_max_acc', 0.1))
        self.max_ang_acc = float(getattr(map_config, 'tracker_max_ang_acc', 2.0))

    def reset(self):
        pass

    def get_action(self, observation, privileged_state=None):
        """
        Get action for tracker.
        obs is assumed to be the tracker observation vector.
        """
        obs = np.asarray(observation, dtype=np.float32)
        
        # --- 1. Parse Observation ---
        # Assuming modern obs structure from env.py
        # Index 0: self_vel (norm)
        # Index 2: self_heading (norm)
        # Index 4: target_bearing (norm)
        # Index 11+: Radar
        
        # Check obs dim to be safe
        expected_dim = 11 + EnvParameters.RADAR_RAYS
        if obs.shape[0] != expected_dim and obs.shape[0] != 75:
            # Fallback for old 27 dim if necessary, or just warn
            # For now assume mostly standard structure
            pass

        # Self State
        current_vel_norm = float(obs[0]) # [-1, 1]
        current_heading_norm = float(obs[2])
        current_heading_deg = (current_heading_norm + 1.0) * 180.0
        
        # Target State
        target_bearing_norm = float(obs[4]) # [-1, 1] => [-180, 180]
        # target_bearing_deg is relative to self heading
        target_bearing_deg = target_bearing_norm * 180.0
        
        # Radar
        radar_start = 11
        radar = obs[radar_start:]
        
        # --- 2. Pure Pursuit Logic ---
        # We want to turn towards the target (bearing 0)
        # Desired angular velocity is proportional to bearing error
        
        # Desired Turn
        # Target bearing is already the error (angle to target)
        desired_turn_deg = np.clip(target_bearing_deg, -self.max_turn_deg, self.max_turn_deg)
        angle_norm = desired_turn_deg / (self.max_turn_deg + 1e-6)
        
        # Desired Speed
        # Full speed ahead
        speed_norm = 1.0
        
        # Raw Action (before mask)
        raw_action = (angle_norm, speed_norm)
        
        # --- 3. Apply Hard Mask ---
        # Checks radar and modifies action if blocked
        safe_action = apply_hard_mask(raw_action, radar, current_heading_deg, role='tracker')
        
        safe_angle_norm, safe_speed_norm = safe_action
        
        # --- 4. Convert to Acceleration (Env expects acceleration control) ---
        # We need to compute angular acc and linear acc to achieve the desired velocity
        
        # Desired Velocity
        desired_speed_val = (safe_speed_norm + 1.0) / 2.0 * self.max_speed
        # Note: safe_speed_norm is [-1, 1]. If masking brakes, it might set it to -1 (speed 0).
        
        desired_heading_delta = safe_angle_norm * self.max_turn_deg
        desired_heading_deg = current_heading_deg + desired_heading_delta
        
        # Current Velocity (Approx)
        current_vel_val = (current_vel_norm + 1.0) / 2.0 * self.max_speed
        current_w_norm = float(obs[1]) # angular vel norm
        current_w = current_w_norm * self.max_turn_deg
        
        # Linear Control (P)
        lin_err = desired_speed_val - current_vel_val
        lin_acc = np.clip(2.0 * lin_err, -self.max_acc, self.max_acc)
        lin_acc_norm = lin_acc / self.max_acc
        
        # Angular Control (PD)
        # Re-calculate desired w based on the SAFE angle
        # desired_w ~= Kp * (desired_heading - current_heading)
        # But since we derived desired_heading from limited turn, desired_heading - current_heading IS desired_heading_delta
        # So we just target that turn rate?
        
        # Logic from CBF/Rule policies:
        # desired_w = Kp * angle_diff - Kd * current_w
        
        Kp_heading = 4.0
        Kd_heading = 0.5
        
        desired_w = Kp_heading * desired_heading_delta
        desired_w -= Kd_heading * current_w
        desired_w = np.clip(desired_w, -self.max_turn_deg, self.max_turn_deg)
        
        Kp_w = 2.0
        ang_acc = Kp_w * (desired_w - current_w)
        ang_acc = np.clip(ang_acc, -self.max_ang_acc, self.max_ang_acc)
        ang_acc_norm = ang_acc / self.max_ang_acc
        
        return np.array([ang_acc_norm, lin_acc_norm], dtype=np.float32)
