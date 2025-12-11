"""
Rule-based policies for tracker and target agents.
This module provides baseline policies that can be used for evaluation and testing.
"""
import math
import numpy as np
import map_config
from env_lib import apply_hard_mask
from map_config import EnvParameters

# Optional CBFTracker (requires cvxpy)
try:
    from cbf_controller import CBFTracker
    CBF_AVAILABLE = True
except ImportError:
    CBFTracker = None
    CBF_AVAILABLE = False
    print("Warning: CBFTracker not available (cvxpy missing?)")

# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_angle(angle_deg: float):
    """Normalize angle to [-180, 180] range."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return float(angle_deg)

def apply_hard_mask_adapter(action, obs_dict, role='tracker', safety_dist=None):
    """
    Adapter for apply_hard_mask to work with obs_dict.
    """
    radar = obs_dict.get('radar', [])
    heading = obs_dict.get('self_heading_deg', 0.0)
    heading = obs_dict.get('self_heading_deg', 0.0)
    return apply_hard_mask(action, radar, heading, role, safety_dist)

def velocity_to_acceleration(desired_angle_deg, desired_speed_factor, obs_dict, role='target'):
    """
    Convert desired velocity (angle, speed_factor) to acceleration command.
    """
    # Get limits
    if role == 'tracker':
        max_acc = float(getattr(map_config, 'tracker_max_acc', 0.1))
        max_ang_acc = float(getattr(map_config, 'tracker_max_ang_acc', 2.0))
        max_speed = float(getattr(map_config, 'tracker_speed', 2.4))
        max_ang_speed = float(getattr(map_config, 'tracker_max_angular_speed', 10.0))
    else:
        max_acc = float(getattr(map_config, 'target_max_acc', 0.1))
        max_ang_acc = float(getattr(map_config, 'target_max_ang_acc', 2.0))
        max_speed = float(getattr(map_config, 'target_speed', 2.0))
        max_ang_speed = float(getattr(map_config, 'target_max_angular_speed', 12.0))
    
    current_vel_norm = obs_dict.get('self_vel', 0.0)
    current_vel = (current_vel_norm + 1.0) / 2.0 * max_speed
    
    # For angular velocity, if missing, assume 0.
    # Tracker has it.
    current_ang_vel_norm = obs_dict.get('self_angular_vel', 0.0)
    current_w = current_ang_vel_norm * max_ang_speed
    
    # Desired velocity
    desired_v = desired_speed_factor * max_speed
    
    # Desired angular velocity (PD controller)
    current_heading = obs_dict.get('self_heading_deg', 0.0)
    angle_diff = _normalize_angle(desired_angle_deg - current_heading)
    
    # Tuned Gains
    Kp_heading = 4.0  # Increased from 0.5 to 4.0 for faster turning
    Kd_heading = 0.5  # Damping to prevent oscillation
    
    # Feedforward term (if we knew target angular velocity, but we don't here)
    desired_w = Kp_heading * angle_diff
    
    # Damping
    if current_w != 0.0:
        desired_w -= Kd_heading * current_w
        
    desired_w = np.clip(desired_w, -max_ang_speed, max_ang_speed)
    
    # Calculate acceleration
    # Linear P-controller
    Kp_v = 2.0
    lin_acc = Kp_v * (desired_v - current_vel)
    
    # Angular P-controller (since we output angular acceleration)
    # ang_acc = Kp_w * (desired_w - current_w)
    Kp_w = 2.0
    ang_acc = Kp_w * (desired_w - current_w)
    
    # Clamp
    lin_acc = np.clip(lin_acc, -max_acc, max_acc)
    ang_acc = np.clip(ang_acc, -max_ang_acc, max_ang_acc)
    
    # Normalize
    lin_acc_norm = lin_acc / max_acc
    ang_acc_norm = ang_acc / max_ang_acc
    
    return np.array([ang_acc_norm, lin_acc_norm], dtype=np.float32)


# ============================================================================
# Observation Parsing
# ============================================================================

def _parse_observation(observation):
    """
    解析新的观测格式
    Tracker: 27维 (相对观测，受FOV限制，360度雷达)
    Target: 24维 (全局观测，全知视角，360度雷达)
    """
    obs = np.asarray(observation, dtype=np.float64)
    
    # 根据维度判断观测类型
    # 根据维度判断观测类型
    if obs.shape[0] == 72:
        # Target观测解析（全局坐标，全知视角，64维雷达）
        return {
            'obs_type': 'target',
            'self_x_norm': float(obs[0]),
            'self_y_norm': float(obs[1]),
            'self_heading_norm': float(obs[2]),
            'tracker_x_norm': float(obs[3]),
            'tracker_y_norm': float(obs[4]),
            'tracker_heading_norm': float(obs[5]),
            'self_vel': float(obs[6]),
            'tracker_vel': float(obs[7]),
            'radar': obs[8:72].astype(np.float64),
            'self_x': (float(obs[0]) + 1.0) / 2.0 * map_config.width,
            'self_y': (float(obs[1]) + 1.0) / 2.0 * map_config.height,
            'self_heading_deg': (float(obs[2]) + 1.0) * 180.0,
            'tracker_x': (float(obs[3]) + 1.0) / 2.0 * map_config.width,
            'tracker_y': (float(obs[4]) + 1.0) / 2.0 * map_config.height,
            'tracker_heading_deg': (float(obs[5]) + 1.0) * 180.0,
        }
    elif obs.shape[0] == 75:
        # Tracker观测解析（相对观测，受FOV限制，64维雷达）
        distance_norm = float(obs[3])
        actual_distance = (distance_norm + 1.0) / 2.0 * EnvParameters.FOV_RANGE
        
        return {
            'obs_type': 'tracker',
            'self_vel': float(obs[0]),
            'self_angular_vel': float(obs[1]),
            'self_heading_deg': (float(obs[2]) + 1.0) * 180.0,
            'target_distance_norm': distance_norm,
            'target_distance': actual_distance,
            'target_bearing_deg': float(obs[4]) * 180.0,
            'target_relative_speed': float(obs[5]),
            'target_relative_angular_vel': float(obs[6]),
            'fov_edge': float(obs[7]),
            'in_fov': float(obs[8]),
            'occluded': float(obs[9]),
            'unobserved_steps': float(obs[10]),
            'radar': obs[11:75].astype(np.float64)
        }
    else:
        raise ValueError(f"Unexpected observation dimension: {obs.shape[0]}, expected 72 (target) or 75 (tracker)")


# ============================================================================
# Target Policy
# ============================================================================

class CoverSeekerTarget:
    """
    CoverSeeker Target Policy (Redesigned)
    - Pre-computes cover points around obstacles.
    - Selects best cover point based on occlusion, distance, and alignment.
    - Switches between Cover, Orbit, and Flee modes.
    """
    def __init__(self):
        self.cover_points = [] # List of (point_np, obs_center_np, obs_radius)
        self.agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
        self._compute_cover_points()
        self.last_target_pos = None

    def reset(self):
        self.last_target_pos = None
        # Re-compute in case map config changed (though usually static per episode)
        self._compute_cover_points()

    def _get_obstacle_info(self, obs):
        """Extract center and effective radius from obstacle dict."""
        otype = obs.get('type')
        if otype == 'circle':
            return np.array([obs['cx'], obs['cy']]), float(obs['r'])
        elif otype == 'rect':
            w, h = obs.get('w', 0), obs.get('h', 0)
            cx = obs['x'] + w / 2.0
            cy = obs['y'] + h / 2.0
            # Approximate radius for bounding circle
            return np.array([cx, cy]), float(math.hypot(w, h) / 2.0)
        elif otype == 'segment':
            x1, y1 = obs['x1'], obs['y1']
            x2, y2 = obs['x2'], obs['y2']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            length = math.hypot(x2 - x1, y2 - y1)
            return np.array([cx, cy]), float(length / 2.0)
        return None, None

    def _compute_cover_points(self):
        """Pre-compute cover points around each obstacle"""
        self.cover_points = []
        obstacles = getattr(map_config, 'obstacles', [])
        
        # Directions: 8 cardinal/ordinal directions
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        
        for obs in obstacles:
            center, radius = self._get_obstacle_info(obs)
            if center is None:
                continue
                
            # Distance: obs_radius + agent_radius * 2 (safety margin)
            safe_dist = radius + self.agent_radius * 3.0 
            
            for d in directions:
                point = center + d * safe_dist
                
                # Check bounds
                if not (0 <= point[0] <= map_config.width and 0 <= point[1] <= map_config.height):
                    continue
                    
                # Check collision (stricter check for cover point utility)
                # We use a slightly larger padding to ensure we can actually stand there
                import env_lib # Import env_lib here to avoid circular dependency if it imports policies
                if env_lib.is_point_blocked(point[0], point[1], padding=self.agent_radius):
                    continue
                    
                self.cover_points.append((point, center, radius))

    def _is_tracker_visible(self, self_pos, tracker_pos):
        """Check if tracker has Line of Sight to self."""
        # Using env_lib.ray_distance_grid from tracker to self
        # If distance < actual distance, it's blocked.
        rel = self_pos - tracker_pos
        dist = np.linalg.norm(rel)
        if dist < 1e-3: return True
        
        angle = math.atan2(rel[1], rel[0])
        # Check ray from tracker
        import env_lib # Import env_lib here to avoid circular dependency if it imports policies
        ray_dist = env_lib.ray_distance_grid(
            (tracker_pos[0], tracker_pos[1]), 
            angle, 
            dist, 
            padding=0.0 # Line of sight check
        )
        return ray_dist >= (dist - 5.0) # Tolerance

    def _select_best_cover(self, self_pos, tracker_pos, self_vel):
        """Select cover point that places obstacle between self and tracker"""
        best_score = -float('inf')
        best_cover = None
        best_obs_center = None
        best_obs_radius = 0.0
        
        # Current heading vector logic could be added, using velocity for now
        if np.linalg.norm(self_vel) > 0.1:
            heading_vec = self_vel / np.linalg.norm(self_vel)
        else:
            heading_vec = np.array([0., 0.])

        for cover_pt, obs_center, obs_radius in self.cover_points:
            # Vectors
            vec_to_tracker = tracker_pos - cover_pt
            vec_to_obs = obs_center - cover_pt
            
            dist_tracker = np.linalg.norm(vec_to_tracker)
            if dist_tracker < 1e-3: continue
            
            dist_obs = np.linalg.norm(vec_to_obs)
            
            # Occlusion Score (Dot Product)
            # 1.0 = Obstacle directly between Cover and Tracker
            # -1.0 = Tracker between Cover and Obstacle
            occlusion = np.dot(vec_to_tracker / dist_tracker, vec_to_obs / (dist_obs + 1e-6))
            
            dist_from_tracker = np.linalg.norm(cover_pt - tracker_pos)
            dist_to_cover = np.linalg.norm(cover_pt - self_pos)
            
            # Alignment Score: Do we need to turn much?
            vec_to_cover = cover_pt - self_pos
            dist_cover_move = np.linalg.norm(vec_to_cover)
            alignment = 0.0
            if dist_cover_move > 1.0 and np.linalg.norm(heading_vec) > 0.1:
                alignment = np.dot(heading_vec, vec_to_cover / dist_cover_move)

            # Heuristic Scoring
            # Prioritize: High occlusion, Far from tracker (safe), Close to self (reachable)
            score = (occlusion * 3.0) + \
                    (dist_from_tracker * 0.01) - \
                    (dist_to_cover * 0.02) + \
                    (alignment * 0.5)
            
            # Penalty if cover point is too close to tracker (unsafe)
            if dist_from_tracker < 100.0:
                score -= 10.0

            if score > best_score:
                best_score = score
                best_cover = cover_pt
                best_obs_center = obs_center
                best_obs_radius = obs_radius

        return best_cover, best_obs_center, best_obs_radius

    def _get_orbit_point(self, self_pos, tracker_pos, obs_center, obs_radius):
        """Calculate a point tangent to the obstacle to maintain cover."""
        # Vector from obstacle to tracker
        vec_o_t = tracker_pos - obs_center
        dist_ot = np.linalg.norm(vec_o_t)
        if dist_ot < 1e-3: return self_pos
        
        # We want to be on the opposite side, slightly rotating
        # Vector from tracker to obstacle
        vec_t_o = -vec_o_t
        
        # Ideal position is "behind" obstacle relative to tracker
        # Normalized direction
        dir_t_o = vec_t_o / dist_ot
        
        # Basic hiding spot
        hiding_dist = obs_radius + self.agent_radius * 3.0
        base_hiding_spot = obs_center + dir_t_o * hiding_dist
        
        # Orbit logic: Move tangential to circle to keep 'thickest' part of obstacle between agents
        # Cross product direction
        tangent = np.array([-dir_t_o[1], dir_t_o[0]])
        
        # Choose direction that moves us away from tracker's potential motion? 
        # Or simple orbit. Let's try to maintain current angular momentum if moving.
        orbit_spot = base_hiding_spot + tangent * 20.0 # Look ahead on orbit
        
        return orbit_spot

    def _get_tangent_navigation_point(self, self_pos, target_pos, obs_center, obs_radius):
        """
        Find an intermediate waypoint to navigate around the obstacle towards the target.
        Uses tangent lines from self_pos to the 'safe circle' around the obstacle.
        """
        # Safe radius for navigation (obstacle + agent + margin)
        safe_radius = obs_radius + self.agent_radius * 1.5
        
        vec_to_c = obs_center - self_pos
        dist_to_c = np.linalg.norm(vec_to_c)
        
        # If we are inside the safe radius, just push out away from center, but bias towards target
        if dist_to_c < safe_radius:
             dir_out = -vec_to_c / (dist_to_c + 1e-6)
             # small blending with target dir?
             return self_pos + dir_out * 20.0

        # Angle of tangents
        # alpha = asin(R / D)
        if safe_radius >= dist_to_c:
             # Should be handled by 'inside' check, but numerical safety
             return target_pos 

        alpha = math.asin(safe_radius / dist_to_c)
        base_angle = math.atan2(vec_to_c[1], vec_to_c[0])
        
        # Two tangent directions
        t1_angle = base_angle + alpha
        t2_angle = base_angle - alpha
        
        # Points far out on these tangents
        t1_dir = np.array([math.cos(t1_angle), math.sin(t1_angle)])
        t2_dir = np.array([math.cos(t2_angle), math.sin(t2_angle)])
        
        # We want the one that gets us closer to target_pos (conceptually)
        # Or more robustly: The one that matches the "side" of the target relative to the obstacle center line.
        
        # Simple heuristic: Choose the tangent direction that has smaller angle to the target direction
        vec_to_target = target_pos - self_pos
        if np.linalg.norm(vec_to_target) < 1e-3: return target_pos
        
        t1_dot = np.dot(t1_dir, vec_to_target)
        t2_dot = np.dot(t2_dir, vec_to_target)
        
        best_dir = t1_dir if t1_dot > t2_dot else t2_dir
        
        return self_pos + best_dir * 50.0

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        if parsed.get("obs_type") != "target":
            return np.zeros(2, dtype=np.float32)

        sx, sy = parsed["self_x"], parsed["self_y"]
        tx, ty = parsed["tracker_x"], parsed["tracker_y"]
        self_pos = np.array([sx, sy])
        tracker_pos = np.array([tx, ty])
        self_vel = np.array([parsed.get('self_vel', 0)*math.cos(math.radians(parsed['self_heading_deg'])),
                             parsed.get('self_vel', 0)*math.sin(math.radians(parsed['self_heading_deg']))])

        # 1. Select Best Cover
        target_pos, best_obs_center, best_obs_radius = self._select_best_cover(self_pos, tracker_pos, self_vel)
        
        # 2. If no good cover, fallback to Flee
        if target_pos is None:
             # Flee directly
            desired_deg = math.degrees(math.atan2(sy - ty, sx - tx))
        else:
            # 3. Check if path is blocked by the cover obstacle itself using simple geometry
            # Distance from center to segment (self, target)
            p1 = self_pos
            p2 = target_pos
            c = best_obs_center
            r = best_obs_radius + self.agent_radius * 1.2 # Safety margin
            
            # Vector p1->p2
            d = p2 - p1
            f = p1 - c
            
            # Quadratic for line-sphere intersection: |d*t + f|^2 = r^2
            # t in [0, 1]
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            # cc = np.dot(f, f) - r*r # variable name conflict with c
            c_val = np.dot(f, f) - r*r
            
            delta = b*b - 4*a*c_val
            
            is_blocked = False
            if delta >= 0 and a > 1e-6:
                delta_sqrt = math.sqrt(delta)
                t1 = (-b - delta_sqrt) / (2*a)
                t2 = (-b + delta_sqrt) / (2*a)
                
                # Check if intersection is within segment
                if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                    is_blocked = True
            
            if is_blocked:
                # Tangent navigation
                target_pos = self._get_tangent_navigation_point(self_pos, target_pos, best_obs_center, best_obs_radius)

            dist_to_target = np.linalg.norm(target_pos - self_pos)
            
            # Simple State Machine:
            # If visible -> Go to cover
            # If hidden -> Orbit / Optimise
            
            is_visible = self._is_tracker_visible(self_pos, tracker_pos)
            
            if not is_visible and dist_to_target < 40.0:
                # Orbit Mode: We are hidden and near cover. 
                # Find the obstacle we are hiding behind
                # For simplicity, finding closest obstacle center from pre-computed list
                closest_obs = None
                min_d = float('inf')
                for _, center, radius in self.cover_points:
                    d = np.linalg.norm(center - self_pos)
                    if d < min_d:
                        min_d = d
                        closest_obs = (center, radius)
                
                if closest_obs:
                     target_pos = self._get_orbit_point(self_pos, tracker_pos, closest_obs[0], closest_obs[1])
            
            # Drive to target_pos
            desired_deg = math.degrees(math.atan2(target_pos[1] - sy, target_pos[0] - sx))

        # 4. Execute Movement
        heading_deg = parsed["self_heading_deg"]
        angle_diff = _normalize_angle(desired_deg - heading_deg)
        max_turn = float(getattr(map_config, "target_max_angular_speed", 12.0))
        angle_norm = np.clip(angle_diff / max_turn, -1.0, 1.0)
        
        # Always try to move at full speed if not extremely close
        # If very close to target and hidden, maybe slow down? 
        # For now, full speed to maintain agility.
        
        raw_action = np.array([angle_norm, 1.0], dtype=np.float32)
        safe_action = apply_hard_mask_adapter(raw_action, parsed, role="target")
        
        safe_angle_deg = safe_action[0] * max_turn
        final_heading = heading_deg + safe_angle_deg
        
        return velocity_to_acceleration(final_heading, safe_action[1], parsed, role="target")

class ZigZagTarget:
    """
    ZigZag (锯齿规避者) - 利用高角速度优势进行锯齿形规避
    
    核心策略：Target 角速度 = 12°/step，Tracker 角速度 = 6°/step
    通过高频、大幅度转向，使 Tracker 永远"追不上"Target 的航向变化。
    
    改进：
    - 距离远时减少zigzag幅度（不必要）
    - 距离近时增大zigzag幅度并增加"冒险穿梭"行为
    """
    def __init__(self):
        self.counter = 0
        self.direction = 1  # 1 = 右转, -1 = 左转
        self.switch_interval = 4  # 切换周期（步数）
        self.last_switch = 0
        self.risky_mode = False
        self.risky_timer = 0
        
    def reset(self):
        self.counter = 0
        self.direction = 1 if np.random.random() > 0.5 else -1
        self.switch_interval = np.random.randint(3, 6)
        self.last_switch = 0
        self.risky_mode = False
        self.risky_timer = 0

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        
        sx, sy = parsed['self_x'], parsed['self_y']
        tx, ty = parsed['tracker_x'], parsed['tracker_y']
        cur_heading = parsed['self_heading_deg']
        
        target_max_turn = float(getattr(map_config, 'target_max_angular_speed', 12.0))
        
        escape_angle = math.degrees(math.atan2(sy - ty, sx - tx))
        dist = math.hypot(sx - tx, sy - ty)
        
        self.counter += 1
        
        # ============ 距离自适应 ZigZag 幅度 ============
        # 远距离: 小幅度或无zigzag (直接逃跑更高效)
        # 近距离: 大幅度zigzag
        # 极近距离: 冒险穿梭 (从tracker身边擦过)
        
        CLOSE_THRESHOLD = 80.0
        MID_THRESHOLD = 180.0
        FAR_THRESHOLD = 300.0
        
        if dist < CLOSE_THRESHOLD:
            # 极近: 冒险穿梭模式
            # 随机触发或当方向有利时
            if not self.risky_mode and np.random.random() < 0.15:
                self.risky_mode = True
                self.risky_timer = 8  # 持续8步
            
            if self.risky_mode:
                self.risky_timer -= 1
                if self.risky_timer <= 0:
                    self.risky_mode = False
                
                # 冒险: 朝向tracker侧方穿过 (垂直于逃跑方向)
                # 这会迫使tracker急转弯
                tangent_angle = escape_angle + 90 * self.direction
                desired_deg = tangent_angle
            else:
                # 高强度zigzag
                zigzag_amplitude = 75.0
                zigzag_offset = self.direction * zigzag_amplitude
                desired_deg = escape_angle + zigzag_offset
                
        elif dist < MID_THRESHOLD:
            # 中距离: 正常zigzag
            zigzag_amplitude = 50.0
            zigzag_offset = self.direction * zigzag_amplitude
            desired_deg = escape_angle + zigzag_offset
            
        elif dist < FAR_THRESHOLD:
            # 远距离: 轻微zigzag
            zigzag_amplitude = 25.0
            zigzag_offset = self.direction * zigzag_amplitude
            desired_deg = escape_angle + zigzag_offset
            
        else:
            # 极远: 直接逃跑，不zigzag
            desired_deg = escape_angle
        
        # ============ 切换方向逻辑 ============
        if dist < MID_THRESHOLD:
            effective_interval = max(2, self.switch_interval - 1)
        else:
            effective_interval = self.switch_interval + 2
        
        if self.counter - self.last_switch >= effective_interval:
            self.direction *= -1
            self.last_switch = self.counter
            self.switch_interval = np.random.randint(3, 6)
        
        # ============ 速度和输出 ============
        angle_diff = _normalize_angle(desired_deg - cur_heading)
        
        if abs(angle_diff) > target_max_turn:
            angle_norm = 1.0 if angle_diff > 0 else -1.0
        else:
            angle_norm = np.clip(angle_diff / target_max_turn, -1.0, 1.0)
        
        speed_norm = 1.0
        
        action = np.array([angle_norm, speed_norm], dtype=np.float32)
        safe_action = apply_hard_mask_adapter(action, parsed, role='target')
        
        safe_angle_deg = safe_action[0] * target_max_turn
        final_heading = cur_heading + safe_angle_deg
        return velocity_to_acceleration(final_heading, safe_action[1], parsed, role='target')

class OrbiterTarget:
    """
    Orbiter (轨道逃逸者)
    切向运动策略 - 围绕 Tracker 做圆周运动。
    """
    def __init__(self):
        self.orbit_radius = 170.0
        self.inner_radius = 120.0
        self.outer_radius = 230.0
        self.panic_radius = 90.0
        self.clockwise = True
        self.blocked_counter = 0
        self.direction_cooldown = 0
        self.block_switch_threshold = 12
        self.min_dir_hold = 18
        self.bias_angle_deg = 25.0
        self.aggression = 0.35
        self.speed_base = 0.9
        
    def reset(self):
        self.clockwise = (np.random.random() > 0.5)
        self.blocked_counter = 0
        self.direction_cooldown = self.min_dir_hold

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        
        sx, sy = parsed['self_x'], parsed['self_y']
        tx, ty = parsed['tracker_x'], parsed['tracker_y']
        
        vec_x = sx - tx
        vec_y = sy - ty
        dist = math.hypot(vec_x, vec_y)
        angle_to_me = math.atan2(vec_y, vec_x)
        
        rel_vec = np.array([vec_x, vec_y], dtype=np.float32)
        rel_dir = rel_vec / max(dist, 1e-6)

        # 1) Panic: sprint straight away if the tracker is very close
        desired_deg = math.degrees(angle_to_me)
        speed_norm = 1.0
        bias_deg = 0.0

        if dist >= self.panic_radius:
            # Tangential direction (orbit) with radius band correction
            if self.clockwise:
                tangential = np.array([rel_dir[1], -rel_dir[0]])
            else:
                tangential = np.array([-rel_dir[1], rel_dir[0]])

            band = max(1.0, self.outer_radius - self.inner_radius)
            radial_error = dist - self.orbit_radius
            radial_gain = np.clip(-radial_error / band, -1.5, 1.5)
            radial_term = radial_gain * rel_dir

            desired_vec = tangential + 0.8 * radial_term

            if dist > self.outer_radius:
                desired_vec += -self.aggression * rel_dir

            # Obstacle bias using radar projected to heading frame
            radar = parsed.get('radar', [])
            num_rays = len(radar)
            if num_rays > 0:
                heading_idx = int(round(((parsed['self_heading_deg'] % 360.0) / 360.0) * num_rays)) % num_rays
                left_idx = (heading_idx + num_rays // 4) % num_rays
                right_idx = (heading_idx - num_rays // 4) % num_rays

                def _ray_norm(idx):
                    return (float(radar[idx]) + 1.0) * 0.5

                forward = _ray_norm(heading_idx)
                left = _ray_norm(left_idx)
                right = _ray_norm(right_idx)

                if forward < 0.35:
                    if left - right > 0.05:
                        bias_deg = self.bias_angle_deg
                    elif right - left > 0.05:
                        bias_deg = -self.bias_angle_deg

            desired_rad = math.atan2(desired_vec[1], desired_vec[0])
            desired_deg = math.degrees(desired_rad) + bias_deg
            speed_norm = np.clip(self.speed_base + 0.3 * min(1.0, abs(radial_gain)), 0.6, 1.0)
            if dist < self.inner_radius:
                speed_norm = 1.0

        # 2) Execution
        cur_heading = parsed['self_heading_deg']
        angle_diff = _normalize_angle(desired_deg - cur_heading)
        max_turn = float(getattr(map_config, 'target_max_angular_speed', 12.0))
        angle_norm = np.clip(angle_diff / max_turn, -1.0, 1.0)

        action = np.array([angle_norm, speed_norm], dtype=np.float32)
        safe_action = apply_hard_mask_adapter(action, parsed, role='target')

        blocked = abs(safe_action[0] - angle_norm) > 0.55
        self.blocked_counter = self.blocked_counter + 1 if blocked else max(0, self.blocked_counter - 1)
        if self.direction_cooldown > 0:
            self.direction_cooldown -= 1

        if self.blocked_counter > self.block_switch_threshold and self.direction_cooldown <= 0:
            self.clockwise = not self.clockwise
            self.blocked_counter = 0
            self.direction_cooldown = self.min_dir_hold

        safe_angle_deg = safe_action[0] * max_turn
        final_heading = cur_heading + safe_angle_deg
        return velocity_to_acceleration(final_heading, safe_action[1], parsed, role='target')

class GreedyTarget:
    """
    Pure Escape Greedy Target Policy with Hard Action Mask.
    - Moves directly away from the tracker.
    - Uses hard mask to avoid immediate collisions.
    """
    def __init__(self):
        pass
    
    def reset(self):
        pass

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        if parsed.get('obs_type') != 'target':
            raise ValueError("GreedyTarget requires target observation")

        # 1. Calculate direction away from tracker
        sx, sy = parsed['self_x'], parsed['self_y']
        tx, ty = parsed['tracker_x'], parsed['tracker_y']
        
        vec_x = sx - tx
        vec_y = sy - ty
        dist = math.hypot(vec_x, vec_y)
        
        if dist < 1e-3:
            # If on top of each other, pick random direction or current heading
            desired_deg = parsed['self_heading_deg']
        else:
            desired_yaw = math.atan2(vec_y, vec_x)
            desired_deg = math.degrees(desired_yaw)
            
        # 2. Convert to relative action
        cur_heading = parsed['self_heading_deg']
        angle_diff = _normalize_angle(desired_deg - cur_heading)
        
        max_turn = float(getattr(map_config, 'target_max_angular_speed', 12.0))
        angle_norm = np.clip(angle_diff / max_turn, -1.0, 1.0)
        speed_norm = 1.0 # Always run at max speed
        
        action = np.array([angle_norm, speed_norm], dtype=np.float32)
        
        # 3. Apply Hard Mask
        # We need to adapt the action using the same helper we assumed earlier or use cbf_controller directly if imported.
        # apply_hard_mask_adapter is defined above.
        
        # Note: apply_hard_mask expects 'role' argument to check specific radii/speeds if needed in cbf_controller.
        safe_action = apply_hard_mask_adapter(action, parsed, role='target')
        
        # 4. Convert control to acceleration (Environment expects acceleration often? Or control?)
        # Tracker policies return control (angle, speed).
        # But `velocity_to_acceleration` was used by previous policies.
        # Let's check `target_runner.py` or default env usage.
        # Usually rule policies return 'action' which matches env action space.
        # Env action space is acceleration if configured? Or velocity?
        # `env.py` has `_control_to_physical` which handles acceleration conversion if needed.
        # BUT `velocity_to_acceleration` helper suggests we want to output ACCELERATION commands.
        # If the environment expects acceleration (which it does via _control_to_physical -> agent_move_accel),
        # then we must convert our desired velocity (heading/speed) to acceleration.
        
        # Recover safe desired attributes from safe_action
        safe_angle_norm = safe_action[0]
        safe_speed_norm = safe_action[1]
        
        safe_angle_deg = safe_angle_norm * max_turn
        safe_desired_heading = cur_heading + safe_angle_deg
        
        return velocity_to_acceleration(safe_desired_heading, safe_speed_norm, parsed, role='target')

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
        expected_dim = 11 + EnvParameters.RADAR_RAYS
        if obs.shape[0] != expected_dim and obs.shape[0] != 75:
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

TRACKER_POLICY_REGISTRY = {
    "CBF": CBFTracker,
    "PurePursuit": PurePursuitTracker
}
if not CBF_AVAILABLE:
    del TRACKER_POLICY_REGISTRY["CBF"]


TARGET_POLICY_REGISTRY = {
    "Greedy": GreedyTarget,
    "CoverSeeker": CoverSeekerTarget,
    "ZigZag": ZigZagTarget,
    "Orbiter": OrbiterTarget,
}