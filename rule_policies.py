"""
Rule-based policies for tracker and target agents.
This module provides baseline policies that can be used for evaluation and testing.
"""
import math
import numpy as np
import map_config
import env_lib
from map_config import EnvParameters

# ============================================================================
# Local Helper Functions (Policy Control Logic)
# ============================================================================

HARD_MASK_SAFETY_MULTIPLIER = 3.5
HARD_MASK_CHECK_WINDOW = 2
HARD_MASK_EMERGENCY_BRAKE = True

def _normalize_angle(angle_deg: float):
    """Normalize angle to [-180, 180] range."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return float(angle_deg)

def apply_hard_mask(action, radar, current_heading_deg, role='tracker', safety_dist=None):
    """
    Hard Mask Function: Force collision avoidance based on radar.
    Moved from env_lib.py to rule_policies.py as it is control logic.
    """
    if safety_dist is None:
        safety_dist = float(getattr(map_config, 'agent_radius', 8.0)) * HARD_MASK_SAFETY_MULTIPLIER

    # Ensure action is array-like and extract angle_norm and speed_norm
    action_arr = np.asarray(action, dtype=np.float32).flatten()
    if action_arr.size != 2:
        # If action is wrong shape, return as-is (defensive)
        return action
    
    angle_norm = float(action_arr[0])
    speed_norm = float(action_arr[1])

    # Max turn capabilities
    if role == 'tracker':
        max_turn = float(getattr(map_config, 'tracker_max_angular_speed', 10.0))
    else:
        max_turn = float(getattr(map_config, 'target_max_angular_speed', 12.0))
        
    angle_delta = angle_norm * max_turn
    target_heading = _normalize_angle(current_heading_deg + angle_delta)
    
    # Convert radar to numpy array
    radar_arr = np.asarray(radar, dtype=np.float32).flatten()
    if len(radar_arr) == 0:
        return np.array([angle_norm, speed_norm], dtype=np.float32)

    num_rays = len(radar_arr)
    angle_step = 360.0 / num_rays
    
    # Calculate target heading index
    th_360 = target_heading % 360.0
    center_idx = int(round(th_360 / angle_step)) % num_rays
    
    max_range = float(getattr(map_config, 'FOV_RANGE', 250.0))
    dists = (radar_arr + 1.0) * 0.5 * max_range
    
    # Check safety
    is_safe = True
    for i in range(-HARD_MASK_CHECK_WINDOW, HARD_MASK_CHECK_WINDOW + 1):
        idx = (center_idx + i) % num_rays
        if dists[idx] <= safety_dist:
            is_safe = False
            break
            
    if is_safe:
        return np.array([angle_norm, speed_norm], dtype=np.float32)
        
    # Search for safe direction
    safe_indices = []
    for i in range(num_rays):
        window_safe = True
        for w in range(-HARD_MASK_CHECK_WINDOW, HARD_MASK_CHECK_WINDOW + 1):
            if dists[(i + w) % num_rays] <= safety_dist:
                window_safe = False
                break
        if window_safe:
            safe_indices.append(i)
    
    if not safe_indices:
        if HARD_MASK_EMERGENCY_BRAKE:
            # Find furthest direction (emergency escape)
            best_idx = -1
            max_avg_dist = -1.0
            for i in range(num_rays):
                avg_dist = 0
                for w in range(-HARD_MASK_CHECK_WINDOW, HARD_MASK_CHECK_WINDOW + 1):
                    avg_dist += dists[(i + w) % num_rays]
                if avg_dist > max_avg_dist:
                    max_avg_dist = avg_dist
                    best_idx = i
            
            if best_idx != -1:
                best_ray_angle = best_idx * angle_step
                needed_turn = _normalize_angle(best_ray_angle - current_heading_deg)
                clamped_turn = np.clip(needed_turn, -max_turn, max_turn)
                new_angle_norm = clamped_turn / (max_turn + 1e-6)
                return np.array([new_angle_norm, -1.0], dtype=np.float32)
        return np.array([angle_norm, -1.0], dtype=np.float32)

    # Find closest safe angle to desired
    best_idx = -1
    min_diff = 1000.0
    
    for idx in safe_indices:
        ray_angle = idx * angle_step
        diff = abs(_normalize_angle(ray_angle - target_heading))
        if diff < min_diff:
            min_diff = diff
            best_idx = idx
            
    best_ray_angle = best_idx * angle_step
    needed_turn = _normalize_angle(best_ray_angle - current_heading_deg)
    clamped_turn = np.clip(needed_turn, -max_turn, max_turn)
    new_angle_norm = clamped_turn / (max_turn + 1e-6)
    
    new_speed_norm = speed_norm
    return np.array([new_angle_norm, new_speed_norm], dtype=np.float32)


# ============================================================================
# Tracker Policies
# ============================================================================

from cbf_controller import CBFTracker

TRACKER_POLICY_REGISTRY = {
    "CBF": CBFTracker,
}


# --- A* Pathfinding (Optimized) ---
# Cache for path results (cleared on map change)
_PATH_CACHE = {}
_PATH_CACHE_MAX_SIZE = 100

def _get_path_cache_key(start_pos, goal_pos, padding):
    """Generate cache key for paths (quantized to grid cells)"""
    if not env_lib._occ_available():
        return None
    cell = env_lib._OCC_CELL
    return (
        int(start_pos[0] / cell), int(start_pos[1] / cell),
        int(goal_pos[0] / cell), int(goal_pos[1] / cell),
        int(padding / cell)
    )

def clear_path_cache():
    """Clear path cache (call when obstacles change)"""
    global _PATH_CACHE
    _PATH_CACHE.clear()

def heuristic(a, b):
    # Use Chebyshev distance for 8-direction movement
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return max(dx, dy) + 0.414 * min(dx, dy)  # Octile distance

def find_path(start_pos, goal_pos, padding=0.0):
    """
    Optimized A* Pathfinding with caching, 8-direction movement, and early termination.
    """
    if not env_lib._occ_available():
        return [goal_pos]
    
    # Check cache
    cache_key = _get_path_cache_key(start_pos, goal_pos, padding)
    if cache_key and cache_key in _PATH_CACHE:
        return _PATH_CACHE[cache_key]

    # Use env_lib utilities
    sx, sy, _, _ = env_lib._world_to_cell(start_pos[0], start_pos[1], env_lib._OCC_CELL)
    gx, gy, _, _ = env_lib._world_to_cell(goal_pos[0], goal_pos[1], env_lib._OCC_CELL)
    
    start_node = (sx, sy)
    goal_node = (gx, gy)
    
    grid_nx, grid_ny = env_lib._OCC_GRID.shape[1], env_lib._OCC_GRID.shape[0]
    
    if not env_lib._clip_idx(sx, sy, grid_nx, grid_ny):
        return [goal_pos]

    import heapq
    
    frontier = []
    heapq.heappush(frontier, (0, start_node))
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}
    
    pad_cells = int(math.ceil(float(padding) / float(env_lib._OCC_CELL)))
    
    found = False
    current = start_node
    
    # Balanced max nodes (was 5000, now 1000)
    MAX_NODES = 1000
    
    # 8-direction movements with proper costs
    DIRECTIONS = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),  # Cardinal
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)  # Diagonal
    ]
    
    while frontier and len(cost_so_far) < MAX_NODES:
        _, current = heapq.heappop(frontier)
        
        if current == goal_node:
            found = True
            break
        
        x, y = current
        for dx, dy, move_cost in DIRECTIONS:
            nx_ = x + dx
            ny_ = y + dy
            next_node = (nx_, ny_)
            
            if not (0 <= nx_ < grid_nx and 0 <= ny_ < grid_ny):
                continue
            
            if env_lib._occ_any_with_pad(nx_, ny_, pad_cells):
                continue
            
            new_cost = cost_so_far[current] + move_cost
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current
    
    # Reconstruct path
    if not found and current == start_node:
        result = [goal_pos]
    else:
        path = []
        curr = goal_node if found else current
        while curr is not None and curr != start_node:
            cx = (curr[0] + 0.5) * env_lib._OCC_CELL
            cy = (curr[1] + 0.5) * env_lib._OCC_CELL
            path.append((cx, cy))
            curr = came_from.get(curr)
        
        path.reverse()
        
        # Simplify path (skip every 3rd point)
        if len(path) > 3:
            result = path[::3]
            if result[-1] != path[-1]:
                result.append(path[-1])
        else:
            result = path if path else [goal_pos]
    
    # Cache result
    if cache_key and len(_PATH_CACHE) < _PATH_CACHE_MAX_SIZE:
        _PATH_CACHE[cache_key] = result
    
    return result


# ============================================================================
# Utility Functions
# ============================================================================

# _normalize_angle is already defined at top of file (line 19)

def apply_hard_mask_adapter(action, obs_dict, role='tracker', safety_dist=None):
    """
    Adapter for apply_hard_mask to work with obs_dict. Adds speed-based inflation to keep clearance.
    """
    radar = obs_dict.get('radar', [])
    heading = obs_dict.get('self_heading_deg', 0.0)
    if safety_dist is None:
        base = float(getattr(map_config, 'agent_radius', 8.0)) * HARD_MASK_SAFETY_MULTIPLIER
        speed_norm = float(obs_dict.get('self_vel', 0.0))
        speed_frac = (speed_norm + 1.0) * 0.5
        safety_dist = base * (1.0 + 0.5 * speed_frac)
    return apply_hard_mask(action, radar, heading, role, safety_dist)

# velocity_to_acceleration removed (switched to velocity control)


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
                # env_lib is imported at top of file
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
        # Check ray from tracker (env_lib is imported at top of file)
        ray_dist = env_lib.ray_distance_grid(
            (tracker_pos[0], tracker_pos[1]), 
            angle, 
            dist, 
            padding=0.0 # Line of sight check
        )
        return ray_dist >= (dist - 5.0) # Tolerance

    def _select_best_cover(self, self_pos, tracker_pos, self_vel):
        """Select cover point that places obstacle between self and tracker (vectorized)"""
        if not self.cover_points:
            return None, None, 0.0
        
        n = len(self.cover_points)
        
        # Pre-allocate arrays
        cover_pts = np.array([cp[0] for cp in self.cover_points])  # (n, 2)
        obs_centers = np.array([cp[1] for cp in self.cover_points])  # (n, 2)
        obs_radii = np.array([cp[2] for cp in self.cover_points])  # (n,)
        
        # Vectorized distance calculations
        vec_to_tracker = tracker_pos - cover_pts  # (n, 2)
        vec_to_obs = obs_centers - cover_pts  # (n, 2)
        vec_to_cover = cover_pts - self_pos  # (n, 2)
        vec_from_tracker = cover_pts - tracker_pos  # (n, 2)
        
        # Use np.einsum for fast norm-squared computation (avoid sqrt when possible)
        dist_tracker_sq = np.einsum('ij,ij->i', vec_to_tracker, vec_to_tracker)  # (n,)
        dist_tracker = np.sqrt(dist_tracker_sq + 1e-6)
        
        dist_obs = np.sqrt(np.einsum('ij,ij->i', vec_to_obs, vec_to_obs) + 1e-6)
        dist_to_cover = np.sqrt(np.einsum('ij,ij->i', vec_to_cover, vec_to_cover) + 1e-6)
        dist_from_tracker = np.sqrt(np.einsum('ij,ij->i', vec_from_tracker, vec_from_tracker) + 1e-6)
        
        # Normalized vectors
        vec_to_tracker_norm = vec_to_tracker / dist_tracker[:, np.newaxis]
        vec_to_obs_norm = vec_to_obs / dist_obs[:, np.newaxis]
        
        # Occlusion score (dot product)
        occlusion = np.einsum('ij,ij->i', vec_to_tracker_norm, vec_to_obs_norm)
        
        # Alignment score
        vel_norm = np.linalg.norm(self_vel)
        if vel_norm > 0.1:
            heading_vec = self_vel / vel_norm
            vec_to_cover_norm = vec_to_cover / dist_to_cover[:, np.newaxis]
            alignment = np.dot(vec_to_cover_norm, heading_vec)
        else:
            alignment = np.zeros(n)
        
        # Heuristic scoring (vectorized)
        scores = (occlusion * 3.0) + \
                 (dist_from_tracker * 0.01) - \
                 (dist_to_cover * 0.02) + \
                 (alignment * 0.5)
        
        # Penalty for too-close points
        scores = np.where(dist_from_tracker < 100.0, scores - 10.0, scores)
        
        # Filter invalid (too close to tracker)
        valid_mask = dist_tracker_sq > 1e-6
        scores = np.where(valid_mask, scores, -np.inf)
        
        best_idx = np.argmax(scores)
        if scores[best_idx] == -np.inf:
            return None, None, 0.0
        
        return cover_pts[best_idx], obs_centers[best_idx], obs_radii[best_idx]

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
                # A* Pathfinding Navigation
                # Tangent navigation was too simple. Use A* to find path around obstacle.
                path = find_path(self_pos, target_pos, padding=self.agent_radius * 1.5)
                if len(path) > 1:
                    # Target the next waypoint in the path
                    target_pos = np.array(path[1]) 
                else:
                    # If path failed or reached, just go to target
                    target_pos = np.array(path[0])

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
        
        # Output direct velocity command: [delta_deg, speed_fraction]
        return np.array([safe_angle_deg, safe_action[1]], dtype=np.float32)

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

        # Output direct velocity command: [delta_deg, speed_fraction]
        return np.array([safe_angle_deg, safe_action[1]], dtype=np.float32)

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
        
        # Output direct velocity command: [delta_deg, speed_fraction]
        return np.array([safe_angle_deg, safe_action[1]], dtype=np.float32)

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
        safe_action = apply_hard_mask_adapter(action, parsed, role='target')
        
        # 4. Convert control to acceleration (Environment expects acceleration often? Or control?)
        
        # Recover safe desired attributes from safe_action
        safe_angle_norm = safe_action[0]
        safe_speed_norm = safe_action[1]
        
        safe_angle_deg = safe_angle_norm * max_turn
        
        # Output direct velocity command: [delta_deg, speed_fraction]
        return np.array([safe_angle_deg, safe_speed_norm], dtype=np.float32)


TARGET_POLICY_REGISTRY = {
    "Greedy": GreedyTarget,
    "CoverSeeker": CoverSeekerTarget,
    "ZigZag": ZigZagTarget,
    "Orbiter": OrbiterTarget,
}