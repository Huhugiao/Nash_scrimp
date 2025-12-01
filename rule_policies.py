"""
Rule-based policies for tracker and target agents.
This module provides baseline policies that can be used for evaluation and testing.
"""
import math
import numpy as np
import heapq
import map_config
from lstm.model_lstm import Model  # 修正导入路径
import env_lib
from cbf_controller import CBFTracker, apply_hard_mask
from map_config import EnvParameters


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
    return apply_hard_mask(action, radar, heading, role, safety_dist)


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
    if obs.shape[0] == 24:
        # Target观测解析（全局坐标，全知视角）
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
            'radar': obs[8:24].astype(np.float64),
            'self_x': (float(obs[0]) + 1.0) / 2.0 * map_config.width,
            'self_y': (float(obs[1]) + 1.0) / 2.0 * map_config.height,
            'self_heading_deg': (float(obs[2]) + 1.0) * 180.0,
            'tracker_x': (float(obs[3]) + 1.0) / 2.0 * map_config.width,
            'tracker_y': (float(obs[4]) + 1.0) / 2.0 * map_config.height,
            'tracker_heading_deg': (float(obs[5]) + 1.0) * 180.0,
        }
    elif obs.shape[0] == 27:
        # Tracker观测解析（相对观测，受FOV限制，360度雷达）
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
            'radar': obs[11:27].astype(np.float64)
        }
    else:
        raise ValueError(f"Unexpected observation dimension: {obs.shape[0]}, expected 24 (target) or 27 (tracker)")


# ============================================================================
# Target Policy
# ============================================================================

class GreedyTarget:
    """
    Greedy target policy: 始终朝向tracker的反方向移动
    - 方向：追捕者->逃逸方向量的反方向
    - 速度：始终全速
    - 避障：完全依赖环境的硬掩码（碰撞检测），不做额外避障规划
    """
    
    def __init__(self):
        pass
    
    def reset(self):
        """Greedy is stateless, nothing to reset."""
        pass
    
    def get_action(self, observation):
        parsed = _parse_observation(observation)
        if parsed.get('obs_type') != 'target':
            raise ValueError("Greedy target policy requires target observation")
        
        cur = np.array([parsed['self_x'], parsed['self_y']], dtype=np.float64)
        tracker = np.array([parsed['tracker_x'], parsed['tracker_y']], dtype=np.float64)
        
        to_tracker = tracker - cur
        dist_tracker = float(np.linalg.norm(to_tracker))
        
        if dist_tracker < 1e-6:
            self_head_deg = float(parsed['self_heading_deg'])
            desired_deg = self_head_deg
        else:
            flee_dir = -(to_tracker / dist_tracker)
            desired_deg = math.degrees(math.atan2(flee_dir[1], flee_dir[0]))
        
        self_head_deg = float(parsed['self_heading_deg'])
        angle_delta = _normalize_angle(desired_deg - self_head_deg)
        
        max_turn = float(getattr(map_config, "target_max_turn_deg", 10.0))
        angle_out = float(np.clip(angle_delta, -max_turn, max_turn))
        speed_out = 1.0
        
        # 应用硬掩码避障
        raw_action = Model.to_normalized_action((angle_out, speed_out))
        return apply_hard_mask_adapter(raw_action, parsed, role='target')


class HidingAStarTarget:
    """
    A* Target Policy with Visibility Cost (Hiding).
    Uses persistent waypoints that balance distance from tracker and visibility.
    Only recomputes path when waypoint is reached.
    """
    def __init__(self):
        self.resolution = 32.0  # Grid resolution in pixels (coarse for performance)
        self.rr = map_config.agent_radius
        self.margin = 20.0
        self.visibility_cost = 50.0  # Penalty for being visible
        self.motion = self._get_motion_model()
        
        # Map bounds
        self.min_x = 0
        self.min_y = 0
        self.max_x = map_config.width
        self.max_y = map_config.height
        self.x_width = int(round((self.max_x - self.min_x) / self.resolution))
        self.y_width = int(round((self.max_y - self.min_y) / self.resolution))
        
        # Waypoint persistence
        self.current_waypoint = None
        self.waypoint_threshold = 15.0  # Distance to consider waypoint reached

    def reset(self):
        self.current_waypoint = None

    class Node:
        def __init__(self, x_ind, y_ind, cost, parent_ind):
            self.x_ind = x_ind
            self.y_ind = y_ind
            self.cost = cost
            self.parent_ind = parent_ind
        
        def __lt__(self, other):
            return self.cost < other.cost

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        if parsed.get('obs_type') != 'target':
            raise ValueError("HidingAStar requires target observation")

        sx, sy = parsed['self_x'], parsed['self_y']
        tx, ty = parsed['tracker_x'], parsed['tracker_y']
        tracker_pos = (tx, ty)

        # Check if we need to compute a new waypoint
        if self.current_waypoint is None or self._reached_waypoint(sx, sy):
            self.current_waypoint = self._compute_strategic_waypoint(sx, sy, tracker_pos)

        gx, gy = self.current_waypoint

        # 2. Run A* Planning to waypoint
        rx, ry = self.planning(sx, sy, gx, gy, tracker_pos)

        # 3. Follow path
        if len(rx) >= 2:
            # rx is [goal ... start], so rx[-2] is the next step
            next_x = rx[-2]
            next_y = ry[-2]
            desired_yaw = math.atan2(next_y - sy, next_x - sx)
        else:
            # Fallback: Greedy flee if A* fails
            dx = sx - tx
            dy = sy - ty
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                desired_yaw = math.radians(parsed['self_heading_deg'])
            else:
                flee_dir = np.array([dx, dy]) / dist
                desired_yaw = math.atan2(flee_dir[1], flee_dir[0])

        # 4. Convert to action
        current_yaw_deg = parsed['self_heading_deg']
        desired_deg = math.degrees(desired_yaw)
        angle_delta = _normalize_angle(desired_deg - current_yaw_deg)

        max_turn = float(getattr(map_config, "target_max_turn_deg", 10.0))
        angle_out = float(np.clip(angle_delta, -max_turn, max_turn))
        speed_out = 1.0

        raw_action = Model.to_normalized_action((angle_out, speed_out))
        
        # Apply hard mask to prevent collision
        return apply_hard_mask_adapter(raw_action, parsed, role='target')

    def _reached_waypoint(self, x, y):
        """Check if current position is close enough to waypoint"""
        if self.current_waypoint is None:
            return True
        
        wx, wy = self.current_waypoint
        dist = math.hypot(x - wx, y - wy)
        return dist < self.waypoint_threshold

    def _compute_strategic_waypoint(self, sx, sy, tracker_pos):
        """
        Compute a strategic waypoint that balances:
        1. Distance from tracker (prefer far)
        2. Low visibility (prefer hidden positions)
        3. Reachability (not in obstacles)
        """
        tx, ty = tracker_pos
        
        # Sample candidate positions on a coarse grid
        candidates = []
        step = 80  # Sample every 80 pixels
        
        for x in range(int(self.margin), int(self.max_x - self.margin), step):
            for y in range(int(self.margin), int(self.max_y - self.margin), step):
                # Skip if in obstacle
                if env_lib.is_point_blocked(x, y, padding=self.rr):
                    continue
                
                # Calculate distance from tracker
                dist_to_tracker = math.hypot(x - tx, y - ty)
                
                # Check visibility
                is_visible = self._is_visible_to_tracker(tracker_pos, (x, y))
                
                # Calculate distance from self (prefer not too far for reachability)
                dist_from_self = math.hypot(x - sx, y - sy)
                
                # Scoring: 
                # - High score for far from tracker
                # - Penalty for being visible
                # - Slight penalty for being too far from self
                score = dist_to_tracker * 1.0  # Prioritize distance
                if is_visible:
                    score -= self.visibility_cost
                score -= dist_from_self * 0.1  # Small penalty for reachability
                
                candidates.append((x, y, score))
        
        if not candidates:
            # Fallback: flee in opposite direction
            dx = sx - tx
            dy = sy - ty
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                return (sx + 100, sy)
            scale = 200.0
            gx = np.clip(sx + (dx / dist) * scale, self.margin, self.max_x - self.margin)
            gy = np.clip(sy + (dy / dist) * scale, self.margin, self.max_y - self.margin)
            return (float(gx), float(gy))
        
        # Select best candidate
        candidates.sort(key=lambda c: c[2], reverse=True)
        best_x, best_y, _ = candidates[0]
        
        return (float(best_x), float(best_y))

    def planning(self, sx, sy, gx, gy, tracker_pos):
        """
        A* path search with visibility cost.
        """
        start_node = self.Node(self._calc_xy_index(sx, self.min_x),
                               self._calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self._calc_xy_index(gx, self.min_x),
                              self._calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set = dict()
        closed_set = dict()
        priority_queue = []

        # Add start node
        open_set[self._calc_grid_index(start_node)] = start_node
        heapq.heappush(priority_queue, (0, self._calc_grid_index(start_node)))

        while True:
            if not priority_queue:
                # Open set is empty, no path found
                break

            _, c_id = heapq.heappop(priority_queue)
            
            if c_id in closed_set:
                continue
            
            if c_id not in open_set:
                continue

            current = open_set[c_id]
            closed_set[c_id] = current
            del open_set[c_id]

            # Check if goal reached (approximate)
            if current.x_ind == goal_node.x_ind and current.y_ind == goal_node.y_ind:
                goal_node.parent_ind = current.parent_ind
                goal_node.cost = current.cost
                break

            # Expand neighbors
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x_ind + self.motion[i][0],
                                 current.y_ind + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self._calc_grid_index(node)

                # 1. Check bounds and static collision
                if not self._verify_node(node):
                    continue

                if n_id in closed_set:
                    continue
                
                # 2. Add Visibility Cost
                # Calculate world position of the node
                node_wx = self._calc_grid_position(node.x_ind, self.min_x)
                node_wy = self._calc_grid_position(node.y_ind, self.min_y)
                
                vis_cost = 0.0
                if self._is_visible_to_tracker(tracker_pos, (node_wx, node_wy)):
                    vis_cost = self.visibility_cost
                
                # Update node cost with visibility penalty
                node.cost += vis_cost

                if n_id not in open_set:
                    open_set[n_id] = node
                    # f(n) = g(n) + h(n)
                    priority = node.cost + self._calc_heuristic(node, goal_node)
                    heapq.heappush(priority_queue, (priority, n_id))
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node
                        priority = node.cost + self._calc_heuristic(node, goal_node)
                        heapq.heappush(priority_queue, (priority, n_id))

        rx, ry = self._calc_final_path(goal_node, closed_set)
        return rx, ry

    def _calc_final_path(self, goal_node, closed_set):
        rx, ry = [self._calc_grid_position(goal_node.x_ind, self.min_x)], \
                 [self._calc_grid_position(goal_node.y_ind, self.min_y)]
        parent_index = goal_node.parent_ind
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self._calc_grid_position(n.x_ind, self.min_x))
            ry.append(self._calc_grid_position(n.y_ind, self.min_y))
            parent_index = n.parent_ind
        return rx, ry

    def _calc_heuristic(self, n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x_ind - n2.x_ind, n1.y_ind - n2.y_ind)
        return d

    def _calc_grid_position(self, index, min_position):
        return index * self.resolution + min_position

    def _calc_xy_index(self, position, min_pos):
        return int(round((position - min_pos) / self.resolution))

    def _calc_grid_index(self, node):
        return (node.y_ind - self.min_y) * self.x_width + (node.x_ind - self.min_x)

    def _verify_node(self, node):
        px = self._calc_grid_position(node.x_ind, self.min_x)
        py = self._calc_grid_position(node.y_ind, self.min_y)

        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False

        # Static collision check using env_lib
        # Use a slightly larger padding to ensure path safety
        if env_lib.is_point_blocked(px, py, padding=self.rr):
            return False

        return True
    
    def _is_visible_to_tracker(self, tracker_pos, node_pos):
        """Check if node is visible from tracker using raycasting."""
        tx, ty = tracker_pos
        nx, ny = node_pos
        
        dx = nx - tx
        dy = ny - ty
        dist = math.hypot(dx, dy)
        
        # Optimization: If too far, assume not relevant
        if dist > EnvParameters.FOV_RANGE * 1.2:
            return False
            
        angle = math.atan2(dy, dx)
        
        # Raycast: check if ray hits obstacle before reaching node
        hit_dist = env_lib.ray_distance_grid((tx, ty), angle, dist + 5.0, padding=0.0)
        
        # If hit_dist is close to dist, line of sight is clear (visible)
        if hit_dist >= dist - 5.0:
            return True
        return False

    @staticmethod
    def _get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion


class APFTarget:
    """
    Artificial Potential Field (APF) Target Policy.
    Uses repulsive forces from the tracker and obstacles to determine direction.
    """
    def __init__(self):
        self.repulsive_gain_tracker = 5.0
        self.repulsive_gain_obs = 2.0
        self.obs_influence_dist = 80.0  # pixels

    def reset(self):
        pass

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        # 1. Repulsive force from Tracker
        cur_pos = np.array([parsed['self_x'], parsed['self_y']])
        tracker_pos = np.array([parsed['tracker_x'], parsed['tracker_y']])
        vec_to_tracker = tracker_pos - cur_pos
        dist_tracker = np.linalg.norm(vec_to_tracker)

        if dist_tracker < 1e-3:
            f_tracker = np.zeros(2)
        else:
            # Direction away from tracker
            dir_away = -vec_to_tracker / dist_tracker
            f_tracker = dir_away * self.repulsive_gain_tracker

        # 2. Repulsive force from Obstacles (Radar)
        f_obs = np.zeros(2)
        radar = parsed['radar']  # 16 rays, normalized [-1, 1]
        max_range = EnvParameters.FOV_RANGE
        num_rays = len(radar)

        for i in range(num_rays):
            # radar value is normalized: (dist / max_range) * 2 - 1
            norm_val = radar[i]
            dist = (norm_val + 1.0) / 2.0 * max_range

            if dist < self.obs_influence_dist:
                # Obstacle is close
                # For target, radar angles are global [0, 2pi)
                angle = 2 * math.pi * i / num_rays
                vec_to_obs = np.array([math.cos(angle), math.sin(angle)])
                
                # Repulsive force: away from obstacle, stronger when closer
                mag = self.repulsive_gain_obs * (1.0 / (dist + 1e-2))**2
                f_obs += -vec_to_obs * mag

        # 3. Total Force
        f_total = f_tracker + f_obs

        if np.linalg.norm(f_total) < 1e-3:
            desired_deg = parsed['self_heading_deg']
        else:
            desired_deg = math.degrees(math.atan2(f_total[1], f_total[0]))

        # 4. Convert to action
        self_head_deg = parsed['self_heading_deg']
        angle_delta = _normalize_angle(desired_deg - self_head_deg)

        max_turn = float(getattr(map_config, "target_max_turn_deg", 10.0))
        angle_out = float(np.clip(angle_delta, -max_turn, max_turn))
        speed_out = 1.0  # Always run at max speed

        raw_action = Model.to_normalized_action((angle_out, speed_out))
        return apply_hard_mask_adapter(raw_action, parsed, role='target')


class DWATarget:
    """
    Dynamic Window Approach (DWA) Target Policy.
    Re-implemented based on dynamic_window_approach.py reference.
    """
    def __init__(self):
        self.max_speed = map_config.target_speed
        self.min_speed = 0.0
        # Max turn per step (degrees)
        self.max_turn_deg = float(getattr(map_config, "target_max_turn_deg", 10.0))
        self.max_yaw_rate = math.radians(self.max_turn_deg) # rad/step

        self.predict_steps = 40  # Look ahead steps
        
        # Cost weights
        self.to_goal_cost_gain = 3
        self.speed_cost_gain = 1
        self.obstacle_cost_gain = 1.0
        
        self.robot_radius = map_config.agent_radius
        
        # Anti-stuck mechanism
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stuck
        self.current_vel = 0.0  # Track current velocity

    def reset(self):
        self.current_vel = 0.0

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        if parsed.get('obs_type') != 'target':
            raise ValueError("DWA target policy requires target observation")

        # 1. Current State [x, y, yaw]
        x = parsed['self_x']
        y = parsed['self_y']
        yaw = math.radians(parsed['self_heading_deg'])
        state = np.array([x, y, yaw])

        # 2. Obstacles (Radar -> Points)
        radar = parsed['radar']
        max_range = EnvParameters.FOV_RANGE
        ob = self._calc_obstacles(x, y, radar, max_range)

        # 3. Goal (Fleeing from Tracker)
        tx, ty = parsed['tracker_x'], parsed['tracker_y']
        goal = self._calc_flee_goal(x, y, tx, ty)

        # 4. Dynamic Window (Admissible velocities)
        # [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]

        # 5. Calculate Control
        u, _ = self._calc_control_and_trajectory(state, dw, goal, ob)

        # Update current velocity for next step
        self.current_vel = u[0]

        # 6. Convert to Action
        # u = [v, yaw_rate_rad]
        # Action: (angle_norm, speed_norm)
        angle_out_deg = math.degrees(u[1])
        speed_out_factor = u[0] / self.max_speed if self.max_speed > 0 else 0.0
        
        # Normalize angle to [-1, 1] relative to max turn
        angle_norm = np.clip(angle_out_deg / self.max_turn_deg, -1.0, 1.0)
        
        # Return raw action derived from DWA
        raw_action = (angle_norm, speed_out_factor)
        return apply_hard_mask_adapter(raw_action, parsed, role='target')

    def _calc_obstacles(self, x, y, radar, max_range):
        ob = []
        num_rays = len(radar)
        angle_step = 2 * math.pi / num_rays
        for i, r in enumerate(radar):
            # r is [-1, 1] -> [0, max_range]
            dist = (r + 1.0) * 0.5 * max_range
            if dist < max_range * 0.95:
                # Global angle
                a = i * angle_step
                ox = x + dist * math.cos(a)
                oy = y + dist * math.sin(a)
                ob.append([ox, oy])
        return np.array(ob) if len(ob) > 0 else np.empty((0, 2))

    def _calc_flee_goal(self, x, y, tx, ty):
        # Vector away from tracker
        dx = x - tx
        dy = y - ty
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return np.array([x + 100, y])
        # Project far away
        scale = 1000.0
        return np.array([x + dx/dist * scale, y + dy/dist * scale])

    def _motion(self, x, u):
        # x: [x, y, yaw]
        # u: [v, yaw_rate]
        yaw = x[2] + u[1]
        px = x[0] + u[0] * math.cos(yaw)
        py = x[1] + u[0] * math.sin(yaw)
        return np.array([px, py, yaw])

    def _predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        traj = [x]
        for _ in range(self.predict_steps):
            x = self._motion(x, [v, y])
            traj.append(x)
        return np.array(traj)

    def _calc_control_and_trajectory(self, x, dw, goal, ob):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_traj = np.array([x])

        # Sample v
        v_samples = np.linspace(0, self.max_speed, 10)
        # Sample yaw_rate
        y_samples = np.linspace(dw[2], dw[3], 11)

        for v in v_samples:
            for y in y_samples:
                trajectory = self._predict_trajectory(x_init, v, y)
                
                # Costs
                to_goal_cost = self.to_goal_cost_gain * self._calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - v)
                ob_cost = self.obstacle_cost_gain * self._calc_obstacle_cost(trajectory, ob)
                
                final_cost = to_goal_cost + speed_cost + ob_cost
                
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_traj = trajectory
        
        # Anti-stuck mechanism: if robot is stuck (best_u[0] ≈ 0 and current_vel ≈ 0)
        # apply a negative maximum yaw rate to escape
        if abs(best_u[0]) < self.robot_stuck_flag_cons and abs(self.current_vel) < self.robot_stuck_flag_cons:
            # Robot is stuck, force a turn to escape
            best_u[1] = -self.max_yaw_rate
        
        return best_u, best_traj

    def _calc_to_goal_cost(self, trajectory, goal):
        # Calculate heading error to goal at the end of trajectory
        last = trajectory[-1]
        dx = goal[0] - last[0]
        dy = goal[1] - last[1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - last[2]
        # Normalize angle diff to [0, pi]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost

    def _calc_obstacle_cost(self, trajectory, ob):
        if len(ob) == 0:
            return 0.0
        
        # Vectorized distance calculation
        # trajectory: (N, 3) -> (N, 1, 2) pos only
        traj_pos = trajectory[:, 0:2][:, np.newaxis, :]
        # ob: (M, 2) -> (1, M, 2)
        ob_pos = ob[np.newaxis, :, :]
        
        # Distances: (N, M)
        dists = np.linalg.norm(traj_pos - ob_pos, axis=2)
        min_dist = np.min(dists)
        
        if min_dist <= self.robot_radius:
            return float("inf")
        
        return 1.0 / min_dist


class RandomTarget:
    """
    Random target policy: Moves with random steering (Brownian-like motion).
    - Direction: Randomly changes heading within max turn limits.
    - Speed: Constant max speed.
    - Obstacle Avoidance: Relies on hard mask.
    """
    def __init__(self):
        pass

    def reset(self):
        pass

    def get_action(self, observation):
        parsed = _parse_observation(observation)
        if parsed.get('obs_type') != 'target':
            raise ValueError("Random target policy requires target observation")
        
        # Random steering: [-1.0, 1.0] corresponds to [-max_turn, max_turn]
        angle_norm = float(np.random.uniform(-1.0, 1.0))
        speed_norm = 1.0 # Full speed
        
        raw_action = (angle_norm, speed_norm)
        return apply_hard_mask_adapter(raw_action, parsed, role='target')


# ============================================================================
# Register Policies
# ============================================================================

TRACKER_POLICY_REGISTRY = {
    "CBF": CBFTracker
}

TARGET_POLICY_REGISTRY = {
    "Greedy": GreedyTarget,
    "APF": APFTarget,
    "DWA": DWATarget,
    "Hiding": HidingAStarTarget,
    "Random": RandomTarget
}