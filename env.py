import math
import numpy as np
from typing import Optional, Union, Tuple
import gymnasium as gym
from gymnasium import spaces
from shapely.geometry import Point, Polygon as ShapelyPolygon

import env_lib, map_config
from map_config import EnvParameters
from mlp.alg_parameters_mlp import TrainingParameters
import pygame


class TrackingEnv(gym.Env):
    Metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 40
    }

    def __init__(self, spawn_outside_fov=False, safety_layer_enabled=None):
        super().__init__()
        self.spawn_outside_fov = bool(spawn_outside_fov)
        self.width = map_config.width
        self.height = map_config.height
        self.pixel_size = map_config.pixel_size
        self.target_speed = map_config.target_speed
        self.tracker_speed = map_config.tracker_speed
        
        # Safety Layer: environment-assisted obstacle avoidance
        # When enabled, actions are modified before execution to prevent collisions
        self.safety_layer_enabled = safety_layer_enabled if safety_layer_enabled is not None else TrainingParameters.SAFETY_LAYER_ENABLED
        
        # State
        self.tracker = None
        self.target = None
        self.step_count = 0
        self.prev_tracker_pos = None
        self.prev_target_pos = None
        
        # FOV config
        self.fov_angle = EnvParameters.FOV_ANGLE
        self.fov_range = EnvParameters.FOV_RANGE
        self.radar_rays = EnvParameters.RADAR_RAYS
        
        # Capture config
        self.capture_radius = map_config.capture_radius
        self.capture_sector_angle_deg = map_config.capture_sector_angle_deg
        self.capture_required_steps = int(getattr(map_config, 'capture_required_steps', 1))
        self._capture_counter = 0
        
        # Visibility tracking
        self.last_observed_target_pos = None
        self.steps_since_observed = 0
        
        # Observation space: tracker(75), target(72)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(75,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Rendering
        self._render_surface = None
        self._fov_cache = None
        self._fov_cache_valid = False
        
        # Trajectory (limit length for memory)
        self.tracker_trajectory = []
        self.target_trajectory = []
        self._max_trajectory_len = 500
        
        # Radar angle cache for performance
        self._radar_angle_cache = {}

    def _get_obs_features(self):
        """
        生成观测向量：
        - Tracker: 相对观测（27维，局部感知）
        - Target: 全局观测（24维，绝对坐标 + 360°雷达）
        """
        tracker_obs = self._get_tracker_observation()
        target_obs = self._get_target_observation()
        return tracker_obs, target_obs

    def _get_tracker_observation(self):
        """优化的Tracker观测生成"""
        obs = np.zeros(75, dtype=np.float32)

        # 1. Self state (3 dims)
        current_vel = self._get_velocity(self.tracker, self.prev_tracker_pos)
        vel_mag = float(np.linalg.norm(current_vel))
        max_speed = max(self.tracker_speed, self.target_speed)
        
        obs[0] = np.clip(vel_mag / (max_speed + 1e-6) * 2.0 - 1.0, -1.0, 1.0)
        obs[1] = self._get_angular_velocity(
            self.tracker, self.prev_tracker_pos,
            getattr(map_config, 'tracker_max_angular_speed', 10.0)
        )
        obs[2] = (self.tracker['theta'] / 180.0) - 1.0

        # 2. Target relative info (optimized)
        true_rel_vec, true_dist = self._get_relative_position(self.tracker, self.target)
        abs_angle = math.atan2(true_rel_vec[1], true_rel_vec[0])
        rel_angle_deg = self._normalize_angle(math.degrees(abs_angle) - self.tracker['theta'])
        fov_half = self.fov_angle * 0.5
        
        # Visibility check
        in_fov = abs(rel_angle_deg) <= fov_half and true_dist <= self.fov_range
        occluded = False
        
        if in_fov:
            occluded = self._is_line_blocked(self.tracker, self.target, padding=2.0)
            if not occluded:
                self.last_observed_target_pos = np.array([
                    self.target['x'] + self.pixel_size * 0.5,
                    self.target['y'] + self.pixel_size * 0.5
                ])
                self.steps_since_observed = 0
            else:
                self.steps_since_observed += 1
        else:
            self.steps_since_observed += 1

        # Determine observation state
        is_visible = in_fov and not occluded
        
        if is_visible:
            # Use real target state
            obs_dist = true_dist
            obs_bearing = rel_angle_deg
        elif self.last_observed_target_pos is not None:
            # Use last known position
            ghost_rel = self.last_observed_target_pos - np.array([
                self.tracker['x'] + self.pixel_size * 0.5,
                self.tracker['y'] + self.pixel_size * 0.5
            ])
            obs_dist = float(np.linalg.norm(ghost_rel))
            obs_bearing = self._normalize_angle(
                math.degrees(math.atan2(ghost_rel[1], ghost_rel[0])) - self.tracker['theta']
            )
        else:
            # No information
            obs_dist = self.fov_range
            obs_bearing = 0.0

        # Normalize observations (5 dims: distance, bearing, rel_speed, rel_ang_vel, fov_edge)
        obs[3] = np.clip((obs_dist / self.fov_range) * 2.0 - 1.0, -1.0, 1.0)
        obs[4] = np.clip(obs_bearing / 180.0, -1.0, 1.0)
        
        # Relative velocities (simplified when not visible)
        if is_visible:
            target_vel = self._get_velocity(self.target, self.prev_target_pos)
            rel_vel = target_vel - current_vel
            obs[5] = np.clip(np.linalg.norm(rel_vel) / (max_speed * 2.0) * 2.0 - 1.0, -1.0, 1.0)
            
            target_ang_vel = self._get_angular_velocity(
                self.target, self.prev_target_pos,
                getattr(map_config, 'target_max_angular_speed', 12.0)
            )
            obs[6] = np.clip(target_ang_vel - obs[1], -1.0, 1.0)
        else:
            obs[5] = 0.0
            obs[6] = 0.0
        
        # FOV edge distance
        fov_edge_dist = min(abs(rel_angle_deg + fov_half), abs(rel_angle_deg - fov_half))
        obs[7] = np.clip((fov_edge_dist / fov_half) * 2.0 - 1.0, -1.0, 1.0)

        # 3. State flags (3 dims)
        obs[8] = 1.0 if in_fov else -1.0
        obs[9] = 1.0 if occluded else -1.0
        obs[10] = np.clip(
            (self.steps_since_observed / EnvParameters.MAX_UNOBSERVED_STEPS) * 2.0 - 1.0,
            -1.0, 1.0
        )

        # 4. Radar (64 dims) - batch computation
        obs[11:75] = self._sense_agent_radar_optimized(self.tracker)
        
        return obs

    def _sense_agent_radar_optimized(self, agent):
        """优化的雷达感知（批量计算 + 缓存）"""
        center = np.array([
            agent['x'] + self.pixel_size * 0.5,
            agent['y'] + self.pixel_size * 0.5
        ], dtype=float)
        
        # Generate angles (360° scan)
        angles = np.linspace(0, 2 * np.pi, self.radar_rays, endpoint=False)
        
        # Batch ray casting (Numba optimized)
        dists = env_lib.ray_distances_multi(
            center, angles, self.fov_range,
            padding=getattr(map_config, 'agent_radius', self.pixel_size * 0.5)
        )
        
        # Normalize
        return (dists / self.fov_range) * 2.0 - 1.0

    def _get_velocity(self, agent, prev_pos):
        """计算智能体速度"""
        if prev_pos is not None:
            dx = agent['x'] - prev_pos['x']
            dy = agent['y'] - prev_pos['y']
            return np.array([dx, dy], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_angular_velocity(self, agent, prev_pos, max_ang_speed):
        """计算归一化角速度"""
        if prev_pos is not None:
            prev_heading = prev_pos.get('theta', 0.0)
            angle_change = self._normalize_angle(agent['theta'] - prev_heading)
            return np.clip(angle_change / (max_ang_speed + 1e-6), -1.0, 1.0)
        return 0.0

    def _get_relative_position(self, from_agent, to_agent):
        """计算相对位置向量和距离"""
        from_center = np.array([
            from_agent['x'] + self.pixel_size * 0.5,
            from_agent['y'] + self.pixel_size * 0.5
        ], dtype=np.float32)
        to_center = np.array([
            to_agent['x'] + self.pixel_size * 0.5,
            to_agent['y'] + self.pixel_size * 0.5
        ], dtype=np.float32)
        relative_vec = to_center - from_center
        distance = float(np.linalg.norm(relative_vec))
        return relative_vec, distance

    def _update_visibility(self, relative_angle_deg, distance, fov_half_deg):
        """基于 FOV 楔形 + 占据栅格线段遮挡更新可见性。"""
        in_angle = abs(relative_angle_deg) <= fov_half_deg
        in_range = distance <= self.fov_range
        in_fov = 1.0 if (in_angle and in_range) else 0.0
        occluded = 0.0

        if in_fov > 0.5:
            if self._is_line_blocked(self.tracker, self.target):
                occluded = 1.0
                self.steps_since_observed += 1
            else:
                target_center = np.array([
                    self.target['x'] + self.pixel_size * 0.5,
                    self.target['y'] + self.pixel_size * 0.5
                ], dtype=np.float32)
                self.last_observed_target_pos = target_center.copy()
                self.steps_since_observed = 0
        else:
            self.steps_since_observed += 1

        return in_fov, occluded

    def _get_target_observation(self):
        """Target 全局观测（72维），接口不变。"""
        obs = np.zeros(72, dtype=np.float32)

        obs[0] = (self.target['x'] / self.width) * 2.0 - 1.0
        obs[1] = (self.target['y'] / self.height) * 2.0 - 1.0
        obs[2] = (self.target['theta'] / 180.0) - 1.0
        obs[3] = (self.tracker['x'] / self.width) * 2.0 - 1.0
        obs[4] = (self.tracker['y'] / self.height) * 2.0 - 1.0
        obs[5] = (self.tracker['theta'] / 180.0) - 1.0

        max_speed = float(max(self.tracker_speed, self.target_speed))
        target_vel = self._get_velocity(self.target, self.prev_target_pos)
        tracker_vel = self._get_velocity(self.tracker, self.prev_tracker_pos)
        obs[6] = np.clip((np.linalg.norm(target_vel) / (max_speed + 1e-6)) * 2.0 - 1.0,
                         -1.0, 1.0)
        obs[7] = np.clip((np.linalg.norm(tracker_vel) / (max_speed + 1e-6)) * 2.0 - 1.0,
                         -1.0, 1.0)

        obs[8:8+64] = self._sense_agent_radar(
            self.target, num_rays=self.radar_rays, full_circle=True
        )
        return obs

    def _normalize_angle(self, angle_deg):
        """将角度归一化到[-180, 180]"""
        angle_deg = angle_deg % 360.0
        if angle_deg > 180.0:
            angle_deg -= 360.0
        return float(angle_deg)

    def _sense_agent_radar(self, agent, num_rays=10, full_circle=False):
        """
        生成多方向雷达读数：统一使用 env_lib.ray_distances_multi。
        优化：使用缓存的角度数组
        """
        center = np.array([
            agent['x'] + self.pixel_size * 0.5,
            agent['y'] + self.pixel_size * 0.5
        ], dtype=np.float64)
        
        # Use cached angles for full_circle (most common case for 64, 16 rays)
        if full_circle:
            cache_key = f'fc_{num_rays}'
            if cache_key not in self._radar_angle_cache:
                self._radar_angle_cache[cache_key] = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
            angles = self._radar_angle_cache[cache_key]
        else:
            # Heading-relative angles (less common, compute inline)
            heading = math.radians(agent.get('theta', 0.0))
            cache_key = f'hr_{num_rays}'
            if cache_key not in self._radar_angle_cache:
                base = np.linspace(-0.5, 0.5, num_rays) * np.pi
                self._radar_angle_cache[cache_key] = base
            angles = heading + self._radar_angle_cache[cache_key]
        
        max_radar_range = float(EnvParameters.FOV_RANGE)
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        
        # Use Numba accelerated batch ray casting
        dists = env_lib.ray_distances_multi(center, angles, max_radar_range, padding=pad)
        
        # Normalize to [-1, 1]
        return (dists / max_radar_range) * 2.0 - 1.0

    def _is_line_blocked(self, agent1, agent2, padding=0.0):
        """
        判断 agent1 和 agent2 中心连线是否被障碍阻断：
        - 检查全长，不再缩短 10%
        - 支持 padding 参数以进行更严格的检测
        """
        x1 = agent1['x'] + self.pixel_size * 0.5
        y1 = agent1['y'] + self.pixel_size * 0.5
        x2 = agent2['x'] + self.pixel_size * 0.5
        y2 = agent2['y'] + self.pixel_size * 0.5
        dx, dy = (x2 - x1), (y2 - y1)
        line_len = math.hypot(dx, dy)
        if line_len <= 1e-6:
            return False

        angle = math.atan2(dy, dx)
        # 检查全长
        check_len = line_len
        dist = env_lib.ray_distance_grid((x1, y1), angle, check_len, padding=padding)
        
        # 若在到达目标前撞到障碍，则认为被遮挡
        # 留一点微小余量防止浮点误差
        return bool(dist < check_len - 1e-3)

    def _ray_distance(self, origin, angle_rad, max_range, padding=None):
        """
        单射线查询接口：
        - 默认 padding=0.0，视觉相关的 FOV/遮挡全部走这里；
        - 碰撞类调用可显式传入 padding。
        """
        pad = 0.0 if padding is None else float(padding)
        return float(env_lib.ray_distance_grid(origin, angle_rad, max_range, padding=pad))

    def _parse_actions(self, action: Union[Tuple, list, np.ndarray, None], target_action: Optional[Tuple] = None):
        if target_action is not None:
            return action, target_action
        if isinstance(action, (tuple, list)) and len(action) == 2:
            first = np.asarray(action[0])
            second = np.asarray(action[1])
            if first.size == 2 and second.size == 2:
                return action[0], action[1]
        return action, target_action

    # _control_to_physical removed (deprecated acceleration logic)

    def _physical_to_control(self, physical_action, role):
        if physical_action is None:
            return None
        angle_delta, speed_factor = physical_action
        if role == 'tracker':
            max_turn = float(getattr(map_config, 'tracker_max_angular_speed', 10.0))
        elif role == 'target':
            max_turn = float(getattr(map_config, 'target_max_angular_speed', 12.0))
        else:
            max_turn = 10.0
        angle_norm = 0.0 if max_turn <= 1e-6 else np.clip(angle_delta / max_turn, -1.0, 1.0)
        speed_norm = np.clip(speed_factor * 2.0 - 1.0, -1.0, 1.0)
        return (float(angle_norm), float(speed_norm))

    def _is_action_valid(self, agent, action, role):
        """简化的动作有效性检查"""
        # Simulate next position
        max_turn = getattr(map_config, f'{role}_max_turn_deg', 45.0)
        angle_delta = np.clip(action[0], -max_turn, max_turn)
        speed = np.clip(action[1], 0.0, 1.0) * (
            self.tracker_speed if role == 'tracker' else self.target_speed
        )
        
        new_theta = (agent['theta'] + angle_delta) % 360.0
        rad = math.radians(new_theta)
        
        new_x = agent['x'] + speed * math.cos(rad)
        new_y = agent['y'] + speed * math.sin(rad)
        
        # Check collision at center
        cx = new_x + self.pixel_size * 0.5
        cy = new_y + self.pixel_size * 0.5
        
        return not env_lib.is_point_blocked(
            cx, cy, padding=getattr(map_config, 'agent_radius', self.pixel_size * 0.5)
        )

    def _apply_safety_layer(self, action, agent_state, role='tracker'):
        """
        Safety Layer: 修正动作以避免碰撞
        
        当 safety_layer_enabled=True 时，使用雷达感知和 hard_mask 
        来修正可能导致碰撞的动作。这让 RL 可以专注于追踪目标，
        而避障由环境自动处理。
        
        多层保护机制:
        1. Hard Mask: 基于雷达检测修正转向角度
        2. 动作有效性检查: 模拟执行后验证是否碰撞
        3. 速度调整: 如果仍有碰撞风险则降低速度
        
        Args:
            action: 原始动作 [angle_norm, speed_norm]
            agent_state: 智能体状态字典
            role: 'tracker' 或 'target'
        
        Returns:
            修正后的安全动作
        """
        if not self.safety_layer_enabled:
            return action
        
        from rule_policies import apply_hard_mask
        
        # 获取 360° 雷达读数
        radar = self._sense_agent_radar_optimized(agent_state)
        current_heading = agent_state.get('theta', 0.0)
        
        # 第1层: Hard Mask 基础修正
        safe_action = apply_hard_mask(action, radar, current_heading, role)
        
        # 第2层: 动作有效性检查（模拟执行）
        if not self._is_action_valid(agent_state, safe_action, role):
            # 尝试降低速度
            for speed_factor in [0.5, 0.25, 0.0]:
                reduced_action = np.array([safe_action[0], safe_action[1] * speed_factor], dtype=np.float32)
                if self._is_action_valid(agent_state, reduced_action, role):
                    safe_action = reduced_action
                    break
            else:
                # 如果前进不行，尝试反方向
                reverse_action = np.array([-safe_action[0], 0.0], dtype=np.float32)
                if self._is_action_valid(agent_state, reverse_action, role):
                    safe_action = reverse_action
                else:
                    # 完全停止
                    safe_action = np.array([0.0, 0.0], dtype=np.float32)
        
        return safe_action

    def step(self, action=None, target_action=None, residual_action=None, action_penalty_coef=0.0):
        """优化的step函数"""
        # Update obstacles if dynamic
        if map_config.dynamic_obstacles:
            map_config.update_dynamic_obstacles()
            env_lib.build_occupancy()

        # Parse actions with better error handling
        if action is None:
            tracker_action = np.zeros(2, dtype=np.float32)
            target_action_parsed = np.zeros(2, dtype=np.float32)
        elif target_action is not None:
            # Both provided explicitly
            tracker_action = np.asarray(action, dtype=np.float32).flatten()
            target_action_parsed = np.asarray(target_action, dtype=np.float32).flatten()
        elif isinstance(action, (tuple, list)) and len(action) == 2:
            # Check if it's (tracker_action, target_action) tuple
            first = np.asarray(action[0], dtype=np.float32).flatten()
            second = np.asarray(action[1], dtype=np.float32).flatten()
            if first.size == 2 and second.size == 2:
                tracker_action = first
                target_action_parsed = second
            else:
                # It's a single action
                tracker_action = np.asarray(action, dtype=np.float32).flatten()
                target_action_parsed = np.zeros(2, dtype=np.float32)
        else:
            # Single action provided
            tracker_action = np.asarray(action, dtype=np.float32).flatten()
            target_action_parsed = np.zeros(2, dtype=np.float32)

        # Validate shapes
        if tracker_action.size != 2:
            raise ValueError(f"tracker_action must have size 2, got {tracker_action.size}")
        if target_action_parsed.size != 2:
            raise ValueError(f"target_action must have size 2, got {target_action_parsed.size}")

        # Store previous positions
        self.prev_tracker_pos = self.tracker.copy()
        self.prev_target_pos = self.target.copy()

        # Apply Safety Layer to tracker action (if enabled)
        # This modifies the action to avoid obstacles, letting RL focus on pursuit
        if self.safety_layer_enabled:
            tracker_action = self._apply_safety_layer(tracker_action, self.tracker, 'tracker')

        # Apply actions
        self.tracker = env_lib.agent_move(
            self.tracker, tracker_action, self.tracker_speed, role='tracker'
        )
        self.target = env_lib.agent_move(
            self.target, target_action_parsed, self.target_speed, role='target'
        )

        # Invalidate FOV cache
        self._fov_cache_valid = False

        # Update trajectories (with length limit)
        self.tracker_trajectory.append((
            self.tracker['x'] + self.pixel_size / 2,
            self.tracker['y'] + self.pixel_size / 2
        ))
        self.target_trajectory.append((
            self.target['x'] + self.pixel_size / 2,
            self.target['y'] + self.pixel_size / 2
        ))
        
        # Limit trajectory length
        if len(self.tracker_trajectory) > self._max_trajectory_len:
            self.tracker_trajectory.pop(0)
        if len(self.target_trajectory) > self._max_trajectory_len:
            self.target_trajectory.pop(0)

        # Check capture
        sector_captured = self._is_target_in_capture_sector()
        if sector_captured:
            self._capture_counter += 1
        else:
            self._capture_counter = max(0, self._capture_counter - 1)

        is_captured = self._capture_counter >= self.capture_required_steps

        # Calculate reward
        tracker_collision = not self._is_action_valid(
            self.prev_tracker_pos, tracker_action, 'tracker'
        )
        target_collision = not self._is_action_valid(
            self.prev_target_pos, target_action_parsed, 'target'
        )

        reward, terminated, truncated, info = env_lib.reward_calculate(
            self.tracker, self.target,
            prev_tracker=self.prev_tracker_pos,
            prev_target=self.prev_target_pos,
            tracker_collision=tracker_collision,
            target_collision=target_collision,
            sector_captured=is_captured,
            capture_progress=self._capture_counter,
            capture_required_steps=self.capture_required_steps,
            residual_action=residual_action,
            action_penalty_coef=action_penalty_coef
        )

        # Get new observation
        self.current_obs = self._get_obs_features()
        
        # Check episode length
        self.step_count += 1
        if self.step_count >= EnvParameters.EPISODE_LEN and not terminated:
            truncated = True

        return self.current_obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # ...existing reset skeleton,但在开头重建障碍和占据栅格...
        super().reset(seed=seed)
        if hasattr(map_config, "regenerate_obstacles"):
            density_level = getattr(map_config, 'current_obstacle_density', None)
            map_config.regenerate_obstacles(density_level=density_level)
        try:
            env_lib.build_occupancy(
                width=self.width,
                height=self.height,
                cell=getattr(map_config, 'occ_cell',
                             getattr(map_config, 'pixel_size', self.pixel_size)),
                obstacles=getattr(map_config, 'obstacles', [])
            )
        except Exception:
            pass

        self.step_count = 0
        self.tracker_trajectory = []
        self.target_trajectory = []

        # 循环尝试生成合法的初始状态，确保 target 在视野内（如果需要）
        for _ in range(100):
            self.tracker = self._sample_tracker_state()
            if self.spawn_outside_fov:
                self.target = self._sample_target_outside_fov(self.tracker)
                break
            else:
                self.target = self._sample_target_in_fov_far_end(self.tracker)
                # 强制检查可见性
                if self._check_initial_visibility():
                    break
        else:
            print("Warning: Failed to spawn visible target after retries.")

        self.prev_tracker_pos = self.tracker.copy()
        self.last_tracker_pos = self.tracker.copy()
        self.prev_target_pos = self.target.copy()
        self.last_target_pos = self.target.copy()

        self.last_observed_target_pos = None
        self.steps_since_observed = 0
        self._capture_counter = 0

        try:
            self._best_distance = float(math.hypot(
                self.tracker['x'] - self.target['x'],
                self.tracker['y'] - self.target['y']
            ))
        except Exception:
            self._best_distance = None

        self._fov_cache = None
        self._fov_cache_valid = False
        self.current_obs = self._get_obs_features()
        return self.current_obs, {}

    def _check_initial_visibility(self):
        """检查当前 target 是否对 tracker 可见"""
        # 1. 检查遮挡 (使用 padding=2.0 确保视线不贴墙)
        if self._is_line_blocked(self.tracker, self.target, padding=2.0):
            return False
        
        # 2. 检查是否在 FOV 角度和距离内
        tx = self.tracker['x'] + self.pixel_size * 0.5
        ty = self.tracker['y'] + self.pixel_size * 0.5
        gx = self.target['x'] + self.pixel_size * 0.5
        gy = self.target['y'] + self.pixel_size * 0.5
        dx, dy = (gx - tx), (gy - ty)
        dist = math.hypot(dx, dy)
        
        if dist > self.fov_range:
            return False
            
        tracker_heading = float(self.tracker.get('theta', 0.0))
        angle_to_target = math.degrees(math.atan2(dy, dx))
        rel = self._normalize_angle(angle_to_target - tracker_heading)
        
        if abs(rel) > self.fov_angle * 0.5:
            return False
            
        return True

    def _sample_tracker_state(self):
        """
        采样tracker初始状态
        特点：
        1. 在地图边缘区域出生（避免中心障碍物）
        2. 朝向地图中心
        3. 不在障碍物内
        """
        margin = 30
        edge_zone = 80  # 边缘区域宽度
        
        map_center_x = self.width / 2
        map_center_y = self.height / 2
        
        for attempt in range(512):
            # 在地图边缘区域采样
            zone = self.np_random.integers(0, 4)  # 0:上 1:右 2:下 3:左
            
            if zone == 0:  # 上边缘
                x_min = margin
                x_max = self.width - margin
                y_min = margin
                y_max = margin + edge_zone
            elif zone == 1:  # 右边缘
                x_min = self.width - margin - edge_zone
                x_max = self.width - margin
                y_min = margin
                y_max = self.height - margin
            elif zone == 2:  # 下边缘
                x_min = margin
                x_max = self.width - margin
                y_min = self.height - margin - edge_zone
                y_max = self.height - margin
            else:  # 左边缘
                x_min = margin
                x_max = margin + edge_zone
                y_min = margin
                y_max = self.height - margin
            
            # 确保范围有效
            if x_max <= x_min or y_max <= y_min:
                continue
            
            x = float(self.np_random.uniform(x_min, x_max))
            y = float(self.np_random.uniform(y_min, y_max))
            
            # 检查是否在障碍物内
            center_x = x + self.pixel_size * 0.5
            center_y = y + self.pixel_size * 0.5
            
            if env_lib.is_point_blocked(center_x, center_y, padding=map_config.agent_radius):
                continue
            
            # 计算朝向地图中心的角度
            dx = map_center_x - center_x
            dy = map_center_y - center_y
            theta = float(math.degrees(math.atan2(dy, dx)) % 360.0)
            
            return {'x': x, 'y': y, 'theta': theta}
        
        # 兜底方案：左上角朝向中心
        print("[WARNING] Failed to spawn tracker in edge zone, using fallback position")
        x = float(margin)
        y = float(margin)
        dx = map_center_x - (x + self.pixel_size * 0.5)
        dy = map_center_y - (y + self.pixel_size * 0.5)
        theta = float(math.degrees(math.atan2(dy, dx)) % 360.0)
        return {'x': x, 'y': y, 'theta': theta}

    def _sample_target_in_fov_far_end(self, tracker):
        """
        在tracker的FOV远端采样target位置（更鲁棒）：
        1) 在[-FOV/2, +FOV/2]内均匀扫描多个角度
        2) 对每个角度从远到近扫描多个距离，选取第一个满足：
           - 在地图有效范围（带margin）
           - 不在障碍物内
           - tracker到该点视线无遮挡
        3) 若仍无解，执行高密度角度扫描；最终兜底到正前方的可见点
        """
        # 基本量
        tracker_cx = tracker['x'] + self.pixel_size * 0.5
        tracker_cy = tracker['y'] + self.pixel_size * 0.5
        head_rad = math.radians(tracker['theta'])
        fov_half = math.radians(EnvParameters.FOV_ANGLE / 2.0)
        # 扫描范围
        min_dist = EnvParameters.FOV_RANGE * 0.65
        max_dist = EnvParameters.FOV_RANGE * 0.92
        margin = 20.0
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))

        def in_bounds(x, y):
            return (x >= margin and x <= self.width - self.pixel_size - margin and
                    y >= margin and y <= self.height - self.pixel_size - margin)

        def visible_from_tracker(px, py):
            # 用env的遮挡检查
            return not self._is_line_blocked({'x': tracker['x'], 'y': tracker['y']},
                                             {'x': px, 'y': py})

        # 角度、距离扫描
        angles = np.linspace(-0.95 * fov_half, 0.95 * fov_half, 49)
        np.random.shuffle(angles)
        dists = np.linspace(max_dist, min_dist, 12)

        for ang_off in angles:
            ang = head_rad + float(ang_off)
            ca, sa = math.cos(ang), math.sin(ang)
            for dist in dists:
                cx = tracker_cx + dist * ca
                cy = tracker_cy + dist * sa
                # 转回 agent 左上角
                x = cx - self.pixel_size * 0.5
                y = cy - self.pixel_size * 0.5
                if not in_bounds(x, y):
                    continue
                if env_lib.is_point_blocked(cx, cy, padding=pad):
                    continue
                if not visible_from_tracker(x, y):
                    continue
                # 朝地图外缘的方向作为默认朝向（偏离中心）
                map_cx, map_cy = self.width / 2.0, self.height / 2.0
                dx, dy = (cx - map_cx), (cy - map_cy)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    theta = float(math.degrees(math.atan2(dy, dx)) % 360.0)
                else:
                    theta = float(self.np_random.uniform(0.0, 360.0))
                return {'x': float(x), 'y': float(y), 'theta': theta}

        # 二次密集角度扫描（更密）
        angles2 = np.linspace(-fov_half, fov_half, 97)
        for ang_off in angles2:
            ang = head_rad + float(ang_off)
            ca, sa = math.cos(ang), math.sin(ang)
            for dist in dists:
                cx = tracker_cx + dist * ca
                cy = tracker_cy + dist * sa
                x = cx - self.pixel_size * 0.5
                y = cy - self.pixel_size * 0.5
                if not in_bounds(x, y):
                    continue
                if env_lib.is_point_blocked(cx, cy, padding=pad):
                    continue
                if not visible_from_tracker(x, y):
                    continue
                map_cx, map_cy = self.width / 2.0, self.height / 2.0
                dx, dy = (cx - map_cx), (cy - map_cy)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    theta = float(math.degrees(math.atan2(dy, dx)) % 360.0)
                else:
                    theta = float(self.np_random.uniform(0.0, 360.0))
                return {'x': float(x), 'y': float(y), 'theta': theta}

        # 兜底：沿正前方由远到近找可见点（不再打印warning）
        for dist in np.linspace(max_dist, min_dist, 20):
            cx = tracker_cx + dist * math.cos(head_rad)
            cy = tracker_cy + dist * math.sin(head_rad)
            x = cx - self.pixel_size * 0.5
            y = cy - self.pixel_size * 0.5
            if not in_bounds(x, y):
                continue
            if env_lib.is_point_blocked(cx, cy, padding=pad):
                continue
            if not visible_from_tracker(x, y):
                continue
            map_cx, map_cy = self.width / 2.0, self.height / 2.0
            dx, dy = (cx - map_cx), (cy - map_cy)
            theta = float(math.degrees(math.atan2(dy, dx)) % 360.0)
            return {'x': float(x), 'y': float(y), 'theta': theta}

        # 最终兜底：放在边界安全点（紧贴FOV中线的近端）
        dist = (min_dist + max_dist) * 0.5
        cx = np.clip(tracker_cx + dist * math.cos(head_rad), margin, self.width - margin)
        cy = np.clip(tracker_cy + dist * math.sin(head_rad), margin, self.height - margin)
        x = float(cx - self.pixel_size * 0.5)
        y = float(cy - self.pixel_size * 0.5)
        theta = float(self.np_random.uniform(0.0, 360.0))
        return {'x': x, 'y': y, 'theta': theta}

    def _sample_target_outside_fov(self, tracker):
        """
        在tracker视野之外采样target：
        条件：角度不在[-FOV/2, +FOV/2]内 或 距离>FOV_RANGE 或 被遮挡。
        优先保证无遮挡可视角外，其次允许遮挡内（仍视为不可见）。
        """
        fov_half_deg = EnvParameters.FOV_ANGLE * 0.5
        max_range = EnvParameters.FOV_RANGE
        margin = 20.0
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        tx = tracker['x'] + self.pixel_size * 0.5
        ty = tracker['y'] + self.pixel_size * 0.5
        heading = tracker['theta']

        def valid(pt):
            x, y = pt
            if x < margin or x > self.width - self.pixel_size - margin:
                return False
            if y < margin or y > self.height - self.pixel_size - margin:
                return False
            cx = x + self.pixel_size * 0.5
            cy = y + self.pixel_size * 0.5
            if env_lib.is_point_blocked(cx, cy, padding=pad):
                return False
            dx, dy = (cx - tx), (cy - ty)
            dist = math.hypot(dx, dy)
            angle_to = math.degrees(math.atan2(dy, dx))
            rel = self._normalize_angle(angle_to - heading)
            in_wedge = (abs(rel) <= fov_half_deg) and (dist <= max_range)
            # 若在视场楔形内但被遮挡也可接受
            if in_wedge and not self._is_line_blocked(tracker, {'x': x, 'y': y}):
                return False
            return True

        # 主采样：随机均匀分布
        for _ in range(600):
            x = float(self.np_random.uniform(margin, self.width - margin - self.pixel_size))
            y = float(self.np_random.uniform(margin, self.height - margin - self.pixel_size))
            if valid((x, y)):
                cx = x + self.pixel_size * 0.5
                cy = y + self.pixel_size * 0.5
                # 朝地图外缘或随机
                map_cx, map_cy = self.width / 2.0, self.height / 2.0
                dx, dy = (cx - map_cx), (cy - map_cy)
                theta = float(math.degrees(math.atan2(dy, dx)) % 360.0)
                return {'x': x, 'y': y, 'theta': theta}

        # 次级：在 tracker 身后半平面扫描
        behind_angles = np.linspace(110, 250, 60)  # 相对朝向的后方区
        for ang_off in behind_angles:
            ang = math.radians((heading + ang_off) % 360.0)
            dist = self.np_random.uniform(max_range * 0.4, max_range * 1.2)
            cx = tx + dist * math.cos(ang)
            cy = ty + dist * math.sin(ang)
            x = cx - self.pixel_size * 0.5
            y = cy - self.pixel_size * 0.5
            if valid((x, y)):
                map_cx, map_cy = self.width / 2.0, self.height / 2.0
                dx2, dy2 = (cx - map_cx), (cy - map_cy)
                theta = float(math.degrees(math.atan2(dy2, dx2)) % 360.0)
                return {'x': x, 'y': y, 'theta': theta}

        # 兜底：放在最大视距后方固定点
        ang = math.radians((heading + 180.0) % 360.0)
        dist = max_range * 1.05
        cx = np.clip(tx + dist * math.cos(ang), margin, self.width - margin)
        cy = np.clip(ty + dist * math.sin(ang), margin, self.height - margin)
        x = cx - self.pixel_size * 0.5
        y = cy - self.pixel_size * 0.5
        theta = float(self.np_random.uniform(0.0, 360.0))
        return {'x': x, 'y': y, 'theta': theta}

    def _sample_agent_state(self, center_bias=False, exclude=None, avoid=None, min_gap=None):
        """废弃方法（保留以兼容旧代码）"""
        return self._sample_tracker_state()

    def _get_fov_points(self, force_recompute=False):
        """
        获取 FOV 扇形点集，用于渲染。
        使用 Warp 批量射线投射来优化计算。
        """
        if self._fov_cache_valid and self._fov_cache is not None and not force_recompute:
            return self._fov_cache

        ss = getattr(map_config, 'ssaa', 1)
        cx_world = self.tracker['x'] + map_config.pixel_size * 0.5
        cy_world = self.tracker['y'] + map_config.pixel_size * 0.5
        cx = cx_world * ss
        cy = cy_world * ss

        heading_deg = self.tracker.get('theta', 0.0)
        heading_rad = math.radians(heading_deg)
        fov_half = math.radians(EnvParameters.FOV_ANGLE / 2.0)
        max_range = EnvParameters.FOV_RANGE

        # Use fixed number of rays for GPU batching (faster than adaptive CPU recursion)
        num_rays = 64
        angles = np.linspace(heading_rad - fov_half, heading_rad + fov_half, num_rays)
        
        # Use Numba accelerated batch ray casting
        dists = env_lib.ray_distances_multi(
            (cx_world, cy_world), angles, max_range, padding=0.0
        )

        pts = [(cx, cy)]
        for i in range(num_rays):
            dist = dists[i]
            angle = angles[i]
            px = cx + dist * ss * math.cos(angle)
            py = cy + dist * ss * math.sin(angle)
            pts.append((px, py))

        self._fov_cache = pts
        self._fov_cache_valid = True
        return pts

    def render(self, mode='rgb_array'):
        if pygame is not None and self._render_surface is None:
            ss = getattr(map_config, 'ssaa', 1)
            self._render_surface = pygame.Surface(
                (self.width * ss, self.height * ss), flags=pygame.SRCALPHA
            )
        fov_points = self._get_fov_points()
        canvas = env_lib.get_canvas(
            self.target, self.tracker,
            self.tracker_trajectory, self.target_trajectory,
            surface=self._render_surface,
            fov_points=fov_points
        )
        return canvas

    def close(self):
        self._render_surface = None

    def _is_target_in_capture_sector(self) -> bool:
        """
        目标在捕获扇形内且无遮挡。
        几何判断保持不变，遮挡用新的 _is_line_blocked（不膨胀）。
        """
        tx = self.tracker['x'] + self.pixel_size * 0.5
        ty = self.tracker['y'] + self.pixel_size * 0.5
        gx = self.target['x'] + self.pixel_size * 0.5
        gy = self.target['y'] + self.pixel_size * 0.5
        dx, dy = (gx - tx), (gy - ty)
        dist = math.hypot(dx, dy)
        if dist > self.capture_radius:
            return False

        tracker_heading = float(self.tracker.get('theta', 0.0))
        angle_to_target = math.degrees(math.atan2(dy, dx))
        rel = self._normalize_angle(angle_to_target - tracker_heading)
        half_sector = self.capture_sector_angle_deg * 0.5
        if abs(rel) > half_sector:
            return False

        fov_half = self.fov_angle * 0.5
        in_fov = (abs(rel) <= fov_half) and (dist <= self.fov_range)
        if not in_fov:
            return False

        if self._is_line_blocked(self.tracker, self.target):
            return False
        return True

    def get_privileged_state(self):
        """
        为规则策略提供特权信息（绝对坐标）
        仅用于规则策略，不用于强化学习训练
        
        Returns:
            dict: 包含tracker和target的绝对位置、朝向等信息
        """
        return {
            'tracker': {
                'x': float(self.tracker['x']),
                'y': float(self.tracker['y']),
                'theta': float(self.tracker['theta']),
                'center_x': float(self.tracker['x'] + self.pixel_size * 0.5),
                'center_y': float(self.tracker['y'] + self.pixel_size * 0.5)
            },
            'target': {
                'x': float(self.target['x']),
                'y': float(self.target['y']),
                'theta': float(self.target['theta']),
                'center_x': float(self.target['x'] + self.pixel_size * 0.5),
                'center_y': float(self.target['y'] + self.pixel_size * 0.5)
            },
            'map': {
                'width': float(self.width),
                'height': float(self.height)
            }
        }

    def polygon_sdf_grad_lse(self, rbt_state, tgt_state, debug=False):
        """
        计算 Visibility SDF 和梯度（优化版本）
        
        使用批量射线追踪构建 FOV 多边形 + 解析梯度近似。
        相比原版减少约 90% 的射线计算。
        
        Args:
            rbt_state: np.ndarray [x, y, theta] 像素坐标（左上角）
            tgt_state: np.ndarray [x, y, theta] 像素坐标（左上角）
            debug: bool 调试标志
            
        Returns:
            grad_rbt: np.ndarray (3,) 机器人状态梯度 [dx, dy, dtheta]
            grad_tgt: np.ndarray (3,) 目标状态梯度（简化为0）
            sdf: float SDF 值（负数=目标在FOV内，正数=在外）
        """
        # 1. 计算中心坐标
        rbt_center = np.array([
            rbt_state[0] + self.pixel_size * 0.5,
            rbt_state[1] + self.pixel_size * 0.5
        ])
        tgt_center = np.array([
            tgt_state[0] + self.pixel_size * 0.5,
            tgt_state[1] + self.pixel_size * 0.5
        ])
        
        # 2. 使用批量射线追踪构建 FOV 多边形（优化）
        heading_rad = math.radians(rbt_state[2])
        fov_half = math.radians(self.fov_angle / 2.0)
        num_rays = 25  # 减少射线数量以提升性能（原50）
        angles = np.linspace(heading_rad - fov_half, heading_rad + fov_half, num_rays)
        
        # 批量射线投射 - 单次调用替代循环
        dists = env_lib.ray_distances_multi(rbt_center, angles, self.fov_range, padding=0.0)
        
        # 构建多边形点
        fov_polygon_points = [rbt_center.tolist()]
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        for i in range(num_rays):
            px = rbt_center[0] + dists[i] * cos_angles[i]
            py = rbt_center[1] + dists[i] * sin_angles[i]
            fov_polygon_points.append([px, py])
        
        # 3. 计算 SDF：目标到 FOV 多边形的距离
        try:
            poly = ShapelyPolygon(fov_polygon_points)
            point = Point(tgt_center)
            
            if poly.contains(point):
                # 目标在 FOV 内，SDF < 0
                sdf = -point.distance(poly.boundary)
            else:
                # 目标在 FOV 外，SDF > 0
                sdf = point.distance(poly.boundary)
        except Exception as e:
            if debug:
                print(f"[polygon_sdf_grad_lse] Polygon construction failed: {e}")
            sdf = self.fov_range * 2.0
        
        # 4. 解析梯度近似（替代数值微分，避免额外射线计算）
        # 使用目标相对于机器人的位置方向作为梯度方向
        rel_vec = tgt_center - rbt_center
        rel_dist = max(np.linalg.norm(rel_vec), 1e-6)
        rel_dir = rel_vec / rel_dist
        
        # 目标到机器人的角度
        angle_to_target = math.atan2(rel_vec[1], rel_vec[0])
        angle_diff = angle_to_target - heading_rad
        # 归一化到 [-pi, pi]
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
        
        # 位置梯度：指向目标方向的反方向
        # 当机器人向目标移动时，SDF减小（目标更容易在FOV内）
        grad_x = -rel_dir[0] * 0.5
        grad_y = -rel_dir[1] * 0.5
        
        # 角度梯度：转向目标会减小SDF
        # 当目标在左侧（angle_diff > 0），左转（正角度变化）减小SDF
        grad_theta = -np.sign(angle_diff) * min(abs(angle_diff), 1.0) * 5.0
        
        grad_rbt = np.array([grad_x, grad_y, grad_theta])
        grad_tgt = np.array([0.0, 0.0, 0.0])
        
        return grad_rbt, grad_tgt, float(sdf)

    def SDF_RT_circular(self, rbt_state, radius, num_rays):
        """
        圆形射线追踪（适配 cbf_qp.py 接口）
        
        在机器人周围进行 360° 圆形扫描，返回所有射线与障碍物的交点。
        
        Args:
            rbt_state: np.ndarray [x, y, theta] 像素坐标（左上角）
            radius: float 射线最大长度
            num_rays: int 射线数量
            
        Returns:
            np.ndarray: shape (N, 2) 障碍物交点坐标 [x, y]
        """
        center = np.array([
            rbt_state[0] + self.pixel_size * 0.5,
            rbt_state[1] + self.pixel_size * 0.5
        ])
        
        # 360° 均匀采样
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        
        # 批量射线追踪
        dists = env_lib.ray_distances_multi(center, angles, radius, padding=0.0)
        
        # 收集撞击点
        points = []
        for i, dist in enumerate(dists):
            if dist < radius:  # 撞到障碍物
                px = center[0] + dist * np.cos(angles[i])
                py = center[1] + dist * np.sin(angles[i])
                points.append([px, py])
        
        return np.array(points) if points else np.array([]).reshape(0, 2)