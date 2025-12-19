import os
import sys
import math
import numpy as np
from typing import Optional, Union, Tuple
import gymnasium as gym
from gymnasium import spaces

import env_lib, map_config
from map_config import EnvParameters
import pygame


class TrackingEnv(gym.Env):
    Metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 40
    }

    def __init__(self, spawn_outside_fov=False, enable_safety_layer=True, bounce_on_collision=False):
        super().__init__()
        self.spawn_outside_fov = bool(spawn_outside_fov)
        self.enable_safety_layer = bool(enable_safety_layer)
        self.bounce_on_collision = bool(bounce_on_collision)
        self.mask_flag = getattr(map_config, 'mask_flag', False)
        self.width = map_config.width
        self.height = map_config.height
        self.pixel_size = map_config.pixel_size
        self.target_speed = map_config.target_speed
        self.tracker_speed = map_config.tracker_speed
        self.tracker = None
        self.target = None
        self._render_surface = None
        self.tracker_trajectory = []
        self.target_trajectory = []
        self.step_count = 0
        self.target_frame_count = 0
        self.prev_tracker_pos = None
        self.last_tracker_pos = None
        self.prev_target_pos = None
        self.last_target_pos = None

        # 视场配置：直接使用 EnvParameters
        self.fov_angle = EnvParameters.FOV_ANGLE
        self.fov_range = EnvParameters.FOV_RANGE
        self.radar_rays = EnvParameters.RADAR_RAYS

        # 捕获配置
        self.capture_radius = float(getattr(map_config, 'capture_radius', 10.0))
        self.capture_sector_angle_deg = float(getattr(map_config, 'capture_sector_angle_deg', 60.0))
        self.capture_required_steps = int(getattr(map_config, 'capture_required_steps', 8))
        self._capture_counter = 0

        self.last_observed_target_pos = None
        self.steps_since_observed = 0
        self._best_distance = None

        # 观测空间保持不变：tracker(27), target(24) 打包成二元组
        # 观测空间：tracker scalar (11) + radar (64) = 75
        obs_dim = 11 + 64
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.current_obs = None

        # FOV 多边形缓存
        self._fov_cache = None
        self._fov_cache_valid = False

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
        """Tracker 相对观测（75维），FOV / 雷达基于统一射线逻辑。"""
        obs = np.zeros(75, dtype=np.float32)

        # 1) 自身状态
        current_vel = self._get_velocity(self.tracker, self.prev_tracker_pos)
        vel_magnitude = float(np.linalg.norm(current_vel))
        max_speed = float(max(self.tracker_speed, self.target_speed))
        normalized_vel = np.clip(vel_magnitude / (max_speed + 1e-6), 0, 1) * 2.0 - 1.0

        normalized_angular_vel = self._get_angular_velocity(
            self.tracker, self.prev_tracker_pos, 
            getattr(map_config, 'tracker_max_angular_speed', 10.0)
        )
        normalized_heading = (self.tracker['theta'] / 180.0) - 1.0

        obs[0] = normalized_vel
        obs[1] = normalized_angular_vel
        obs[2] = normalized_heading

        # 2) 计算可见性 & 确定目标参考状态
        # 先计算真实相对信息用于判断可见性
        true_rel_vec, true_dist = self._get_relative_position(self.tracker, self.target)
        absolute_angle = math.atan2(true_rel_vec[1], true_rel_vec[0])
        true_rel_angle_deg = self._normalize_angle(
            math.degrees(absolute_angle) - self.tracker['theta']
        )
        fov_half = self.fov_angle * 0.5
        
        # 调用可见性更新 (复用现有逻辑)
        in_fov, occluded = self._update_visibility(true_rel_angle_deg, true_dist, fov_half)
        
        # 3) 构造用于观测的目标状态 (Last Known Position Logic)
        obs_target_state = None
        obs_target_vel = np.zeros(2, dtype=np.float32)
        
        is_visible = (in_fov > 0.5 and occluded < 0.5)
        
        if is_visible:
            # 真实目标可见：使用真实状态
            obs_target_state = self.target
            obs_target_vel = self._get_velocity(self.target, self.prev_target_pos)
        elif self.last_observed_target_pos is not None:
            # 目标不可见但有记忆：使用最后一次观测到的位置 (Ghost)
            # Ghost 视为静止，且没有角度信息(theta=0)
            obs_target_state = {
                'x': self.last_observed_target_pos[0] - self.pixel_size * 0.5,
                'y': self.last_observed_target_pos[1] - self.pixel_size * 0.5,
                'theta': 0.0 # 无法得知朝向
            }
            obs_target_vel = np.zeros(2, dtype=np.float32) # 假设Ghost静止
        else:
            # 目标不可见且无记忆：完全丢失
            obs_target_state = None

        # 4) 基于观测状态计算相对特征
        if obs_target_state is not None:
            # 有参考目标 (真实或Ghost)
            rel_vec, distance = self._get_relative_position(self.tracker, obs_target_state)
            normalized_distance = np.clip((distance / self.fov_range) * 2.0 - 1.0, -1.0, 1.0)
            
            abs_ang = math.atan2(rel_vec[1], rel_vec[0])
            rel_ang = self._normalize_angle(math.degrees(abs_ang) - self.tracker['theta'])
            normalized_bearing = np.clip(rel_ang / 180.0, -1.0, 1.0)
            
            # 相对速度
            relative_vel = obs_target_vel - current_vel
            relative_speed = float(np.linalg.norm(relative_vel))
            max_relative_speed = max_speed * 2.0
            normalized_relative_speed = np.clip(
                (relative_speed / (max_relative_speed + 1e-6)) * 2.0 - 1.0, -1.0, 1.0
            ) 
            
            # FOV Edge (基于观测到的角度)
            fov_edge_angle = min(abs(rel_ang + fov_half), abs(rel_ang - fov_half))
            normalized_fov_edge = np.clip((fov_edge_angle / fov_half) * 2.0 - 1.0, -1.0, 1.0)
            
            # 相对角速度 (Ghost视为0角速度)
            if is_visible:
                target_ang_vel = self._get_angular_velocity(
                    self.target, self.prev_target_pos,
                    getattr(map_config, 'target_max_angular_speed', 12.0)
                )
            else:
                target_ang_vel = 0.0
            
            normalized_relative_angular_vel = np.clip(target_ang_vel - normalized_angular_vel, -1.0, 1.0)
            
        else:
            # 没有任何目标信息
            normalized_distance = -1.0 # 认为非常远? 或者 1.0? 通常 -1.0 代表 0 距离, 1.0 代表 max range. 
            # 这里原逻辑: distance=0 -> -1.0; distance=max -> 1.0. 
            # 如果没看到，设为 1.0 (最远) 或 0.0 (中间) 比较合理，或者保持原状。
            # 为了让网络知道没东西，我们设为边界值 1.0 (超出视野)
            normalized_distance = 1.0 
            normalized_bearing = 0.0
            normalized_relative_speed = 0.0
            normalized_relative_angular_vel = 0.0
            normalized_fov_edge = 1.0

        obs[3] = normalized_distance
        obs[4] = normalized_bearing
        obs[5] = normalized_relative_speed
        obs[6] = normalized_relative_angular_vel
        obs[7] = normalized_fov_edge

        # 5) 状态特征
        max_unobserved = float(EnvParameters.MAX_UNOBSERVED_STEPS)
        normalized_unobserved = np.clip(
            (self.steps_since_observed / max_unobserved) * 2.0 - 1.0,
            -1.0, 1.0
        )
        obs[8] = in_fov
        obs[9] = occluded
        obs[10] = normalized_unobserved

        # 4) 雷达 64 维，360°
        obs[11:11+64] = self._sense_agent_radar(
            self.tracker, num_rays=self.radar_rays, full_circle=True
        )
        return obs

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
        """
        center = np.array([
            agent['x'] + self.pixel_size * 0.5,
            agent['y'] + self.pixel_size * 0.5
        ], dtype=float)
        heading = math.radians(agent.get('theta', 0.0))
        if full_circle:
            angles = [2 * math.pi * i / num_rays for i in range(num_rays)]
        else:
            angle_range = math.pi
            angles = [
                heading + (i / (num_rays - 1) - 0.5) * angle_range
                for i in range(num_rays)
            ]
        max_radar_range = float(EnvParameters.FOV_RANGE)
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        
        # Use Numba accelerated batch ray casting (handled inside env_lib)
        dists = env_lib.ray_distances_multi(center, angles, max_radar_range, padding=pad)
            
        readings = (np.asarray(dists, dtype=np.float32) / max_radar_range) * 2.0 - 1.0
        return readings

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

    def _control_to_physical(self, action, role):
        if action is None:
            return None
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError("action must contain exactly two elements")
        if np.all(np.abs(arr) <= 1.0 + 1e-6):
            if role == 'tracker':
                max_acc = float(getattr(map_config, 'tracker_max_acc', 0.1))
                max_ang_acc = float(getattr(map_config, 'tracker_max_ang_acc', 2.0))
            elif role == 'target':
                max_acc = float(getattr(map_config, 'target_max_acc', 0.1))
                max_ang_acc = float(getattr(map_config, 'target_max_ang_acc', 2.0))
            else:
                max_acc = 0.1
                max_ang_acc = 2.0
            
            # Action[0] = Angular Acc, Action[1] = Linear Acc
            ang_acc = float(np.clip(arr[0], -1.0, 1.0) * max_ang_acc)
            lin_acc = float(np.clip(arr[1], -1.0, 1.0) * max_acc)
            return ang_acc, lin_acc
        return float(arr[0]), float(arr[1])

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
        """检查动作是否会导致碰撞"""
        action_physical = self._control_to_physical(action, role)
        if action_physical is None:
            return True

        ang_acc, lin_acc = action_physical
        
        # Simulate physics
        sim_agent = agent.copy()
        max_speed = self.tracker_speed if role == 'tracker' else self.target_speed
        max_ang_speed = float(getattr(map_config, f'{role}_max_angular_speed', 10.0))
        
        # Update velocities
        sim_agent['v'] = float(sim_agent.get('v', 0.0) + lin_acc)
        sim_agent['w'] = float(sim_agent.get('w', 0.0) + ang_acc)
        
        # Clip
        sim_agent['v'] = float(np.clip(sim_agent['v'], 0.0, max_speed))
        sim_agent['w'] = float(np.clip(sim_agent['w'], -max_ang_speed, max_ang_speed))
        
        # Update pose
        new_theta = (sim_agent['theta'] + sim_agent['w']) % 360.0
        rad_theta = math.radians(new_theta)
        new_x = np.clip(sim_agent['x'] + sim_agent['v'] * math.cos(rad_theta), 
                        0, self.width - self.pixel_size)
        new_y = np.clip(sim_agent['y'] + sim_agent['v'] * math.sin(rad_theta), 
                        0, self.height - self.pixel_size)

        center_x = new_x + self.pixel_size * 0.5
        center_y = new_y + self.pixel_size * 0.5

        safety_margin = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        return not env_lib.is_point_blocked(center_x, center_y, padding=safety_margin)
    
    def _find_valid_action(self, agent, original_action, role, max_attempts=8):
        """找到有效的替代动作（仅调整角度，不降低速度）"""
        physical_action = self._control_to_physical(original_action, role)
        if physical_action is None:
            return original_action, False

        # 检查输入格式
        is_normalized_input = False
        if isinstance(original_action, (tuple, list, np.ndarray)):
            arr = np.asarray(original_action, dtype=np.float32).reshape(-1)
            if arr.shape[0] == 2 and np.all(np.abs(arr) <= 1.0 + 1e-6):
                is_normalized_input = True

        # 原动作有效则直接返回
        if self._is_action_valid(agent, original_action, role):
            return original_action, False

        base_angle, base_speed = physical_action
        max_turn = (float(getattr(map_config, 'tracker_max_angular_speed', 10.0)) if role == 'tracker' 
                    else float(getattr(map_config, 'target_max_angular_speed', 12.0)))

        def make_candidate(angle_deg, speed_factor):
            clipped_angle = float(np.clip(angle_deg, -max_turn, max_turn))
            clipped_speed = float(np.clip(speed_factor, 0.0, 1.0))
            candidate_physical = (clipped_angle, clipped_speed)
            return (self._physical_to_control(candidate_physical, role) 
                    if is_normalized_input else candidate_physical)

        attempts = 0
        def try_candidate(angle_deg, speed_factor):
            nonlocal attempts
            if attempts >= max_attempts:
                return None
            attempts += 1
            candidate = make_candidate(angle_deg, speed_factor)
            return candidate if self._is_action_valid(agent, candidate, role) else None

        # 仅通过角度偏移寻找可行解（速度保持不变）
        for delta in (10, -10, 20, -20, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 120, -120, 135, -135, 180, -180):
            candidate = try_candidate(base_angle + delta, base_speed)
            if candidate is not None:
                return candidate, True

        # 兜底：返回原始动作（未修正）
        return original_action, False

    def step(self, action: Union[Tuple, list, np.ndarray] = None,
             target_action: Optional[Tuple] = None):
        self.step_count += 1
        tracker_action, target_action = self._parse_actions(action, target_action)
        tracker_corrected = False
        target_corrected = False
        if tracker_action is not None:
            # Apply safety layer only if enabled
            if self.enable_safety_layer:
                tracker_action, tracker_corrected = self._find_valid_action(
                    self.tracker, tracker_action, 'tracker'
                )
            tracker_phys = self._control_to_physical(tracker_action, 'tracker')
            if tracker_phys is not None:
                ang_acc, lin_acc = tracker_phys
                max_ang_speed = float(getattr(map_config, 'tracker_max_angular_speed', 10.0))
                self.tracker = env_lib.agent_move_accel(
                    self.tracker, lin_acc, ang_acc, self.tracker_speed, max_ang_speed, 
                    role='tracker', enable_safety_layer=self.enable_safety_layer
                )
        if target_action is not None:
            # Target ALWAYS uses safety layer (to prevent getting stuck)
            target_action, target_corrected = self._find_valid_action(
                self.target, target_action, 'target'
            )
            target_phys = self._control_to_physical(target_action, 'target')
            if target_phys is not None:
                ang_acc, lin_acc = target_phys
                max_ang_speed = float(getattr(map_config, 'target_max_angular_speed', 12.0))
                self.target = env_lib.agent_move_accel(
                    self.target, lin_acc, ang_acc, self.target_speed, max_ang_speed, 
                    role='target', enable_safety_layer=True  # Always enabled for target
                )

        self._fov_cache_valid = False
        self.tracker_trajectory.append((
            self.tracker['x'] + self.pixel_size / 2.0,
            self.tracker['y'] + self.pixel_size / 2.0
        ))
        self.target_trajectory.append((
            self.target['x'] + self.pixel_size / 2.0,
            self.target['y'] + self.pixel_size / 2.0
        ))
        max_len = getattr(map_config, 'trail_max_len', 600)
        if len(self.tracker_trajectory) > max_len:
            self.tracker_trajectory = self.tracker_trajectory[-max_len:]
        if len(self.target_trajectory) > max_len:
            self.target_trajectory = self.target_trajectory[-max_len:]

        if self.last_tracker_pos is not None:
            self.prev_tracker_pos = self.last_tracker_pos.copy()
        self.last_tracker_pos = self.tracker.copy()
        if self.last_target_pos is not None:
            self.prev_target_pos = self.last_target_pos.copy()
        self.last_target_pos = self.target.copy()

        # When safety layer is disabled, check for actual collisions
        if not self.enable_safety_layer:
            agent_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
            tracker_center_x = self.tracker['x'] + self.pixel_size * 0.5
            tracker_center_y = self.tracker['y'] + self.pixel_size * 0.5
            if env_lib.is_point_blocked(tracker_center_x, tracker_center_y, padding=agent_radius):
                tracker_corrected = True
                if self.bounce_on_collision:
                    bounce_dist = 25.0
                    heading_rad = math.radians(self.tracker['theta'])
                    new_x = self.tracker['x'] - bounce_dist * math.cos(heading_rad)
                    new_y = self.tracker['y'] - bounce_dist * math.sin(heading_rad)
                    self.tracker['x'] = float(np.clip(new_x, 0, self.width - self.pixel_size))
                    self.tracker['y'] = float(np.clip(new_y, 0, self.height - self.pixel_size))
                    self.tracker['v'] = 0.0
            target_center_x = self.target['x'] + self.pixel_size * 0.5
            target_center_y = self.target['y'] + self.pixel_size * 0.5
            if env_lib.is_point_blocked(target_center_x, target_center_y, padding=agent_radius):
                target_corrected = True

        in_sector = self._is_target_in_capture_sector()
        if in_sector:
            self._capture_counter = min(
                self._capture_counter + 1, self.capture_required_steps
            )
        else:
            self._capture_counter = 0
        sector_captured = (self._capture_counter >= self.capture_required_steps)

        reward, terminated, truncated, info = env_lib.reward_calculate(
            self.tracker, self.target,
            prev_tracker=self.prev_tracker_pos,
            prev_target=self.prev_target_pos,
            tracker_collision=bool(tracker_corrected),
            target_collision=bool(target_corrected),
            sector_captured=bool(sector_captured),
            capture_progress=int(self._capture_counter),
            capture_required_steps=int(self.capture_required_steps),
            bounce_on_collision=self.bounce_on_collision
        )

        try:
            cur_dist = float(math.hypot(
                self.tracker['x'] - self.target['x'],
                self.tracker['y'] - self.target['y']
            ))
            if self._best_distance is None or cur_dist < (self._best_distance - 1e-6):
                self._best_distance = cur_dist
                info['closest_record_improved'] = True
            else:
                info['closest_record_improved'] = False
            info['closest_record_value'] = float(
                self._best_distance if self._best_distance is not None else cur_dist
            )
        except Exception:
            pass

        self.current_obs = self._get_obs_features()
        if self.step_count >= EnvParameters.EPISODE_LEN and not terminated:
            truncated = True

        self.target_frame_count += 1
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
        self.target_frame_count = 0
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
        dists = env_lib.ray_distances_multi((cx_world, cy_world), angles, max_range, padding=0.0)

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