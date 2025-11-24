"""
Control Barrier Function (CBF) based controller for tracker agent.
This module implements CBF-QP for safe obstacle avoidance with soft constraints.
"""
import numpy as np
import cvxpy as cp
import math
import map_config
from lstm.model_lstm import Model
from map_config import EnvParameters


# ============================================================================
# Tunable Parameters (可调参数)
# ============================================================================

# Hard Mask Safety Parameters (硬掩码安全参数)
HARD_MASK_SAFETY_MULTIPLIER = 1.6  # 硬掩码安全距离倍数（相对于agent_radius）
HARD_MASK_EMERGENCY_BRAKE = True   # 当所有方向都被堵死时是否紧急刹车

# CBF Controller Parameters (CBF控制器参数)
CBF_SAFE_DISTANCE = 20           # CBF安全距离（像素），障碍物距此距离内触发约束
CBF_SLACK_PENALTY = 5e4            # 松弛变量惩罚系数，值越大越倾向于满足硬约束

# CBF Gamma Parameters (CBF Gamma参数 - 控制保守程度)
CBF_GAMMA_FAR = 5                # 远离障碍时的gamma（h >= 0），值越大越激进
CBF_GAMMA_NEAR = 1.2              # 接近障碍时的gamma（-5 < h < 0），值越小越保守
CBF_GAMMA_CRITICAL = 0.05          # 极近距离的gamma（h <= -5），用于脱困

# CBF Angular Weight Parameters (CBF角速度权重)
CBF_ANGULAR_WEIGHT_NORMAL = 3      # 正常情况下的角速度影响权重
CBF_ANGULAR_WEIGHT_CRITICAL = 6    # 极近距离时的角速度权重（鼓励转向脱困）

# Tracker Control Gains (追踪控制增益)
TRACKER_VELOCITY_GAIN = 2.0        # 速度控制增益（相对于目标距离）
TRACKER_ANGULAR_GAIN = 4.0         # 角速度控制增益（相对于目标方位角）

# Memory and Search Parameters (记忆与搜索参数)
MEMORY_ARRIVAL_THRESHOLD = 5.0     # 到达记忆点的判定距离（像素）
SEARCH_SPIN_SPEED_FACTOR = 1.0     # 搜索模式旋转速度倍数（相对于最大角速度）


# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_angle(angle_deg: float):
    """Normalize angle to [-180, 180] range."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return float(angle_deg)

def apply_hard_mask(action, radar, current_heading_deg, role='tracker', safety_dist=None):
    """
    硬掩码函数：基于雷达读数强制避障
    如果动作指向的方向有近距离障碍物，则强制偏转到最近的安全方向。
    
    Args:
        action: 归一化动作 (angle_norm, speed_norm)
        radar: 雷达读数数组 [-1, 1]
        current_heading_deg: 当前朝向角度（度）
        role: 'tracker' 或 'target'
        safety_dist: 安全距离（像素），None时使用默认值
    
    Returns:
        修正后的动作 (angle_norm, speed_norm)
    """
    if safety_dist is None:
        safety_dist = float(getattr(map_config, 'agent_radius', 8.0)) * HARD_MASK_SAFETY_MULTIPLIER

    # 1. 解析动作 (angle_norm, speed_norm)
    if isinstance(action, (tuple, list)):
        angle_norm, speed_norm = float(action[0]), float(action[1])
    else:
        angle_norm, speed_norm = float(action[0]), float(action[1])

    # 获取最大转向能力
    if role == 'tracker':
        max_turn = float(getattr(map_config, 'tracker_max_turn_deg', 10.0))
    else:
        max_turn = float(getattr(map_config, 'target_max_turn_deg', 10.0))
        
    angle_delta = angle_norm * max_turn
    
    # 2. 计算目标全局角度
    target_heading = _normalize_angle(current_heading_deg + angle_delta)
    
    if len(radar) == 0:
        return action

    # 3. 映射到雷达射线
    num_rays = len(radar)
    angle_step = 360.0 / num_rays
    
    # 将角度映射到 [0, 360) 并找到最近索引
    th_360 = target_heading % 360.0
    closest_idx = int(round(th_360 / angle_step)) % num_rays
    
    # 计算实际距离
    max_range = float(EnvParameters.FOV_RANGE)
    dists = (radar + 1.0) * 0.5 * max_range
    
    # 检查当前方向是否安全
    if dists[closest_idx] > safety_dist:
        return (angle_norm, speed_norm)
        
    # 4. 搜索最近的安全方向
    safe_indices = [i for i, d in enumerate(dists) if d > safety_dist]
    
    if not safe_indices:
        # 如果所有方向都被堵死
        if HARD_MASK_EMERGENCY_BRAKE:
            return (angle_norm, -1.0)  # 紧急刹车
        else:
            return action  # 保持原动作（可能导致碰撞）

    # 寻找角度差最小的安全索引
    best_idx = -1
    min_diff = 1000.0
    
    for idx in safe_indices:
        ray_angle = idx * angle_step
        diff = abs(_normalize_angle(ray_angle - target_heading))
        if diff < min_diff:
            min_diff = diff
            best_idx = idx
            
    # 5. 生成修正后的动作
    best_ray_angle = best_idx * angle_step
    needed_turn = _normalize_angle(best_ray_angle - current_heading_deg)
    
    # 限制转向幅度
    clamped_turn = np.clip(needed_turn, -max_turn, max_turn)
    
    # 归一化回动作空间
    new_angle_norm = clamped_turn / (max_turn + 1e-6)
    
    return (float(new_angle_norm), float(speed_norm))


# ============================================================================
# CBF Controller
# ============================================================================

class CBFController:
    """
    Control Barrier Function (CBF) 控制器
    使用软约束CBF-QP求解器实现安全避障
    """
    def __init__(self, safe_dist=None, max_speed=None, max_turn_rate=None):
        """
        Args:
            safe_dist: 安全距离（像素），None时使用全局参数
            max_speed: 最大速度，None时使用map_config中的值
            max_turn_rate: 最大角速度（rad/s），None时根据转向限制计算
        """
        self.safe_dist = safe_dist if safe_dist is not None else CBF_SAFE_DISTANCE
        self.max_speed = max_speed if max_speed is not None else float(getattr(map_config, 'tracker_speed', 2.4))
        
        if max_turn_rate is not None:
            self.max_turn_rate = max_turn_rate
        else:
            max_turn_deg = float(getattr(map_config, "tracker_max_turn_deg", 10.0))
            dt = 1.0 / 40.0
            self.max_turn_rate = math.radians(max_turn_deg) / dt

    def solve_cbf_qp(self, u_ref, closest_obs):
        """
        求解CBF-QP优化问题
        
        Args:
            u_ref: 参考控制输入 [v_ref, w_ref]
            closest_obs: 最近障碍物在局部坐标系中的位置 np.array([x, y]) 或 None
        
        Returns:
            最优控制输入 [v, w]
        """
        u = cp.Variable(2)  # [v, w]
        v, w = u[0], u[1]
        
        # 松弛变量（软约束）
        delta = cp.Variable(1, nonneg=True)

        # 目标函数：跟踪参考输入 + 惩罚松弛变量
        objective = cp.Minimize(
            cp.sum_squares(u - u_ref) + CBF_SLACK_PENALTY * cp.sum_squares(delta)
        )

        constraints = []

        # 运动学约束
        constraints += [
            v >= 0,
            v <= self.max_speed,
            w >= -self.max_turn_rate,
            w <= self.max_turn_rate
        ]

        # 障碍物避障约束（CBF）
        if closest_obs is not None:
            obs_vec = closest_obs
            obs_dist = np.linalg.norm(obs_vec)

            if obs_dist > 1e-3:
                obs_dir = obs_vec / obs_dist
                
                # CBF参数
                r = self.safe_dist
                h = obs_dist - r
                
                # 自适应gamma和角速度权重
                if h >= 0:
                    # 远离障碍：激进策略
                    gamma = CBF_GAMMA_FAR
                    l = CBF_ANGULAR_WEIGHT_NORMAL
                elif h > -5.0:
                    # 接近障碍：保守策略
                    gamma = CBF_GAMMA_NEAR
                    l = CBF_ANGULAR_WEIGHT_NORMAL
                else:
                    # 极近距离：极保守+鼓励转向
                    gamma = CBF_GAMMA_CRITICAL
                    l = CBF_ANGULAR_WEIGHT_CRITICAL

                # 软CBF约束: LgLfh + gamma * h + delta >= 0
                constraints.append(
                    -(obs_dir[0] * v + obs_dir[1] * l * w) + gamma * h + delta >= 0
                )

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            try:
                prob.solve(verbose=False)
            except Exception:
                return np.array([0.0, 0.0])

        if u.value is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return np.array([0.0, 0.0])

        return u.value


# ============================================================================
# Helper Functions
# ============================================================================

def _get_closest_obstacle_point(radar_norm, radar_angles, max_range):
    """
    从雷达读数中找到最近的障碍物点
    
    Args:
        radar_norm: 归一化雷达距离 [-1, 1]
        radar_angles: 雷达射线角度（局部坐标系）
        max_range: 最大探测距离
    
    Returns:
        最近障碍物的局部坐标 np.array([x, y]) 或 None
    """
    distances = (radar_norm + 1.0) * 0.5 * max_range
    valid_mask = distances < (max_range * 0.99)
    
    if not np.any(valid_mask):
        return None
        
    valid_dists = distances[valid_mask]
    valid_angles = radar_angles[valid_mask]
    
    min_idx = np.argmin(valid_dists)
    d = valid_dists[min_idx]
    a = valid_angles[min_idx]
    
    return np.array([d * math.cos(a), d * math.sin(a)])


# ============================================================================
# CBF Tracker Policy
# ============================================================================

class CBFTracker:
    """
    基于CBF的Tracker策略
    结合CBF-QP避障和目标追踪逻辑
    支持目标位置记忆和搜索模式
    """
    def __init__(self):
        self.last_target_pos = None

    def reset(self):
        """重置策略状态（清除记忆）"""
        self.last_target_pos = None

    def get_action(self, observation, privileged_state=None):
        """
        根据观测生成动作
        
        Args:
            observation: 27维Tracker观测向量
            privileged_state: 特权状态信息（用于更新记忆）
        
        Returns:
            归一化动作 (angle_norm, speed_norm)
        """
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape[0] != 27:
             return np.zeros(2)

        # 1. 恢复全局朝向（用于雷达对齐）
        heading_deg = (obs[2] + 1.0) * 180.0
        heading_rad = math.radians(heading_deg)

        # 2. 处理雷达（局部坐标系）
        radar_norm = obs[11:27]
        max_range = float(EnvParameters.FOV_RANGE)
        n_rays = len(radar_norm)
        
        angle_step = 2.0 * math.pi / n_rays
        global_angles = np.array([i * angle_step for i in range(n_rays)], dtype=np.float64)
        local_angles = global_angles - heading_rad
        
        closest_obs = _get_closest_obstacle_point(radar_norm, local_angles, max_range)
        
        # 3. 目标状态与记忆逻辑
        in_fov = float(obs[8])
        occluded = float(obs[9])
        is_visible = (in_fov > 0.5) and (occluded < 0.5)

        target_dist = 0.0
        target_bearing_rad = 0.0
        has_target = False

        if is_visible:
            # 可见：直接使用观测
            dist_norm = float(obs[3])
            bearing_norm = float(obs[4])
            target_dist = (dist_norm + 1.0) / 2.0 * max_range
            target_bearing_deg = bearing_norm * 180.0
            target_bearing_rad = math.radians(target_bearing_deg)
            has_target = True
            
            # 更新记忆
            if privileged_state is not None:
                t_pos = privileged_state.get('target')
                if t_pos:
                    self.last_target_pos = np.array([t_pos['x'], t_pos['y']])
        
        elif self.last_target_pos is not None and privileged_state is not None:
            # 不可见但有记忆：导航到最后已知位置
            my_pos = privileged_state.get('tracker')
            if my_pos:
                curr_pos = np.array([my_pos['x'], my_pos['y']])
                curr_theta = math.radians(my_pos['theta'])
                
                diff = self.last_target_pos - curr_pos
                dist = np.linalg.norm(diff)
                
                # 检查是否到达记忆点
                if dist > MEMORY_ARRIVAL_THRESHOLD: 
                    target_dist = dist
                    global_angle = math.atan2(diff[1], diff[0])
                    bearing = global_angle - curr_theta
                    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
                    target_bearing_rad = bearing
                    has_target = True
                else:
                    # 到达记忆点，清除记忆并进入搜索模式
                    self.last_target_pos = None
        
        # 4. 计算参考控制
        max_speed = float(getattr(map_config, 'tracker_speed', 2.4))
        max_turn_deg = float(getattr(map_config, "tracker_max_turn_deg", 10.0))
        dt = 1.0 / 40.0
        phys_max_w = math.radians(max_turn_deg) / dt
        
        if has_target:
            # 追踪模式：朝向目标前进
            v_ref = np.clip(TRACKER_VELOCITY_GAIN * target_dist, 0, max_speed)
            w_ref = np.clip(TRACKER_ANGULAR_GAIN * target_bearing_rad, -phys_max_w, phys_max_w)
        else:
            # 搜索模式：原地旋转
            v_ref = 0.0
            w_ref = phys_max_w * SEARCH_SPIN_SPEED_FACTOR
        
        u_ref = np.array([v_ref, w_ref])
        
        # 5. 初始化CBF控制器
        controller = CBFController(
            safe_dist=CBF_SAFE_DISTANCE,
            max_speed=max_speed,
            max_turn_rate=phys_max_w
        )
        
        # 6. 求解QP
        u_star = controller.solve_cbf_qp(u_ref, closest_obs)
        v_cmd, w_cmd = u_star[0], u_star[1]
        
        # 7. 转换为动作
        angle_delta_deg = math.degrees(w_cmd * dt)
        angle_out = float(np.clip(angle_delta_deg, -max_turn_deg, max_turn_deg))
        speed_factor = min(v_cmd / max_speed, 1.0) if max_speed > 0 else 0.0
        
        raw_action = Model.to_normalized_action((angle_out, speed_factor))
        
        # 8. 应用硬掩码（安全层）
        return apply_hard_mask(raw_action, radar_norm, heading_deg, role='tracker')
