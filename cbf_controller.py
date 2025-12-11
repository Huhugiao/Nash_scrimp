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

# Hard Mask Safety Parameters (moved to env_lib.py)
from env_lib import apply_hard_mask, HARD_MASK_SAFETY_MULTIPLIER, HARD_MASK_CHECK_WINDOW, HARD_MASK_EMERGENCY_BRAKE

# CBF Controller Parameters (CBF控制器参数)
CBF_SAFE_DISTANCE = 16.0           
CBF_SLACK_PENALTY = 1e5            

# CBF Gamma Parameters (CBF Gamma参数 - 控制保守程度)
CBF_GAMMA_FAR = 5                
CBF_GAMMA_NEAR = 1.2              
CBF_GAMMA_CRITICAL = 0.05          

# CBF Angular Weight Parameters (CBF角速度权重)
CBF_ANGULAR_WEIGHT_NORMAL = 3      
CBF_ANGULAR_WEIGHT_CRITICAL = 6    

# Tracker Control Gains (追踪控制增益)
TRACKER_VELOCITY_GAIN = 2.0        
TRACKER_ANGULAR_GAIN = 4.0         

# Memory and Search Parameters (记忆与搜索参数)
MEMORY_ARRIVAL_THRESHOLD = 5.0     
SEARCH_SPIN_SPEED_FACTOR = 1.0     
NAVIGATION_GAP_THRESHOLD = 48.0    

# Recovery Parameters
STUCK_THRESHOLD = 15               # Reduced to react faster (was 30)
RECOVERY_DURATION = 40             # Duration to lock direction (was 15)


# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_angle(angle_deg: float):
    """Normalize angle to [-180, 180] range."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return float(angle_deg)



# ============================================================================
# CBF Controller
# ============================================================================

class CBFController:
    """
    Control Barrier Function (CBF) 控制器
    使用软约束CBF-QP求解器实现安全避障
    优化：使用 cp.Parameter 避免重复编译问题
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
            max_turn_deg = float(getattr(map_config, "tracker_max_angular_speed", 10.0))
            dt = 1.0 / 40.0
            self.max_turn_rate = math.radians(max_turn_deg) / dt
            
        self._init_problem()

    def _init_problem(self):
        """初始化并缓存 QP 问题结构"""
        self.u = cp.Variable(2)  # [v, w]
        self.delta = cp.Variable(1, nonneg=True)
        
        # 定义参数
        self.u_ref_param = cp.Parameter(2)
        self.cbf_A_param = cp.Parameter(2) # 对应 [obs_dir[0], obs_dir[1]*l]
        self.cbf_b_param = cp.Parameter(1) # 对应 gamma * h
        
        # 目标函数
        objective = cp.Minimize(
            cp.sum_squares(self.u - self.u_ref_param) + CBF_SLACK_PENALTY * cp.sum_squares(self.delta)
        )

        constraints = [
            self.u[0] >= 0,
            self.u[0] <= self.max_speed,
            self.u[1] >= -self.max_turn_rate,
            self.u[1] <= self.max_turn_rate,
            # CBF 约束: -(A @ u) + b + delta >= 0
            # 当无障碍时，A=0, b=0 => delta >= 0 (恒成立)
            -(self.cbf_A_param @ self.u) + self.cbf_b_param + self.delta >= 0
        ]

        self.prob = cp.Problem(objective, constraints)

    def solve_cbf_qp(self, u_ref, closest_obs):
        """
        求解CBF-QP优化问题 (使用参数更新)
        """
        # Sanitize inputs
        if u_ref is None or not np.all(np.isfinite(u_ref)):
            return np.array([0.0, 0.0])
        
        self.u_ref_param.value = np.array(u_ref, dtype=np.float64)
        
        if closest_obs is not None and np.linalg.norm(closest_obs) > 1e-3:
            obs_dist = np.linalg.norm(closest_obs)
            obs_dir = closest_obs / obs_dist
            
            # CBF参数
            r = self.safe_dist
            h = obs_dist - r
            
            # 自适应gamma和角速度权重
            if h >= 0:
                gamma = CBF_GAMMA_FAR
                l = CBF_ANGULAR_WEIGHT_NORMAL
            elif h > -5.0:
                gamma = CBF_GAMMA_NEAR
                l = CBF_ANGULAR_WEIGHT_NORMAL
            else:
                gamma = CBF_GAMMA_CRITICAL
                l = CBF_ANGULAR_WEIGHT_CRITICAL

            # 更新约束参数
            # 原始约束: -(obs_dir[0] * v + obs_dir[1] * l * w) + gamma * h + delta >= 0
            A_val = np.array([obs_dir[0], obs_dir[1] * l], dtype=np.float64)
            b_val = np.array([gamma * h], dtype=np.float64)
            
            if np.all(np.isfinite(A_val)) and np.all(np.isfinite(b_val)):
                self.cbf_A_param.value = A_val
                self.cbf_b_param.value = b_val
            else:
                self.cbf_A_param.value = np.zeros(2)
                self.cbf_b_param.value = np.zeros(1)
        else:
            # 无障碍物，使约束失效 (0 >= 0)
            self.cbf_A_param.value = np.zeros(2)
            self.cbf_b_param.value = np.zeros(1)

        try:
            # warm_start=True 加速后续求解
            # 注意：OSQP在某些版本下更新参数时若开启warm_start可能报 "Problem data validation" 错误
            # 对于小规模问题(2变量)，关闭warm_start影响微乎其微，但能显著提高稳定性
            self.prob.solve(solver=cp.OSQP, verbose=False, warm_start=False)
        except Exception:
            return np.array([0.0, 0.0])

        if self.u.value is None or self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return np.array([0.0, 0.0])

        return self.u.value


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

def _get_navigation_heading(radar_norm, current_heading_rad, target_bearing_rad, max_range):
    """
    基于雷达寻找最佳导航方向（Follow the Gap）
    如果目标方向被阻挡，寻找最近的可行方向
    """
    dists = (radar_norm + 1.0) * 0.5 * max_range
    n_rays = len(dists)
    angle_step = 2.0 * math.pi / n_rays
    
    # 目标在全局坐标系下的角度
    target_global = current_heading_rad + target_bearing_rad
    
    # 检查目标方向是否被阻挡 (检查目标方向及相邻射线)
    target_idx = int(round(target_global / angle_step)) % n_rays
    check_indices = [(target_idx - 1) % n_rays, target_idx, (target_idx + 1) % n_rays]
    
    blocked = False
    for idx in check_indices:
        if dists[idx] < NAVIGATION_GAP_THRESHOLD:
            blocked = True
            break
            
    if not blocked:
        return target_bearing_rad
        
    # 如果被阻挡，寻找所有“安全”的射线
    safe_indices = [i for i, d in enumerate(dists) if d > NAVIGATION_GAP_THRESHOLD]
    
    if not safe_indices:
        # 如果所有方向都被阻挡（陷入死胡同或极度拥挤），寻找距离最远的方向（Least Bad）
        best_idx = np.argmax(dists)
    else:
        # 在安全射线中寻找角度最接近目标方向的
        best_idx = -1
        min_diff = float('inf')
        
        for idx in safe_indices:
            ray_global = idx * angle_step
            # 计算角度差 (处理圆周周期性)
            diff = ray_global - target_global
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            
            if abs(diff) < min_diff:
                min_diff = abs(diff)
                best_idx = idx
            
    if best_idx != -1:
        best_global = best_idx * angle_step
        # 转回相对bearing
        best_rel = best_global - current_heading_rad
        best_rel = (best_rel + math.pi) % (2 * math.pi) - math.pi
        return best_rel
        
    return target_bearing_rad


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
        # 初始化一次控制器，避免重复创建开销
        self.controller = CBFController()
        # Stuck detection state
        self.stuck_count = 0
        self.recovery_mode = False
        self.recovery_timer = 0
        self.recovery_target_global_angle = 0.0
        self.last_u = np.zeros(2)

    def reset(self):
        """重置策略状态（清除记忆）"""
        self.last_target_pos = None
        self.stuck_count = 0
        self.recovery_mode = False
        self.recovery_timer = 0
        self.recovery_target_global_angle = 0.0

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
        # 允许 27 (旧) 或 75 (新) 或 动态计算的维度
        # Tracker scalar (11) + Radar (RADAR_RAYS)
        expected_dim = 11 + EnvParameters.RADAR_RAYS
        if obs.shape[0] != expected_dim and obs.shape[0] != 27:
             # Fallback or error? For now, if not matching expected, return 0
             # But let's be flexible if it matches the new config
             return np.zeros(2)

        # 1. Stuck Detection & Recovery Logic
        # obs[0] is normalized velocity [-1, 1]. -1 means 0 speed.
        current_vel_norm = float(obs[0])
        if current_vel_norm < -0.9: # Speed near 0
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)
            
        # 2. 恢复全局朝向（用于雷达对齐）
        heading_deg = (obs[2] + 1.0) * 180.0
        heading_rad = math.radians(heading_deg)

        # 3. 处理雷达（局部坐标系）
        # 自动适配雷达维度
        radar_start = 11
        radar_end = radar_start + EnvParameters.RADAR_RAYS
        if obs.shape[0] == 27: # Backward compatibility
             radar_end = 27
        
        radar_norm = obs[radar_start:radar_end]
        max_range = float(EnvParameters.FOV_RANGE)
        n_rays = len(radar_norm)
        
        angle_step = 2.0 * math.pi / n_rays
        global_angles = np.array([i * angle_step for i in range(n_rays)], dtype=np.float64)
        local_angles = global_angles - heading_rad
        
        closest_obs = _get_closest_obstacle_point(radar_norm, local_angles, max_range)
        
        # 4. Trigger Recovery
        if self.stuck_count > STUCK_THRESHOLD and not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_timer = RECOVERY_DURATION
            self.stuck_count = 0
            
            # Determine best escape direction ONCE when entering recovery
            dists = (radar_norm + 1.0) * 0.5 * max_range
            
            # Find direction with max averaged distance (smoothing)
            window = 2
            smoothed_dists = []
            for i in range(n_rays):
                val = 0.0
                for w in range(-window, window+1):
                    val += dists[(i+w)%n_rays]
                smoothed_dists.append(val)
            
            best_idx = np.argmax(smoothed_dists)
            # Convert to global angle
            self.recovery_target_global_angle = best_idx * angle_step

        # 5. Handle Recovery State
        if self.recovery_mode:
            self.recovery_timer -= 1
            if self.recovery_timer <= 0:
                self.recovery_mode = False
        
        # 6. Calculate Reference Control
        max_speed = float(getattr(map_config, 'tracker_speed', 2.4))
        max_speed = float(getattr(map_config, 'tracker_speed', 2.4))
        max_turn_deg = float(getattr(map_config, "tracker_max_angular_speed", 10.0))
        dt = 1.0 / 40.0
        phys_max_w = math.radians(max_turn_deg) / dt
        
        if self.recovery_mode:
            # Recovery Mode: Drive towards the open space
            # Calculate bearing to the locked global angle
            target_bearing_rad = self.recovery_target_global_angle - heading_rad
            target_bearing_rad = (target_bearing_rad + math.pi) % (2*math.pi) - math.pi
            
            # Set high gains for recovery
            v_ref = max_speed
            w_ref = np.clip(TRACKER_ANGULAR_GAIN * target_bearing_rad, -phys_max_w, phys_max_w)
            
            # Use CBF to ensure we don't crash while escaping
            u_ref = np.array([v_ref, w_ref])
            u_star = self.controller.solve_cbf_qp(u_ref, closest_obs)
            
        else:
            # Normal Logic
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
            
            if has_target:
                # 追踪模式：朝向目标前进
                # 使用 Follow the Gap 启发式算法修正目标方向，避免直冲障碍物
                nav_bearing_rad = _get_navigation_heading(radar_norm, heading_rad, target_bearing_rad, max_range)
                
                v_ref = np.clip(TRACKER_VELOCITY_GAIN * target_dist, 0, max_speed)
                w_ref = np.clip(TRACKER_ANGULAR_GAIN * nav_bearing_rad, -phys_max_w, phys_max_w)
            else:
                # 搜索模式：原地旋转
                v_ref = 0.0
                w_ref = phys_max_w * SEARCH_SPIN_SPEED_FACTOR
            
            u_ref = np.array([v_ref, w_ref])
            
            # 5. 使用预初始化的控制器求解QP
            u_star = self.controller.solve_cbf_qp(u_ref, closest_obs)

        v_cmd, w_cmd = u_star[0], u_star[1]
        
        # 7. 转换为动作 (Velocity Command)
        angle_delta_deg = math.degrees(w_cmd * dt)
        angle_out = float(np.clip(angle_delta_deg, -max_turn_deg, max_turn_deg))
        speed_factor = min(v_cmd / max_speed, 1.0) if max_speed > 0 else 0.0
        
        # 8. 应用硬掩码（安全层）
        # raw_vel_action: (angle_norm, speed_norm)
        raw_vel_action = Model.to_normalized_action((angle_out, speed_factor))
        safe_vel_action = apply_hard_mask(raw_vel_action, radar_norm, heading_deg, role='tracker')
        
        # 9. Convert to Acceleration
        # safe_vel_action is (angle_norm, speed_norm)
        safe_angle_norm, safe_speed_norm = safe_vel_action
        
        # Recover desired values
        desired_angle_delta = safe_angle_norm * max_turn_deg
        desired_speed = safe_speed_norm * max_speed # safe_speed_norm is [-1, 1]?
        # Model.to_normalized_action maps speed [0, 1] -> [-1, 1].
        # So safe_speed_norm is [-1, 1].
        # desired_speed = (safe_speed_norm + 1) / 2 * max_speed?
        # Wait, apply_hard_mask returns what Model.to_normalized_action returns.
        # Model.to_normalized_action returns [-1, 1].
        # So yes.
        desired_speed = (safe_speed_norm + 1.0) / 2.0 * max_speed
        
        # Desired Heading
        desired_heading_deg = heading_deg + desired_angle_delta
        
        # Current State
        current_vel = (obs[0] + 1.0) / 2.0 * max_speed
        current_w_norm = obs[1]
        current_w = current_w_norm * max_turn_deg # deg/step
        
        # Calculate Acceleration
        # Linear P-controller
        Kp_v = 2.0
        lin_acc = Kp_v * (desired_speed - current_vel)
        max_acc = float(getattr(map_config, 'tracker_max_acc', 0.1))
        lin_acc = np.clip(lin_acc, -max_acc, max_acc)
        lin_acc_norm = lin_acc / max_acc
        
        # Angular PD-controller
        max_ang_acc = float(getattr(map_config, 'tracker_max_ang_acc', 2.0))
        angle_diff = _normalize_angle(desired_heading_deg - heading_deg)
        
        # Tuned Gains (same as rule_policies.py)
        Kp_heading = 4.0
        Kd_heading = 0.5
        
        desired_w = Kp_heading * angle_diff
        if current_w != 0.0:
            desired_w -= Kd_heading * current_w
            
        desired_w = np.clip(desired_w, -max_turn_deg, max_turn_deg)
        
        Kp_w = 2.0
        ang_acc = Kp_w * (desired_w - current_w)
        
        ang_acc = np.clip(ang_acc, -max_ang_acc, max_ang_acc)
        ang_acc_norm = ang_acc / max_ang_acc
        
        return np.array([ang_acc_norm, lin_acc_norm], dtype=np.float32)
