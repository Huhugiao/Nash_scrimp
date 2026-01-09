import math
import numpy as np
import map_config

import pygame
import pygame.gfxdraw

# --- Numba Acceleration ---
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorators
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    def prange(*args): return range(*args)

# === Occupancy grid cache ===
_OCC_GRID = None
_OCC_CELL = None
_OCC_W = None
_OCC_H = None
_OCC_VALID = False

def _grid_dims(width, height, cell):
    nx = int(math.ceil(float(width) / float(cell)))
    ny = int(math.ceil(float(height) / float(cell)))
    return nx, ny

def _world_to_cell(x, y, cell):
    gx = float(x) / float(cell)
    gy = float(y) / float(cell)
    return int(math.floor(gx)), int(math.floor(gy)), gx, gy

def _clip_idx(ix, iy, nx, ny):
    return 0 <= ix < nx and 0 <= iy < ny

def _mark_rect(grid, cell, x, y, w, h):
    nx, ny = grid.shape[1], grid.shape[0]
    x1 = max(0, int(math.floor(x / cell)))
    y1 = max(0, int(math.floor(y / cell)))
    x2 = min(nx, int(math.ceil((x + w) / cell)))
    y2 = min(ny, int(math.ceil((y + h) / cell)))
    if x2 > x1 and y2 > y1:
        grid[y1:y2, x1:x2] = True

def _mark_circle(grid, cell, cx, cy, r):
    # 保留接口（当前未使用圆形障碍）
    nx, ny = grid.shape[1], grid.shape[0]
    x1 = max(0, int(math.floor((cx - r) / cell)))
    y1 = max(0, int(math.floor((cy - r) / cell)))
    x2 = min(nx, int(math.ceil((cx + r) / cell)))
    y2 = min(ny, int(math.ceil((cy + r) / cell)))
    rr = float(r) ** 2
    for iy in range(y1, y2):
        cyc = (iy + 0.5) * cell
        dy2 = (cyc - cy) ** 2
        for ix in range(x1, x2):
            cxc = (ix + 0.5) * cell
            if (cxc - cx) ** 2 + dy2 <= rr:
                grid[iy, ix] = True

def _dist2_point_to_segment(px, py, x1, y1, x2, y2):
    vx, vy = (x2 - x1), (y2 - y1)
    wx, wy = (px - x1), (py - y1)
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-9:
        return (px - x1)**2 + (py - y1)**2
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return (px - proj_x)**2 + (py - proj_y)**2

def _mark_segment(grid, cell, x1, y1, x2, y2, thick):
    nx, ny = grid.shape[1], grid.shape[0]
    pad = float(thick) * 0.5
    xmin, xmax = min(x1, x2) - pad, max(x1, x2) + pad
    ymin, ymax = min(y1, y2) - pad, max(y1, y2) + pad
    ix1 = max(0, int(math.floor(xmin / cell)))
    iy1 = max(0, int(math.floor(ymin / cell)))
    ix2 = min(nx, int(math.ceil(xmax / cell)))
    iy2 = min(ny, int(math.ceil(ymax / cell)))
    r2 = pad ** 2
    for iy in range(iy1, iy2):
        cy = (iy + 0.5) * cell
        for ix in range(ix1, ix2):
            cx = (ix + 0.5) * cell
            if _dist2_point_to_segment(cx, cy, x1, y1, x2, y2) <= r2:
                grid[iy, ix] = True

# --- Numba Kernels ---
@njit(fastmath=True)
def _numba_ray_cast_kernel(ox, oy, angle, max_range, grid, cell_size, nx, ny, pad_cells):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Avoid division by zero
    if abs(cos_a) < 1e-9: cos_a = 1e-9 if cos_a >= 0 else -1e-9
    if abs(sin_a) < 1e-9: sin_a = 1e-9 if sin_a >= 0 else -1e-9

    # World to Grid
    gx = ox / cell_size
    gy = oy / cell_size
    ix = int(gx)
    iy = int(gy)

    # Check bounds (start point)
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        return 0.0

    # Check start point collision
    if pad_cells > 0:
        x1 = max(0, ix - pad_cells)
        x2 = min(nx, ix + pad_cells + 1)
        y1 = max(0, iy - pad_cells)
        y2 = min(ny, iy + pad_cells + 1)
        if np.any(grid[y1:y2, x1:x2]):
            return 0.0
    else:
        if grid[iy, ix]:
            return 0.0

    # DDA Setup
    step_x = 1 if cos_a > 0 else -1
    step_y = 1 if sin_a > 0 else -1

    cell_x = float(ix)
    cell_y = float(iy)

    if step_x > 0:
        dist_to_vx = (cell_x + 1.0 - gx) * cell_size
    else:
        dist_to_vx = (gx - cell_x) * cell_size

    if step_y > 0:
        dist_to_vy = (cell_y + 1.0 - gy) * cell_size
    else:
        dist_to_vy = (gy - cell_y) * cell_size

    tMaxX = dist_to_vx / abs(cos_a)
    tMaxY = dist_to_vy / abs(sin_a)

    tDeltaX = cell_size / abs(cos_a)
    tDeltaY = cell_size / abs(sin_a)

    dist = 0.0
    
    while dist <= max_range:
        if tMaxX < tMaxY:
            dist = tMaxX
            tMaxX += tDeltaX
            ix += step_x
        else:
            dist = tMaxY
            tMaxY += tDeltaY
            iy += step_y

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            break
            
        # Check collision
        if pad_cells > 0:
            x1 = max(0, ix - pad_cells)
            x2 = min(nx, ix + pad_cells + 1)
            y1 = max(0, iy - pad_cells)
            y2 = min(ny, iy + pad_cells + 1)
            if np.any(grid[y1:y2, x1:x2]):
                return min(dist, max_range)
        else:
            if grid[iy, ix]:
                return min(dist, max_range)

    return max_range

@njit(fastmath=True)
def _numba_ray_cast_batch(ox, oy, angles, max_range, grid, cell_size, nx, ny, pad_cells):
    n = len(angles)
    res = np.empty(n, dtype=np.float32)
    for i in range(n):
        res[i] = _numba_ray_cast_kernel(ox, oy, angles[i], max_range, grid, cell_size, nx, ny, pad_cells)
    return res

def build_occupancy(width=None, height=None, cell=None, obstacles=None):
    """根据当前地图和障碍构建占据栅格，用于快速射线/碰撞查询。"""
    global _OCC_GRID, _OCC_CELL, _OCC_W, _OCC_H, _OCC_VALID
    width = float(width or map_config.width)
    height = float(height or map_config.height)
    cell = float(cell or getattr(map_config, 'occ_cell', getattr(map_config, 'pixel_size', 8.0)))
    obstacles = obstacles or getattr(map_config, 'obstacles', [])

    nx, ny = _grid_dims(width, height, cell)
    grid = np.zeros((ny, nx), dtype=np.bool_)

    for obs in obstacles:
        typ = obs.get('type')
        if typ == 'rect':
            _mark_rect(grid, cell, float(obs['x']), float(obs['y']), float(obs['w']), float(obs['h']))
        elif typ == 'circle':
            _mark_circle(grid, cell, float(obs['cx']), float(obs['cy']), float(obs['r']))
        elif typ == 'segment':
            _mark_segment(
                grid, cell,
                float(obs['x1']), float(obs['y1']),
                float(obs['x2']), float(obs['y2']),
                float(obs.get('thick', 8.0))
            )

    _OCC_GRID, _OCC_CELL, _OCC_W, _OCC_H, _OCC_VALID = grid, cell, width, height, True

def _occ_available():
    return _OCC_VALID and _OCC_GRID is not None

def _occ_any_with_pad(ix, iy, pad_cells):
    ny, nx = _OCC_GRID.shape
    if pad_cells <= 0:
        return not _clip_idx(ix, iy, nx, ny) or bool(_OCC_GRID[iy, ix])
    x1, x2 = max(0, ix - pad_cells), min(nx, ix + pad_cells + 1)
    y1, y2 = max(0, iy - pad_cells), min(ny, iy + pad_cells + 1)
    if x1 >= x2 or y1 >= y2:
        return True
    return bool(_OCC_GRID[y1:y2, x1:x2].any())

# === Analytical Ray Intersection Helpers ===

def _ray_intersect_circle(ox, oy, dx, dy, cx, cy, r):
    fx = ox - cx
    fy = oy - cy
    b = 2 * (dx * fx + dy * fy)
    c = (fx * fx + fy * fy) - r * r
    disc = b * b - 4 * c
    if disc < 0:
        return float('inf')
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2
    t2 = (-b + sqrt_disc) / 2
    
    if t2 < 0: return float('inf')
    if t1 > 0: return t1
    return t2 # Inside circle

def _ray_intersect_segment_line(ox, oy, dx, dy, x1, y1, x2, y2):
    rx = x2 - x1
    ry = y2 - y1
    cross = dx * ry - dy * rx
    if abs(cross) < 1e-9:
        return float('inf')
    
    dp_x = x1 - ox
    dp_y = y1 - oy
    
    t = (dp_x * ry - dp_y * rx) / cross
    u = (dp_x * dy - dp_y * dx) / cross
    
    if 0 <= u <= 1 and t >= 0:
        return t
    return float('inf')

def _ray_intersect_capsule_segment(ox, oy, dx, dy, p1x, p1y, p2x, p2y, r):
    dist = _ray_intersect_circle(ox, oy, dx, dy, p1x, p1y, r)
    dist = min(dist, _ray_intersect_circle(ox, oy, dx, dy, p2x, p2y, r))
    
    vx = p2x - p1x
    vy = p2y - p1y
    length = math.hypot(vx, vy)
    if length < 1e-9: return dist
    
    nx = -vy / length
    ny = vx / length
    
    off_x = nx * r
    off_y = ny * r
    
    t1 = _ray_intersect_segment_line(ox, oy, dx, dy, p1x + off_x, p1y + off_y, p2x + off_x, p2y + off_y)
    t2 = _ray_intersect_segment_line(ox, oy, dx, dy, p1x - off_x, p1y - off_y, p2x - off_x, p2y - off_y)
    
    return min(dist, t1, t2)

def _ray_intersect_rect(ox, oy, dx, dy, rx, ry, rw, rh):
    t_min = float('inf')
    if abs(dx) > 1e-9:
        t = (rx - ox) / dx
        y = oy + t * dy
        if t >= 0 and ry <= y <= ry + rh: t_min = min(t_min, t)
        t = (rx + rw - ox) / dx
        y = oy + t * dy
        if t >= 0 and ry <= y <= ry + rh: t_min = min(t_min, t)
    if abs(dy) > 1e-9:
        t = (ry - oy) / dy
        x = ox + t * dx
        if t >= 0 and rx <= x <= rx + rw: t_min = min(t_min, t)
        t = (ry + rh - oy) / dy
        x = ox + t * dx
        if t >= 0 and rx <= x <= rx + rw: t_min = min(t_min, t)
    return t_min

def _ray_intersect_obstacles(origin, angle_rad, max_range, padding=0.0):
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    
    min_dist = float(max_range)
    
    for obs in getattr(map_config, 'obstacles', []):
        d = float('inf')
        if obs['type'] == 'rect':
            d = _ray_intersect_rect(ox, oy, dx, dy, obs['x'] - padding, obs['y'] - padding, obs['w'] + 2*padding, obs['h'] + 2*padding)
        elif obs['type'] == 'circle':
            d = _ray_intersect_circle(ox, oy, dx, dy, obs['cx'], obs['cy'], obs['r'] + padding)
        elif obs.get('type') == 'segment':
            thick = float(obs.get('thick', 8.0))
            radius = thick * 0.5 + padding
            d = _ray_intersect_capsule_segment(ox, oy, dx, dy, obs['x1'], obs['y1'], obs['x2'], obs['y2'], radius)
        
        if d < min_dist:
            min_dist = d
            
    return min_dist

def ray_distance_grid(origin, angle_rad, max_range, padding=0.0):
    """
    执行射线检测。
    默认使用精确解析几何计算 (_ray_intersect_obstacles)，保证雷达精度。
    不再使用占据栅格 (Occupancy Grid) 进行射线检测。
    """
    return _ray_intersect_obstacles(origin, angle_rad, max_range, padding)

def _ray_distance_grid_legacy(origin, angle_rad, max_range, padding=0.0):
    """
    (Legacy) 在占据栅格上沿方向 angle_rad 执行 DDA 射线。
    保留供参考或性能回退（但当前被 ray_distance_grid 取代）。
    """
    if not _occ_available():
        return _trace_ray_for_fov(origin, angle_rad, max_range)

    ox, oy = float(origin[0]), float(origin[1])
    
    if NUMBA_AVAILABLE:
        nx, ny = _OCC_GRID.shape[1], _OCC_GRID.shape[0]
        pad_cells = int(math.ceil(float(padding) / float(_OCC_CELL)))
        return float(_numba_ray_cast_kernel(ox, oy, float(angle_rad), float(max_range), 
                                          _OCC_GRID, float(_OCC_CELL), nx, ny, pad_cells))

    # Fallback to pure Python DDA
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    # 为了避免“起点所在 cell 提前命中”的问题，我们显式跳过起始 cell，
    # 从穿出当前 cell 边界之后才开始检测占据情况。
    eps = 1e-12
    cos_a = cos_a if abs(cos_a) > eps else (eps if cos_a >= 0 else -eps)
    sin_a = sin_a if abs(sin_a) > eps else (eps if sin_a >= 0 else -eps)

    ix, iy, gx, gy = _world_to_cell(ox, oy, _OCC_CELL)
    nx, ny = _OCC_GRID.shape[1], _OCC_GRID.shape[0]

    # 起点落在栅格外，视为立刻撞墙
    if not _clip_idx(ix, iy, nx, ny):
        return 0.0

    pad_cells = int(math.ceil(float(padding) / float(_OCC_CELL)))

    # === 关键修改 1：起点 cell 只用于“起点是否在障碍内”的检查 ===
    # 若起点 cell 就在障碍内，则返回 0（即贴在墙上）
    if _occ_any_with_pad(ix, iy, pad_cells):
        return 0.0

    # === 关键修改 2：DDA 从“穿出当前 cell 边界”的那一刻开始 ===
    step_x = 1 if cos_a > 0 else -1
    step_y = 1 if sin_a > 0 else -1

    # grid 坐标：gx, gy 是连续的 cell 坐标（单位：cell）
    # 计算当前 cell 的左/右/上/下边界（以 cell 坐标表示）
    cell_x = math.floor(gx)
    cell_y = math.floor(gy)

    if step_x > 0:
        next_vx = cell_x + 1.0  # 右边界
        dist_to_vx = (next_vx - gx) * _OCC_CELL
    else:
        next_vx = cell_x       # 左边界
        dist_to_vx = (gx - next_vx) * _OCC_CELL

    if step_y > 0:
        next_vy = cell_y + 1.0  # 下边界
        dist_to_vy = (next_vy - gy) * _OCC_CELL
    else:
        next_vy = cell_y        # 上边界
        dist_to_vy = (gy - next_vy) * _OCC_CELL

    # tMaxX/Y 表示从起点出发走到“穿出当前 cell 到达下一条网格线”的物理距离
    tMaxX = dist_to_vx / abs(cos_a)
    tMaxY = dist_to_vy / abs(sin_a)

    # 每跨一条竖线/横线，需要增加的距离
    tDeltaX = _OCC_CELL / abs(cos_a)
    tDeltaY = _OCC_CELL / abs(sin_a)

    dist = 0.0
    while dist <= max_range and _clip_idx(ix, iy, nx, ny):
        # 每次循环都表示“跨出一个 cell”，然后检查新 cell
        if tMaxX < tMaxY:
            dist = tMaxX
            tMaxX += tDeltaX
            ix += step_x
        else:
            dist = tMaxY
            tMaxY += tDeltaY
            iy += step_y

        if not _clip_idx(ix, iy, nx, ny):
            break

        if _occ_any_with_pad(ix, iy, pad_cells):
            return float(min(dist, max_range))

    return float(max_range)

def ray_distances_multi(origin, angles_rad, max_range, padding=0.0):
    # 强制不使用 Numba/OccGrid 路径，以确保使用精确的几何计算
    # if NUMBA_AVAILABLE and _occ_available(): ...
    
    return np.array([ray_distance_grid(origin, ang, max_range, padding) for ang in angles_rad],
                    dtype=np.float32)

def reward_calculate(tracker, target, prev_tracker=None, prev_target=None,
                     tracker_collision=False, target_collision=False,
                     sector_captured=False, capture_progress=0, capture_required_steps=0,
                     radar=None):
    """计算奖励函数"""
    info = {
        'capture_progress': int(capture_progress),
        'capture_required_steps': int(capture_required_steps),
        'tracker_collision': bool(tracker_collision),
        'target_collision': bool(target_collision)
    }

    reward = 0.0
    terminated = False

    dx = (tracker['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    dy = (tracker['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    curr_dist = math.hypot(dx, dy)

    if prev_tracker is not None and prev_target is not None:
        p_dx = (prev_tracker['x'] + map_config.pixel_size * 0.5) - (prev_target['x'] + map_config.pixel_size * 0.5)
        p_dy = (prev_tracker['y'] + map_config.pixel_size * 0.5) - (prev_target['y'] + map_config.pixel_size * 0.5)
        prev_dist = math.hypot(p_dx, p_dy)
    else:
        prev_dist = curr_dist

    alpha = 0.05
    reward += alpha * (prev_dist - curr_dist)
    reward -= 0.01  # time penalty

    # Proximity penalty/reward: 基于到最近障碍物的垂直距离变化
    if radar is not None and len(radar) > 0:
        max_range = float(getattr(map_config, 'fov_range', 250.0))
        agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
        
        # 当前帧：找到最近障碍物的边缘距离
        min_radar_normalized = float(min(radar))
        curr_center_dist = (min_radar_normalized + 1.0) * 0.5 * max_range
        curr_edge_dist = curr_center_dist - agent_radius
        
        # 前一帧：如果有 prev_tracker，用其位置重新计算雷达（需要环境提供）
        # 这里简化处理：假设 info 中可以传递 prev_radar
        # 如果没有，则使用当前距离作为基准（首次调用）
        if prev_tracker is not None and 'prev_radar' in info:
            prev_radar = info['prev_radar']
            prev_min_normalized = float(min(prev_radar))
            prev_center_dist = (prev_min_normalized + 1.0) * 0.5 * max_range
            prev_edge_dist = prev_center_dist - agent_radius
        else:
            # 首次或无前帧数据，不施加速度奖励
            prev_edge_dist = curr_edge_dist
        
        # 计算垂直距离变化速度
        obstacle_velocity = curr_edge_dist - prev_edge_dist  # 正值=远离，负值=靠近
        
        # 奖励/惩罚系数
        velocity_coef = 0.02  # 可调整强度
        
        if obstacle_velocity < 0:
            # 靠近障碍物：施加惩罚
            proximity_penalty = velocity_coef * obstacle_velocity  # 负值惩罚
            reward += proximity_penalty
            info['obstacle_approaching'] = True
            info['obstacle_velocity'] = float(obstacle_velocity)
            info['proximity_penalty'] = float(proximity_penalty)
        else:
            # 远离障碍物：施加奖励
            proximity_reward = velocity_coef * obstacle_velocity
            reward += proximity_reward
            info['obstacle_receding'] = True
            info['obstacle_velocity'] = float(obstacle_velocity)
            info['proximity_reward'] = float(proximity_reward)
        
        # 记录距离信息
        info['min_edge_distance'] = float(curr_edge_dist)

    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    if sector_captured:
        terminated = True
        info['reason'] = 'tracker_caught_target'
        reward += success_reward
    elif tracker_collision:
        terminated = True
        reward -= success_reward
        info['reason'] = 'tracker_collision'

    return float(reward), bool(terminated), False, info

def _to_hi_res(pt):
    ss = getattr(map_config, 'ssaa', 1)
    return int(round(pt[0] * ss)), int(round(pt[1] * ss))

def _draw_grid(surface):
    if pygame is None or not getattr(map_config, 'draw_grid', True):
        return
    ss = getattr(map_config, 'ssaa', 1)
    step = int(map_config.grid_step * ss)
    w, h = surface.get_size()
    for x in range(0, w, step):
        pygame.draw.line(surface, map_config.grid_color, (x, 0), (x, h), 1)
    for y in range(0, h, step):
        pygame.draw.line(surface, map_config.grid_color, (0, y), (w, y), 1)

def _draw_agent(surface, agent, color, role=None):
    """
    绘制智能体：
    - 外圈：圆环表示碰撞体积 (agent_radius)
    - 内部：箭头表示朝向
    """
    if pygame is None:
        return

    ss = float(getattr(map_config, 'ssaa', 1))
    x_world = float(agent['x']) + float(map_config.pixel_size) * 0.5
    y_world = float(agent['y']) + float(map_config.pixel_size) * 0.5
    cx, cy = int(x_world * ss), int(y_world * ss)

    if role == 'target':
        r_world = getattr(map_config, 'target_radius', getattr(map_config, 'agent_radius', 8.0))
    elif role == 'tracker':
        r_world = getattr(map_config, 'tracker_radius', getattr(map_config, 'agent_radius', 8.0))
    else:
        r_world = getattr(map_config, 'agent_radius', 8.0)
        
    r_i = max(3, int(r_world * ss))
    
    thickness = max(1, int(1.5 * ss))
    pygame.draw.circle(surface, color[:3], (cx, cy), r_i, thickness)
    
    theta_deg = agent.get('theta', 0.0)
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    tip_len = r_i * 0.8
    base_len = r_i * 0.4
    wing_len = r_i * 0.5
    
    p1 = (cx + tip_len * cos_t, cy + tip_len * sin_t)
    p2 = (cx - base_len * cos_t - wing_len * sin_t, cy - base_len * sin_t + wing_len * cos_t)
    p_indent = (cx - (base_len * 0.5) * cos_t, cy - (base_len * 0.5) * sin_t)
    p3 = (cx - base_len * cos_t + wing_len * sin_t, cy - base_len * sin_t - wing_len * cos_t)
    
    pygame.draw.polygon(surface, color[:3], [p1, p2, p_indent, p3])

def _draw_trail(surface, traj, rgba, width_px):
    if pygame is None or len(traj) < 2:
        return
    ss = getattr(map_config, 'ssaa', 1)
    max_len = getattr(map_config, 'trail_max_len', 600)
    points = traj[-max_len:]
    if len(points) < 2:
        return
        
    screen_pts = [_to_hi_res(p) for p in points]
    r, g, b = rgba[:3]
    base_alpha = rgba[3] if len(rgba) > 3 else 200
    w = max(int(width_px * ss), 1)
    
    # 渐变轨迹：从旧到新，透明度逐渐增加
    n = len(screen_pts)
    for i in range(n - 1):
        progress = i / max(1, n - 1)
        alpha = int(base_alpha * (progress ** 1.5)) # 稍微非线性，尾巴消失得快
        if alpha < 10: continue
        
        color = (r, g, b, alpha)
        start = screen_pts[i]
        end = screen_pts[i+1]
        pygame.draw.line(surface, color, start, end, w)

def _rect_contains(px, py, rect, padding=0.0):
    return (rect['x'] - padding <= px <= rect['x'] + rect['w'] + padding and
            rect['y'] - padding <= py <= rect['y'] + rect['h'] + padding)

def _circle_contains(px, py, circ, padding=0.0):
    return math.hypot(px - circ['cx'], py - circ['cy']) <= circ['r'] + padding

def _segment_contains(px, py, seg, padding=0.0):
    thick = float(seg.get('thick', 8.0))
    return _dist2_point_to_segment(px, py, seg['x1'], seg['y1'], seg['x2'], seg['y2']) <= (0.5 * thick + padding)**2

def is_point_blocked(px, py, padding=0.0):
    """点是否与任何障碍碰撞；使用精确几何检测。
    
    注意：为了保证物理碰撞的精确性，这里不再使用 Occupancy Grid (栅格)。
    虽然栅格能加速，但在边缘处会导致精度损失（虚边问题），影响训练和可视化一致性。
    对于单点查询，直接遍历几何体的开销是可以接受的。
    """
    # 这一行被注释掉，以强制使用精确检测
    # if _occ_available(): ...

    for obs in getattr(map_config, 'obstacles', []):
        if obs['type'] == 'rect' and _rect_contains(px, py, obs, padding):
            return True
        if obs['type'] == 'circle' and _circle_contains(px, py, obs, padding):
            return True
        if obs.get('type') == 'segment' and _segment_contains(px, py, obs, padding):
            return True
    return False

def find_colliding_obstacle(px, py, padding=0.0):
    """
    找到与指定点碰撞的障碍物及碰撞点信息。
    
    Args:
        px, py: 点的坐标
        padding: 碰撞检测的膨胀半径
        
    Returns:
        dict: 包含碰撞信息，或 None 如果没有碰撞
            - 'collision': bool, 是否碰撞
            - 'collision_point': tuple (x, y), 碰撞点坐标
            - 'collided_obstacle': dict, 被撞障碍物对象
    """
    for obs in getattr(map_config, 'obstacles', []):
        if obs['type'] == 'rect':
            if _rect_contains(px, py, obs, padding):
                # 计算最近的碰撞点 (矩形边上最近点)
                cx = max(obs['x'], min(px, obs['x'] + obs['w']))
                cy = max(obs['y'], min(py, obs['y'] + obs['h']))
                return {
                    'collision': True,
                    'collision_point': (cx, cy),
                    'collided_obstacle': obs
                }
        elif obs['type'] == 'circle':
            if _circle_contains(px, py, obs, padding):
                # 计算碰撞点 (圆边上最近点)
                dx = px - obs['cx']
                dy = py - obs['cy']
                dist = math.hypot(dx, dy)
                if dist > 1e-6:
                    cx = obs['cx'] + dx / dist * obs['r']
                    cy = obs['cy'] + dy / dist * obs['r']
                else:
                    cx, cy = obs['cx'], obs['cy'] - obs['r']
                return {
                    'collision': True,
                    'collision_point': (cx, cy),
                    'collided_obstacle': obs
                }
        elif obs.get('type') == 'segment':
            if _segment_contains(px, py, obs, padding):
                # 计算最近点到线段的投影
                x1, y1, x2, y2 = obs['x1'], obs['y1'], obs['x2'], obs['y2']
                vx, vy = (x2 - x1), (y2 - y1)
                wx, wy = (px - x1), (py - y1)
                seg_len2 = vx * vx + vy * vy
                if seg_len2 <= 1e-9:
                    cx, cy = x1, y1
                else:
                    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
                    cx = x1 + t * vx
                    cy = y1 + t * vy
                return {
                    'collision': True,
                    'collision_point': (cx, cy),
                    'collided_obstacle': obs
                }
    return None

def _resolve_obstacle_collision(old_pos, new_pos):
    if is_point_blocked(new_pos['x'] + map_config.pixel_size * 0.5,
                        new_pos['y'] + map_config.pixel_size * 0.5):
        new_pos['x'], new_pos['y'] = old_pos['x'], old_pos['y']
    return new_pos

def _draw_obstacles(surface, exclude_obstacle=None):
    """绘制矩形边界 + 线段墙
    
    Args:
        exclude_obstacle: 排除绘制的障碍物(被碰撞的障碍物单独高亮绘制)
    """
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    for obs in getattr(map_config, 'obstacles', []):
        # 跳过被排除的障碍物 (用于碰撞高亮)
        if exclude_obstacle is not None and obs is exclude_obstacle:
            continue
        color = obs.get('color', (80, 80, 80, 255))
        if obs['type'] == 'rect':
            pygame.draw.rect(
                surface, color,
                (int(obs['x']*ss), int(obs['y']*ss),
                 int(obs['w']*ss), int(obs['h']*ss))
            )
        elif obs['type'] == 'circle':
            pygame.draw.circle(
                surface, color,
                (int(obs['cx']*ss), int(obs['cy']*ss)),
                int(obs['r']*ss)
            )
        elif obs.get('type') == 'segment':
            pygame.draw.line(
                surface, color,
                (int(obs['x1']*ss), int(obs['y1']*ss)),
                (int(obs['x2']*ss), int(obs['y2']*ss)),
                max(1, int(float(obs.get('thick', 8.0))*ss))
            )

def _draw_collision_highlight(surface, collision_info):
    """高亮绘制被碰撞的障碍物"""
    if pygame is None or not collision_info:
        return
    
    obs = collision_info.get('collided_obstacle')
    if obs is None:
        return
        
    ss = getattr(map_config, 'ssaa', 1)
    color = getattr(map_config, 'COLLISION_OBSTACLE_COLOR', (180, 50, 255, 255))
    
    if obs['type'] == 'rect':
        # 绘制填充矩形
        pygame.draw.rect(
            surface, color,
            (int(obs['x']*ss), int(obs['y']*ss),
             int(obs['w']*ss), int(obs['h']*ss))
        )
        # 绘制发光边框效果
        border_color = (255, 100, 255, 255)
        pygame.draw.rect(
            surface, border_color,
            (int(obs['x']*ss)-2, int(obs['y']*ss)-2,
             int(obs['w']*ss)+4, int(obs['h']*ss)+4), 3
        )
    elif obs['type'] == 'circle':
        pygame.draw.circle(
            surface, color,
            (int(obs['cx']*ss), int(obs['cy']*ss)),
            int(obs['r']*ss)
        )
        # 绘制发光边框
        border_color = (255, 100, 255, 255)
        pygame.draw.circle(
            surface, border_color,
            (int(obs['cx']*ss), int(obs['cy']*ss)),
            int(obs['r']*ss)+3, 3
        )
    elif obs.get('type') == 'segment':
        pygame.draw.line(
            surface, color,
            (int(obs['x1']*ss), int(obs['y1']*ss)),
            (int(obs['x2']*ss), int(obs['y2']*ss)),
            max(1, int(float(obs.get('thick', 8.0))*ss)) + 4
        )

def _draw_collision_point(surface, collision_info):
    """绘制碰撞点标记 - 尺寸与智能体相当"""
    if pygame is None or not collision_info:
        return
        
    point = collision_info.get('collision_point')
    if point is None:
        return
        
    ss = getattr(map_config, 'ssaa', 1)
    color = getattr(map_config, 'COLLISION_POINT_COLOR', (255, 255, 0, 255))
    
    # 获取智能体半径作为参考尺寸
    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    
    cx, cy = int(point[0] * ss), int(point[1] * ss)
    r = int(agent_radius * ss * 0.5)  # 进一步缩小至0.6倍半径，更加精致
    
    # 绘制实心圆作为碰撞点
    pygame.draw.circle(surface, color, (cx, cy), r, 0)  # 实心
    # 绘制一圈白色边框增加可见度
    pygame.draw.circle(surface, (255, 255, 255, 255), (cx, cy), r + 2, 1)

def _draw_fov(surface, tracker, fov_points=None):
    """基于预计算的 fov_points 绘制半透明扇形。"""
    if pygame is None or not fov_points or len(fov_points) < 3:
        return
    
    # 1. 绘制填充 (Fill)
    fill_color = (80, 140, 255, 30) 
    pygame.gfxdraw.filled_polygon(surface, fov_points, fill_color)
    
    # 2. 绘制轮廓 (Outline) - 仅绘制两侧射线
    outline_color = (80, 140, 255, 200)
    
    center = fov_points[0]
    p_left = fov_points[1]
    p_right = fov_points[-1]
    
    # 转换为整数坐标
    c_int = (int(center[0]), int(center[1]))
    pl_int = (int(p_left[0]), int(p_left[1]))
    pr_int = (int(p_right[0]), int(p_right[1]))
    
    pygame.draw.line(surface, outline_color, c_int, pl_int, 1)
    pygame.draw.line(surface, outline_color, c_int, pr_int, 1)

def _trace_ray_for_fov(origin, angle_rad, max_range):
    """占据栅格 DDA 不可用时的备份射线（粗到细）。"""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dist, coarse_step = 0.0, max(map_config.pixel_size * 3.0, 6.0)
    hit = False
    while dist <= max_range:
        sx = origin[0] + cos_a * dist
        sy = origin[1] + sin_a * dist
        if (sx < 0 or sx > map_config.width or
                sy < 0 or sy > map_config.height or
                is_point_blocked(sx, sy)):
            hit = True
            break
        dist += coarse_step

    if not hit:
        return max_range

    lo, hi = max(0, dist - coarse_step), min(dist, max_range)
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        sx = origin[0] + cos_a * mid
        sy = origin[1] + sin_a * mid
        if (sx < 0 or sx > map_config.width or
                sy < 0 or sy > map_config.height or
                is_point_blocked(sx, sy)):
            hi = mid
        else:
            lo = mid
    return hi

def _draw_capture_sector(surface, tracker):
    """捕获扇区：继续使用 padding=0.0，避免提前被障碍截断。"""
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    cx_world = tracker['x'] + map_config.pixel_size * 0.5
    cy_world = tracker['y'] + map_config.pixel_size * 0.5
    cx, cy = cx_world * ss, cy_world * ss

    heading_rad = math.radians(tracker.get('theta', 0.0))
    half_sector = math.radians(getattr(map_config, 'capture_sector_angle_deg', 60.0)) * 0.5
    radius = float(getattr(map_config, 'capture_radius', 10.0))

    pts = [(cx, cy)]
    num_rays = 20
    for i in range(num_rays + 1):
        ang = heading_rad - half_sector + (2 * half_sector * i / num_rays)
        dist = ray_distance_grid(
            (cx_world, cy_world),
            ang,
            radius,
            padding=0.0   # 这里也不再膨胀
        )
        pts.append((cx + dist * ss * math.cos(ang), cy + dist * ss * math.sin(ang)))

    if len(pts) > 2:
        # 1. 填充
        fill_color = getattr(map_config, 'CAPTURE_SECTOR_COLOR', (80, 200, 120, 40))
        pygame.gfxdraw.filled_polygon(surface, pts, fill_color)
        
        # 2. 轮廓 - 增加清晰度
        outline_color = (80, 200, 120, 200)
        pygame.gfxdraw.aapolygon(surface, pts, outline_color)
        pygame.draw.lines(surface, outline_color, True, pts, 1)

def get_canvas(target, tracker, tracker_traj, target_traj, surface=None, fov_points=None, collision_info=None):
    """
    渲染画布。
    
    Args:
        collision_info: 可选的碰撞信息字典，包含:
            - 'collision': bool, 是否发生碰撞
            - 'collision_point': tuple (x, y), 碰撞点坐标
            - 'collided_obstacle': dict, 被撞障碍物信息
    """
    w, h = map_config.width, map_config.height
    ss = getattr(map_config, 'ssaa', 1)
    if pygame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    if surface is None:
        surface = pygame.Surface((w * ss, h * ss), flags=pygame.SRCALPHA)
    surface.fill(map_config.background_color)

    _draw_grid(surface)
    
    # 如果发生碰撞，需要先绘制普通障碍物，再高亮被撞障碍物
    if collision_info and collision_info.get('collision'):
        _draw_obstacles(surface, exclude_obstacle=collision_info.get('collided_obstacle'))
        _draw_collision_highlight(surface, collision_info)
    else:
        _draw_obstacles(surface)
    
    _draw_fov(surface, tracker, fov_points)
    _draw_capture_sector(surface, tracker)
    _draw_trail(surface, tracker_traj, map_config.trail_color_tracker, map_config.trail_width)
    _draw_trail(surface, target_traj, map_config.trail_color_target, map_config.trail_width)
    
    # 如果发生碰撞，用特殊颜色绘制tracker
    if collision_info and collision_info.get('collision'):
        collision_color = getattr(map_config, 'COLLISION_TRACKER_COLOR', (255, 50, 50, 255))
        _draw_agent(surface, tracker, collision_color, role='tracker')
        _draw_collision_point(surface, collision_info)
    else:
        _draw_agent(surface, tracker, map_config.tracker_color, role='tracker')
    
    _draw_agent(surface, target, map_config.target_color, role='target')

    canvas = pygame.transform.smoothscale(surface, (w, h)) if ss > 1 else surface
    return pygame.surfarray.array3d(canvas).swapaxes(0, 1)

def agent_move(agent, action, moving_size, role=None):
    angle_delta, speed_factor = float(action[0]), float(action[1])

    if role == 'tracker':
        max_turn = float(getattr(map_config, 'tracker_max_turn_deg', 45.0))
    elif role == 'target':
        max_turn = float(getattr(map_config, 'target_max_turn_deg', 45.0))
    else:
        max_turn = float(getattr(map_config, 'max_turn_deg', 45.0))

    angle_delta = float(np.clip(angle_delta, -max_turn, max_turn))
    speed = float(np.clip(speed_factor, 0.0, 1.0) * moving_size)

    new_angle = (agent.get('theta', 0.0) + angle_delta) % 360.0
    agent['theta'] = float(new_angle)

    old_state = dict(agent)
    rad_angle = math.radians(new_angle)
    agent['x'] = float(np.clip(agent['x'] + speed * math.cos(rad_angle),
                               0, map_config.width - map_config.pixel_size))
    agent['y'] = float(np.clip(agent['y'] + speed * math.sin(rad_angle),
                               0, map_config.height - map_config.pixel_size))

    return _resolve_obstacle_collision(old_state, agent)

def agent_move_accel(agent, lin_acc, ang_acc, max_speed, max_ang_speed, role=None, enable_safety_layer=True):
    """
    Acceleration-based movement update.
    Updates agent state (x, y, theta, v, w) in place.
    
    Args:
        enable_safety_layer: If False, disables collision rollback protection.
    """
    old_state = dict(agent)
    
    # Update velocities
    agent['v'] = float(agent.get('v', 0.0) + lin_acc)
    agent['w'] = float(agent.get('w', 0.0) + ang_acc)
    
    # Clip velocities
    # Assuming forward only for now as per original logic
    agent['v'] = float(np.clip(agent['v'], 0.0, max_speed))
    agent['w'] = float(np.clip(agent['w'], -max_ang_speed, max_ang_speed))
    
    # Update pose
    # Note: w is in degrees/step
    new_theta = (agent['theta'] + agent['w']) % 360.0
    agent['theta'] = float(new_theta)
    
    rad_theta = math.radians(new_theta)
    agent['x'] = float(np.clip(agent['x'] + agent['v'] * math.cos(rad_theta),
                               0, map_config.width - map_config.pixel_size))
    agent['y'] = float(np.clip(agent['y'] + agent['v'] * math.sin(rad_theta),
                               0, map_config.height - map_config.pixel_size))
    
    # Only apply collision rollback if safety layer is enabled
    if enable_safety_layer:
        return _resolve_obstacle_collision(old_state, agent)
    else:
        return agent
# ============================================================================
# Hard Mask Safety Parameters (Moved from cbf_controller.py)
# ============================================================================
HARD_MASK_SAFETY_MULTIPLIER = 2.5
HARD_MASK_CHECK_WINDOW = 1
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
    (Moved here to avoid circular dependency with cbf_controller/cvxpy)
    """
    if safety_dist is None:
        safety_dist = float(getattr(map_config, 'agent_radius', 8.0)) * HARD_MASK_SAFETY_MULTIPLIER

    if hasattr(action, '__len__'):
        angle_norm = float(action[0])
        speed_norm = float(action[1])
    else:
        return action

    # Max turn capabilities
    if role == 'tracker':
        max_turn = float(getattr(map_config, 'tracker_max_angular_speed', 10.0))
    else:
        max_turn = float(getattr(map_config, 'target_max_angular_speed', 12.0))
        
    angle_delta = angle_norm * max_turn
    target_heading = _normalize_angle(current_heading_deg + angle_delta)
    
    if len(radar) == 0:
        return (angle_norm, speed_norm)

    num_rays = len(radar)
    angle_step = 360.0 / num_rays
    
    th_360 = target_heading % 360.0
    center_idx = int(round(th_360 / angle_step)) % num_rays
    
    max_range = float(getattr(map_config, 'FOV_RANGE', 250.0))
    dists = (radar + 1.0) * 0.5 * max_range
    
    # Check safety
    is_safe = True
    for i in range(-HARD_MASK_CHECK_WINDOW, HARD_MASK_CHECK_WINDOW + 1):
        idx = (center_idx + i) % num_rays
        if dists[idx] <= safety_dist:
            is_safe = False
            break
            
    if is_safe:
        return (angle_norm, speed_norm)
        
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
                return (float(new_angle_norm), -1.0)
        return (angle_norm, -1.0)

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
    
    turn_magnitude = abs(needed_turn)
    if turn_magnitude > 60.0:
        new_speed_norm = -1.0
    elif turn_magnitude > 30.0:
        new_speed_norm = min(speed_norm, 0.0) 
    else:
        new_speed_norm = float(speed_norm)
    
    return (float(new_angle_norm), float(new_speed_norm))
