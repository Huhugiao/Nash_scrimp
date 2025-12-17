import math
import numpy as np
import map_config

import pygame
import pygame.gfxdraw

from numba import njit

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

# === Ray Casting Optimization ===
_RAY_CACHE = {}  # Cache for repeated ray queries
_CACHE_ENABLED = True
_CACHE_MAX_SIZE = 10000

def _get_ray_cache_key(ox, oy, angle, max_range, pad):
    """Generate cache key for ray queries (rounded to reduce cache misses)"""
    return (
        round(ox, 1), round(oy, 1), 
        round(angle, 3), round(max_range, 1), round(pad, 1)
    )

def ray_distance_grid(origin, angle_rad, max_range, padding=0.0):
    """优化的射线投射：直接使用Numba内核 + 缓存层"""
    if not _occ_available():
        return float(max_range)
    
    ox, oy = float(origin[0]), float(origin[1])
    
    # Cache lookup (optional, can be disabled for pure speed)
    if _CACHE_ENABLED:
        cache_key = _get_ray_cache_key(ox, oy, angle_rad, max_range, padding)
        if cache_key in _RAY_CACHE:
            return _RAY_CACHE[cache_key]
    
    # Use Numba kernel directly - this is the key optimization!
    nx, ny = _OCC_GRID.shape[1], _OCC_GRID.shape[0]
    pad_cells = int(math.ceil(float(padding) / float(_OCC_CELL)))
    
    result = float(_numba_ray_cast_kernel(
        ox, oy, float(angle_rad), float(max_range),
        _OCC_GRID, float(_OCC_CELL), nx, ny, pad_cells
    ))
    
    # Cache result
    if _CACHE_ENABLED and len(_RAY_CACHE) < _CACHE_MAX_SIZE:
        _RAY_CACHE[cache_key] = result
    
    return result

def clear_ray_cache():
    """Clear ray cache (call when obstacles change)"""
    global _RAY_CACHE
    _RAY_CACHE.clear()

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
    clear_ray_cache()

def ray_distances_multi(origin, angles_rad, max_range, padding=0.0):
    """
    批量射线投射：使用 Numba 加速的批量计算
    
    Args:
        origin: tuple/array (x, y) 世界坐标
        angles_rad: array-like 射线角度（弧度）
        max_range: float 最大射线距离
        padding: float 安全边距
        
    Returns:
        np.ndarray: 每条射线的距离
    """
    if not _occ_available():
        return np.full(len(angles_rad), max_range, dtype=np.float32)
    
    ox, oy = float(origin[0]), float(origin[1])
    nx, ny = _OCC_GRID.shape[1], _OCC_GRID.shape[0]
    pad_cells = int(math.ceil(float(padding) / float(_OCC_CELL)))
    
    # Ensure angles is numpy array
    angles_arr = np.asarray(angles_rad, dtype=np.float64)
    
    # Use Numba accelerated batch ray casting
    return _numba_ray_cast_batch(ox, oy, angles_arr, float(max_range), 
                                _OCC_GRID, float(_OCC_CELL), nx, ny, pad_cells)

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

def reward_calculate(tracker, target, prev_tracker=None, prev_target=None,
                     tracker_collision=False, target_collision=False,
                     sector_captured=False, capture_progress=0, capture_required_steps=0,
                     residual_action=None, action_penalty_coef=0.0):
    """
    计算奖励函数：
    1. 差分奖励 (Potential-based Reward Shaping):
       R_approach = alpha * (dist_{t-1} - dist_{t})
       - 鼓励智能体每一步都向目标靠近。
       - 相比绝对距离奖励，差分奖励能防止智能体在特定距离“刷分”，
         因为只有距离缩小时才有正奖励，原地不动或震荡的累积奖励为0。
    
    2. 时间惩罚 (Time Penalty):
       - 每步固定扣分，迫使智能体以最短路径完成任务。
    
    3. 终局奖励 (Terminal Reward):
       - 捕获成功给予大额正奖励。
       - 发生碰撞给予大额负奖励。
    """
    info = {
        'capture_progress': int(capture_progress),
        'capture_required_steps': int(capture_required_steps),
        'tracker_collision': bool(tracker_collision),
        'target_collision': bool(target_collision),
        'action_penalty': 0.0
    }

    reward = 0.0

    # Residual Action Penalty (L2 regularization)
    if residual_action is not None and action_penalty_coef > 0:
        # Penalize squared magnitude of residual action
        action_penalty = action_penalty_coef * np.sum(np.square(residual_action))
        reward -= action_penalty
        info['action_penalty'] = -float(action_penalty)

    # --- Dense shaping 部分（每步）---
    terminated = False

    # 1) 差分奖励：R = alpha * (prev_dist - curr_dist)
    # 计算当前距离
    dx = (tracker['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    dy = (tracker['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    curr_dist = math.hypot(dx, dy)

    # 计算上一时刻距离
    if prev_tracker is not None and prev_target is not None:
        p_dx = (prev_tracker['x'] + map_config.pixel_size * 0.5) - (prev_target['x'] + map_config.pixel_size * 0.5)
        p_dy = (prev_tracker['y'] + map_config.pixel_size * 0.5) - (prev_target['y'] + map_config.pixel_size * 0.5)
        prev_dist = math.hypot(p_dx, p_dy)
    else:
        # 第一帧没有上一时刻，假设距离不变
        prev_dist = curr_dist

    # alpha 系数：决定引导力度
    alpha = 0.05
    reward_approach = alpha * (prev_dist - curr_dist)
    reward += reward_approach

    # 2) 时间惩罚：鼓励更快结束
    time_penalty = 0.01
    reward -= time_penalty

    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    if sector_captured:
        terminated = True
        info['reason'] = 'tracker_caught_target'
        reward += success_reward
    elif tracker_collision:
        terminated = True
        reward -= success_reward

    return float(reward), bool(terminated), False, info

def _to_hi_res(pt):
    ss = getattr(map_config, 'ssaa', 1)
    return int(round(pt[0] * ss)), int(round(pt[1] * ss))

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

    # Get agent radius
    if role == 'target':
        r_world = getattr(map_config, 'target_radius', getattr(map_config, 'agent_radius', 8.0))
    elif role == 'tracker':
        r_world = getattr(map_config, 'tracker_radius', getattr(map_config, 'agent_radius', 8.0))
    else:
        r_world = getattr(map_config, 'agent_radius', 8.0)
        
    r_i = max(3, int(r_world * ss))
    
    # 1. Draw outline circle
    thickness = max(1, int(1.5 * ss))
    pygame.draw.circle(surface, color[:3], (cx, cy), r_i, thickness)
    
    # 2. Draw arrow (triangle)
    theta_deg = agent.get('theta', 0.0)
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    # Arrow dimensions
    tip_len = r_i * 0.8  # Tip extends to 80% of radius
    base_len = r_i * 0.4 # Base extends back
    wing_len = r_i * 0.5 # Width
    
    # Vertices
    p1 = (cx + tip_len * cos_t, cy + tip_len * sin_t)
    p2 = (
        cx - base_len * cos_t - wing_len * sin_t,
        cy - base_len * sin_t + wing_len * cos_t
    )
    p_indent = (
        cx - (base_len * 0.5) * cos_t,
        cy - (base_len * 0.5) * sin_t
    )
    p3 = (
        cx - base_len * cos_t + wing_len * sin_t,
        cy - base_len * sin_t - wing_len * cos_t
    )
    
    pygame.draw.polygon(surface, color[:3], [p1, p2, p_indent, p3])

def _draw_grid(surface):
    if pygame is None or not getattr(map_config, 'draw_grid', True):
        return
    ss = getattr(map_config, 'ssaa', 1)
    step = int(getattr(map_config, 'grid_step', 40) * ss)
    if step <= 0:
        return
    w, h = surface.get_size()
    color = getattr(map_config, 'grid_color', (40, 40, 40, 60))
    for x in range(0, w, step):
        pygame.draw.line(surface, color, (x, 0), (x, h), 1)
    for y in range(0, h, step):
        pygame.draw.line(surface, color, (0, y), (w, y), 1)

def _draw_obstacles(surface):
    """仅绘制 map_config.obstacles（rect/circle/segment）。"""
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    for obs in getattr(map_config, 'obstacles', []):
        color = obs.get('color', (80, 80, 80, 255))
        typ = obs.get('type')
        if typ == 'rect':
            pygame.draw.rect(
                surface, color,
                (int(obs['x'] * ss), int(obs['y'] * ss),
                 int(obs['w'] * ss), int(obs['h'] * ss))
            )
        elif typ == 'circle':
            pygame.draw.circle(
                surface, color,
                (int(obs['cx'] * ss), int(obs['cy'] * ss)),
                int(float(obs['r']) * ss)
            )
        elif typ == 'segment':
            pygame.draw.line(
                surface, color,
                (int(obs['x1'] * ss), int(obs['y1'] * ss)),
                (int(obs['x2'] * ss), int(obs['y2'] * ss)),
                max(1, int(float(obs.get('thick', 8.0)) * ss))
            )

def _draw_fov(surface, tracker, fov_points=None):
    """基于预计算的 fov_points 绘制半透明扇形。"""
    if pygame is None or not fov_points or len(fov_points) < 3:
        return
    try:
        fill_color = (80, 140, 255, 30)
        outline_color = (80, 140, 255, 200)
        pygame.gfxdraw.filled_polygon(surface, fov_points, fill_color)

        # 只画两条侧边，保持清爽
        c = fov_points[0]
        pl = fov_points[1]
        pr = fov_points[-1]
        pygame.draw.line(surface, outline_color, (int(c[0]), int(c[1])), (int(pl[0]), int(pl[1])), 1)
        pygame.draw.line(surface, outline_color, (int(c[0]), int(c[1])), (int(pr[0]), int(pr[1])), 1)
    except Exception:
        pass

def _trace_ray_for_fov(origin, angle_rad, max_range):
    """occupancy 不可用时的备份射线（粗到细）。"""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dist, coarse_step = 0.0, max(getattr(map_config, 'pixel_size', 8.0) * 3.0, 6.0)
    hit = False
    while dist <= max_range:
        sx = origin[0] + cos_a * dist
        sy = origin[1] + sin_a * dist
        if (sx < 0 or sx > map_config.width or sy < 0 or sy > map_config.height or
                is_point_blocked(sx, sy)):
            hit = True
            break
        dist += coarse_step
    if not hit:
        return float(max_range)

    lo, hi = max(0.0, dist - coarse_step), min(dist, max_range)
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        sx = origin[0] + cos_a * mid
        sy = origin[1] + sin_a * mid
        if (sx < 0 or sx > map_config.width or sy < 0 or sy > map_config.height or
                is_point_blocked(sx, sy)):
            hi = mid
        else:
            lo = mid
    return float(hi)

def get_canvas(target, tracker, tracker_traj, target_traj, surface=None, fov_points=None):
    """渲染：背景 + 网格(可选) + 障碍物 + FOV + 红/蓝圆。"""
    w, h = map_config.width, map_config.height
    ss = getattr(map_config, 'ssaa', 1)
    if pygame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    if surface is None:
        surface = pygame.Surface((w * ss, h * ss), flags=pygame.SRCALPHA)
    surface.fill(map_config.background_color)

    # 重新开启：障碍物 & 视野渲染（你要看的东西）
    _draw_grid(surface)
    _draw_obstacles(surface)
    _draw_fov(surface, tracker, fov_points)

    # Draw capture sector (pie slice) for tracker
    tx = float(tracker['x']) + float(map_config.pixel_size) * 0.5
    ty = float(tracker['y']) + float(map_config.pixel_size) * 0.5
    t_cx, t_cy = int(tx * ss), int(ty * ss)
    cap_r = getattr(map_config, 'capture_radius', 0.0)
    cap_angle = getattr(map_config, 'capture_sector_angle_deg', 360.0)

    if cap_r > 0:
        r_cap_px = int(cap_r * ss)
        # If angle is close to 360, draw circle
        if cap_angle >= 359.0:
            pygame.draw.circle(surface, (0, 255, 255, 60), (t_cx, t_cy), r_cap_px, 1)
        else:
            # Draw sector
            heading = float(tracker.get('theta', 0.0))
            half_angle = cap_angle * 0.5
            start_angle = math.radians(heading - half_angle)
            end_angle = math.radians(heading + half_angle)
            
            # Interpolate arc points
            steps = max(2, int(cap_angle / 5)) 
            sector_points = [(t_cx, t_cy)]
            for i in range(steps + 1):
                a = start_angle + (end_angle - start_angle) * i / steps
                px = t_cx + r_cap_px * math.cos(a)
                py = t_cy + r_cap_px * math.sin(a)
                sector_points.append((int(px), int(py)))
            
            # Draw semi-transparent filled sector
            if len(sector_points) > 2:
                pygame.gfxdraw.filled_polygon(surface, sector_points, (0, 255, 255, 60))
                pygame.gfxdraw.aapolygon(surface, sector_points, (0, 255, 255, 120))

    # 智能体：Tracker为蓝，Target为红
    _draw_agent(surface, tracker, (0, 0, 255), role='tracker')
    _draw_agent(surface, target, (255, 0, 0), role='target')

    canvas = pygame.transform.smoothscale(surface, (w, h)) if ss > 1 else surface
    return pygame.surfarray.array3d(canvas).swapaxes(0, 1)

def agent_move(agent, action, moving_size, role=None):
    """简化的运动模型（velocity-based，移除冗余检查）"""
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

    rad_angle = math.radians(new_angle)
    new_x = agent['x'] + speed * math.cos(rad_angle)
    new_y = agent['y'] + speed * math.sin(rad_angle)
    
    # Boundary clipping
    agent['x'] = float(np.clip(new_x, 0, map_config.width - map_config.pixel_size))
    agent['y'] = float(np.clip(new_y, 0, map_config.height - map_config.pixel_size))

    # Update state for observations
    agent['w'] = float(angle_delta)
    agent['v'] = float(speed)

    # Collision check (revert position on collision)
    cx = agent['x'] + map_config.pixel_size * 0.5
    cy = agent['y'] + map_config.pixel_size * 0.5
    if is_point_blocked(cx, cy, padding=0.0):
        # Revert on collision
        agent['x'] -= speed * math.cos(rad_angle)
        agent['y'] -= speed * math.sin(rad_angle)
        agent['x'] = float(np.clip(agent['x'], 0, map_config.width - map_config.pixel_size))
        agent['y'] = float(np.clip(agent['y'], 0, map_config.height - map_config.pixel_size))
        
        # Velocity becomes 0 on collision
        agent['v'] = 0.0
    
    return agent

# agent_move_accel removed (deprecated acceleration logic)

def _rect_contains(px, py, rect, padding=0.0):
    return (rect['x'] - padding <= px <= rect['x'] + rect['w'] + padding and
            rect['y'] - padding <= py <= rect['y'] + rect['h'] + padding)

def _circle_contains(px, py, circ, padding=0.0):
    return math.hypot(px - circ['cx'], py - circ['cy']) <= float(circ['r']) + padding

def _segment_contains(px, py, seg, padding=0.0):
    thick = float(seg.get('thick', 8.0))
    return _dist2_point_to_segment(px, py, seg['x1'], seg['y1'], seg['x2'], seg['y2']) <= (0.5 * thick + padding) ** 2

def is_point_blocked(px, py, padding=0.0):
    """点是否与任何障碍碰撞；优先使用占据栅格。"""
    if _occ_available():
        ix, iy, _, _ = _world_to_cell(px, py, _OCC_CELL)
        pad_cells = int(math.ceil(float(padding) / float(_OCC_CELL)))
        return _occ_any_with_pad(ix, iy, pad_cells)

    for obs in getattr(map_config, 'obstacles', []):
        typ = obs.get('type')
        if typ == 'rect' and _rect_contains(px, py, obs, float(padding)):
            return True
        if typ == 'circle' and _circle_contains(px, py, obs, float(padding)):
            return True
        if typ == 'segment' and _segment_contains(px, py, obs, float(padding)):
            return True
    return False

def _resolve_obstacle_collision(old_pos, new_pos):
    """若 new_pos 的中心点落入障碍，则回退到 old_pos。"""
    cx = new_pos['x'] + map_config.pixel_size * 0.5
    cy = new_pos['y'] + map_config.pixel_size * 0.5
    if is_point_blocked(cx, cy, padding=0.0):
        new_pos['x'], new_pos['y'] = old_pos['x'], old_pos['y']
    return new_pos
