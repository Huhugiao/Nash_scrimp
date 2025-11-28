import math
import numpy as np
import map_config

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    pygame = None

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

def ray_distance_grid(origin, angle_rad, max_range, padding=0.0):
    """
    在占据栅格上沿方向 angle_rad 执行 DDA 射线。
    如果 Numba 可用，使用加速内核。
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
    if NUMBA_AVAILABLE and _occ_available():
        ox, oy = float(origin[0]), float(origin[1])
        nx, ny = _OCC_GRID.shape[1], _OCC_GRID.shape[0]
        pad_cells = int(math.ceil(float(padding) / float(_OCC_CELL)))
        # Ensure angles is numpy array
        angles_arr = np.asarray(angles_rad, dtype=np.float64)
        return _numba_ray_cast_batch(ox, oy, angles_arr, float(max_range), 
                                   _OCC_GRID, float(_OCC_CELL), nx, ny, pad_cells)
    
    return np.array([ray_distance_grid(origin, ang, max_range, padding) for ang in angles_rad],
                    dtype=np.float32)

def reward_calculate(tracker, target, tracker_collision=False, target_collision=False,
                     sector_captured=False, capture_progress=0, capture_required_steps=0):
    info = {
        'capture_progress': int(capture_progress),
        'capture_required_steps': int(capture_required_steps),
        'tracker_collision': bool(tracker_collision),
        'target_collision': bool(target_collision)
    }

    # --- Dense shaping 部分（每步）---
    reward = 0.0
    terminated = False

    # 1) 距离 shaping：越近越好
    dx = (tracker['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    dy = (tracker['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    dist = math.hypot(dx, dy)

    max_ref_dist = float(getattr(map_config.EnvParameters, 'FOV_RANGE', max(map_config.width, map_config.height)))
    d_norm = max(0.0, min(dist / max_ref_dist, 1.0))  # [0,1]
    w_dist = 0.02  # 距离奖励权重（可调）
    reward += w_dist * (1.0 - d_norm)

    # # 2) 捕获进度 shaping：在捕获扇区内时稍微加分
    # if capture_required_steps > 0 and capture_progress > 0:
    #     frac = float(capture_progress) / float(capture_required_steps)
    #     frac = max(0.0, min(frac, 1.0))
    #     w_cap = 0.05  # 捕获进度奖励权重（可调）
    #     reward += w_cap * frac

    # 3) 时间惩罚：鼓励更快结束
    time_penalty = 0.001  # 每步小惩罚（可调）
    reward -= time_penalty

    # --- 终局奖励部分（与原逻辑保持一致，只是参数化）---
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

def _draw_agent(surface, agent, color):
    """
    绘制美化后的智能体：
    - 科技感机甲风格
    - 核心发光
    - 尾部推进器
    - 动态光环
    """
    if pygame is None:
        return
    
    # 获取参数
    ss = getattr(map_config, 'ssaa', 1)
    x_world = agent['x'] + map_config.pixel_size * 0.5
    y_world = agent['y'] + map_config.pixel_size * 0.5
    cx, cy = int(x_world * ss), int(y_world * ss)
    
    # 基础半径
    r = map_config.agent_radius * ss
    heading = math.radians(agent.get('theta', 0.0))
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    
    # 颜色解包
    base_rgb = color[:3]
    r_val, g_val, b_val = base_rgb
    
    # --- 1. 能量护盾 (Collision Radius) ---
    # 使用多层透明圆模拟光晕
    shield_r = int(r * 1.4)
    # 外层淡光
    try:
        pygame.gfxdraw.filled_circle(surface, cx, cy, shield_r, (*base_rgb, 20))
        pygame.gfxdraw.aacircle(surface, cx, cy, shield_r, (*base_rgb, 40))
    except AttributeError:
        pass
    
    # --- 2. 机体 (Body) ---
    # 设计一个类似 "星际战机" 的形状
    # 顶点定义 (相对坐标, 假设朝右)
    scale = 1.1
    # 形状：[鼻锥, 右翼前, 右翼后, 尾部凹陷, 左翼后, 左翼前]
    pts_rel = [
        (r * 1.3, 0),           # Nose
        (r * 0.2, r * 0.7),     # Right Wing Front
        (-r * 0.6, r * 0.9),    # Right Wing Back
        (-r * 0.3, 0),          # Rear Center (Engine)
        (-r * 0.6, -r * 0.9),   # Left Wing Back
        (r * 0.2, -r * 0.7)     # Left Wing Front
    ]
    
    # 旋转并转换坐标
    poly_pts = []
    for dx, dy in pts_rel:
        dx *= scale
        dy *= scale
        px = cx + dx * cos_h - dy * sin_h
        py = cy + dx * sin_h + dy * cos_h
        poly_pts.append((px, py))
        
    # 填充机体 (深色底 + 亮色边)
    darker_rgb = (max(0, r_val-40), max(0, g_val-40), max(0, b_val-40))
    try:
        pygame.gfxdraw.filled_polygon(surface, poly_pts, (*darker_rgb, 230))
        pygame.gfxdraw.aapolygon(surface, poly_pts, (*base_rgb, 255))
    except AttributeError:
        pygame.draw.polygon(surface, darker_rgb, poly_pts)
        pygame.draw.polygon(surface, base_rgb, poly_pts, 1)
    
    # --- 3. 核心反应堆 (Core) ---
    core_offset = 0
    core_x = cx + core_offset * cos_h
    core_y = cy + core_offset * sin_h
    core_radius = int(r * 0.3)
    
    # 核心发光 (白色)
    try:
        pygame.gfxdraw.filled_circle(surface, int(core_x), int(core_y), core_radius, (255, 255, 255, 255))
        # 核心光晕
        pygame.gfxdraw.filled_circle(surface, int(core_x), int(core_y), int(core_radius*1.8), (*base_rgb, 100))
    except AttributeError:
        pygame.draw.circle(surface, (255, 255, 255), (int(core_x), int(core_y)), core_radius)

    # --- 4. 尾焰 (Thrusters) ---
    # 在两个翼尖后方画小的粒子流
    thruster_offset = -r * 0.6 * scale
    wing_spread = r * 0.6 * scale
    
    # 计算两个引擎位置
    eng1_x = cx + thruster_offset * cos_h - wing_spread * sin_h
    eng1_y = cy + thruster_offset * sin_h + wing_spread * cos_h
    
    eng2_x = cx + thruster_offset * cos_h - (-wing_spread) * sin_h
    eng2_y = cy + thruster_offset * sin_h + (-wing_spread) * cos_h
    
    # 尾焰颜色 (偏黄/白)
    flame_color = (255, 230, 150, 180)
    flame_r = int(r * 0.25)
    
    try:
        pygame.gfxdraw.filled_circle(surface, int(eng1_x), int(eng1_y), flame_r, flame_color)
        pygame.gfxdraw.filled_circle(surface, int(eng2_x), int(eng2_y), flame_r, flame_color)
    except AttributeError:
        pass

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
    """点是否与任何障碍碰撞；优先使用栅格。"""
    if _occ_available():
        ix, iy, _, _ = _world_to_cell(px, py, _OCC_CELL)
        pad_cells = int(math.ceil(float(padding) / float(_OCC_CELL)))
        return _occ_any_with_pad(ix, iy, pad_cells)

    for obs in getattr(map_config, 'obstacles', []):
        if obs['type'] == 'rect' and _rect_contains(px, py, obs, padding):
            return True
        if obs['type'] == 'circle' and _circle_contains(px, py, obs, padding):
            return True
        if obs.get('type') == 'segment' and _segment_contains(px, py, obs, padding):
            return True
    return False

def _resolve_obstacle_collision(old_pos, new_pos):
    if is_point_blocked(new_pos['x'] + map_config.pixel_size * 0.5,
                        new_pos['y'] + map_config.pixel_size * 0.5):
        new_pos['x'], new_pos['y'] = old_pos['x'], old_pos['y']
    return new_pos

def _draw_obstacles(surface):
    """绘制矩形边界 + 线段墙"""
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    for obs in getattr(map_config, 'obstacles', []):
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

def _draw_fov(surface, tracker, fov_points=None):
    """基于预计算的 fov_points 绘制半透明扇形。"""
    if pygame is None or not fov_points or len(fov_points) < 3:
        return
    try:
        # 1. 绘制填充 (Fill)
        # 使用稍淡一点的颜色，减少视觉干扰
        fill_color = (80, 140, 255, 30) 
        pygame.gfxdraw.filled_polygon(surface, fov_points, fill_color)
        
        # 2. 绘制轮廓 (Outline) - 仅绘制两侧射线
        # 修复波浪状边缘问题：不再绘制连接障碍物点的远端轮廓线，
        # 而是只绘制 FOV 锥体的两条侧边，保持视觉整洁。
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
        
    except Exception:
        pass

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
        try:
            # 1. 填充
            fill_color = getattr(map_config, 'CAPTURE_SECTOR_COLOR', (80, 200, 120, 40))
            pygame.gfxdraw.filled_polygon(surface, pts, fill_color)
            
            # 2. 轮廓 - 增加清晰度
            outline_color = (80, 200, 120, 200)
            pygame.gfxdraw.aapolygon(surface, pts, outline_color)
            pygame.draw.lines(surface, outline_color, True, pts, 1)
        except Exception:
            pass

def get_canvas(target, tracker, tracker_traj, target_traj, surface=None, fov_points=None):
    w, h = map_config.width, map_config.height
    ss = getattr(map_config, 'ssaa', 1)
    if pygame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    if surface is None:
        surface = pygame.Surface((w * ss, h * ss), flags=pygame.SRCALPHA)
    surface.fill(map_config.background_color)

    _draw_grid(surface)
    _draw_obstacles(surface)
    _draw_fov(surface, tracker, fov_points)
    _draw_capture_sector(surface, tracker)
    _draw_trail(surface, tracker_traj, map_config.trail_color_tracker, map_config.trail_width)
    _draw_trail(surface, target_traj, map_config.trail_color_target, map_config.trail_width)
    _draw_agent(surface, tracker, map_config.tracker_color)
    _draw_agent(surface, target, map_config.target_color)

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