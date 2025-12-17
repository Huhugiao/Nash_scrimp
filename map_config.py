import os
import random, math

# ============================================================================
# Environment Parameters
# ============================================================================

class EnvParameters:
    RENDER_FPS = 30  # Increased from 24 for smoother visualization
    EPISODE_LEN = 449  # Rounded for cleaner logic
    NUM_TARGET_POLICIES = 4

    # FOV configuration (optimized values)
    FOV_ANGLE = 90  # 90° front facing cone
    FOV_RANGE = 250.0  # Maximum visibility range
    RADAR_RAYS = 64  # Power of 2 for efficiency
    MAX_UNOBSERVED_STEPS = 90  # ~3 seconds at 30 FPS

# ============================================================================
# Obstacle Density Levels
# ============================================================================

class ObstacleDensity:
    """障碍物密度级别（线段墙为主）"""
    NONE = "none"
    SPARSE = "sparse"
    MEDIUM = "medium"
    DENSE = "dense"

    ALL_LEVELS = [NONE, SPARSE, MEDIUM, DENSE]

DEFAULT_OBSTACLE_DENSITY = ObstacleDensity.DENSE
current_obstacle_density = DEFAULT_OBSTACLE_DENSITY

_jitter_px = 0
_jitter_seed = 0

# ============================================================================
# Map Configuration
# ============================================================================

# 画布尺寸（正方形，便于栅格）
width = 640
height = 640

# 基础单位与速度
pixel_size = 4
# target 略慢，tracker 稍快
target_speed = 2.0
tracker_speed = 2.6  # pixels/step

# Acceleration limits (Removed - System is now Velocity-based)
# tracker_max_acc, target_max_acc, etc. are deprecated.

# 物理极限 (Max Speed/Angular Speed)
# tracker_speed defined above (2.4)
tracker_max_angular_speed = 5 # degrees/step
target_max_angular_speed = 10.0 # degrees/step

# 捕获逻辑（保持接口，但参数更“紧凑”）
capture_radius = 25.0
capture_sector_angle_deg = 40.0
capture_required_steps = 1
CAPTURE_SECTOR_COLOR = (90, 220, 140, 50)

# 渲染质量开关
FAST = os.getenv('SCRIMP_RENDER_MODE', 'fast').lower() == 'fast'

# GIF 输出配置
# 提升至 640 (原生分辨率) 以获得最高清晰度，虽然文件稍大但视觉效果最好
gif_max_side = 640

# 视觉主题：浅背景 + 高对比色
# 改为纯白背景以优化GIF压缩率
background_color = (255, 255, 255)
grid_color = (245, 245, 245)
grid_step = 64

# 轨迹与智能体外观
trail_color_tracker = (80, 130, 255, 190)
trail_color_target  = (255, 120, 120, 190)
# 缩短轨迹长度以减少GIF每帧的像素变化量，显著降低文件大小
trail_max_len = 100  # Reduced for memory
trail_width = 2

# 增大墙体厚度：视觉更清晰，也让边界遮挡更合理
wall_thickness = 8  # Increased for better visibility

# 障碍物颜色（深灰略带透明）
OBSTACLE_COLOR = (70, 70, 80, 255)

# 边界墙（稍微加粗）
WALL_OBSTACLES = [
    {'type': 'rect', 'x': 0, 'y': 0, 'w': width, 'h': wall_thickness, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 0, 'y': height - wall_thickness, 'w': width, 'h': wall_thickness, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 0, 'y': 0, 'w': wall_thickness, 'h': height, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': width - wall_thickness, 'y': 0, 'w': wall_thickness, 'h': height, 'color': OBSTACLE_COLOR},
]

# ============================================================================
# 线段为主的障碍布局
# 设计成“城市场景”：几条主干道 + 巷战瓶颈，保持对称，但避免完全规则。
# ============================================================================

# ============================================================================
# 均匀分布的障碍物设计 (Grid-based Uniform Distribution)
# Map Size: 640x640
# Expanded to reduce empty space at edges.
# ============================================================================

_SPARSE_OBSTACLES = [
    # 3x3 Grid
    # Centers: 80, 320, 560 (Spacing 240)
    # Margins: ~60px
    
    # Row 1 (y=80)
    {'type': 'rect', 'x': 60, 'y': 60, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 320, 'cy': 80, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 540, 'y': 60, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    
    # Row 2 (y=320)
    {'type': 'circle', 'cx': 80, 'cy': 320, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 300, 'y': 300, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR}, # Center
    {'type': 'circle', 'cx': 560, 'cy': 320, 'r': 20, 'color': OBSTACLE_COLOR},

    # Row 3 (y=560)
    {'type': 'rect', 'x': 60, 'y': 540, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 320, 'cy': 560, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 540, 'y': 540, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
]

_MEDIUM_OBSTACLES = [
    # 4x4 Grid
    # Centers: 80, 240, 400, 560 (Spacing 160)
    # Margins: ~60px

    # Row 1 (y=80)
    {'type': 'rect', 'x': 60, 'y': 60, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 240, 'y1': 50, 'x2': 240, 'y2': 110, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 400, 'y1': 50, 'x2': 400, 'y2': 110, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 540, 'y': 60, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},

    # Row 2 (y=240)
    {'type': 'segment', 'x1': 50, 'y1': 240, 'x2': 110, 'y2': 240, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 240, 'cy': 240, 'r': 25, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 400, 'cy': 240, 'r': 25, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 530, 'y1': 240, 'x2': 590, 'y2': 240, 'thick': 10, 'color': OBSTACLE_COLOR},

    # Row 3 (y=400)
    {'type': 'segment', 'x1': 50, 'y1': 400, 'x2': 110, 'y2': 400, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 240, 'cy': 400, 'r': 25, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 400, 'cy': 400, 'r': 25, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 530, 'y1': 400, 'x2': 590, 'y2': 400, 'thick': 10, 'color': OBSTACLE_COLOR},

    # Row 4 (y=560)
    {'type': 'rect', 'x': 60, 'y': 540, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 240, 'y1': 530, 'x2': 240, 'y2': 590, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 400, 'y1': 530, 'x2': 400, 'y2': 590, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 540, 'y': 540, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
]

_DENSE_OBSTACLES = [
    # 5x5 Grid
    # Centers: 70, 195, 320, 445, 570 (Spacing 125)
    # Margins: ~50px

    # Row 1 (y=70)
    {'type': 'rect', 'x': 50, 'y': 50, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 195, 'y1': 40, 'x2': 195, 'y2': 100, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 300, 'y': 50, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 445, 'y1': 40, 'x2': 445, 'y2': 100, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 550, 'y': 50, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},

    # Row 2 (y=195)
    {'type': 'segment', 'x1': 40, 'y1': 195, 'x2': 100, 'y2': 195, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 195, 'cy': 195, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 320, 'y1': 165, 'x2': 320, 'y2': 225, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 445, 'cy': 195, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 540, 'y1': 195, 'x2': 600, 'y2': 195, 'thick': 10, 'color': OBSTACLE_COLOR},

    # Row 3 (y=320)
    {'type': 'rect', 'x': 50, 'y': 300, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 165, 'y1': 320, 'x2': 225, 'y2': 320, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 300, 'y': 300, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR}, # Center
    {'type': 'segment', 'x1': 415, 'y1': 320, 'x2': 475, 'y2': 320, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 550, 'y': 300, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},

    # Row 4 (y=445)
    {'type': 'segment', 'x1': 40, 'y1': 445, 'x2': 100, 'y2': 445, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 195, 'cy': 445, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 320, 'y1': 415, 'x2': 320, 'y2': 475, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 445, 'cy': 445, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 540, 'y1': 445, 'x2': 600, 'y2': 445, 'thick': 10, 'color': OBSTACLE_COLOR},

    # Row 5 (y=570)
    {'type': 'rect', 'x': 50, 'y': 550, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 195, 'y1': 540, 'x2': 195, 'y2': 600, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 300, 'y': 550, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 445, 'y1': 540, 'x2': 445, 'y2': 600, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 550, 'y': 550, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
]

OBSTACLE_PRESETS = {
    ObstacleDensity.NONE: [],
    ObstacleDensity.SPARSE: _SPARSE_OBSTACLES,
    ObstacleDensity.MEDIUM: _MEDIUM_OBSTACLES,
    ObstacleDensity.DENSE: _DENSE_OBSTACLES,
}

# 初始化障碍物
obstacles = list(WALL_OBSTACLES)
obstacles.extend(OBSTACLE_PRESETS.get(DEFAULT_OBSTACLE_DENSITY, []))

# 占据栅格单元尺寸（比像素大，降低算力）
occ_cell = 4.0

def regenerate_obstacles(count=None, seed=None, density_level=None):
    """根据当前密度级别重构障碍列表（无随机采样）。"""
    global current_obstacle_density, obstacles
    if density_level is not None:
        current_obstacle_density = density_level
    obstacles[:] = list(WALL_OBSTACLES)
    obstacles.extend(OBSTACLE_PRESETS.get(current_obstacle_density, []))
    return obstacles

def set_obstacle_density(density_level):
    """设置障碍物密度级别，并重生成障碍列表。"""
    global current_obstacle_density
    if density_level not in ObstacleDensity.ALL_LEVELS:
        raise ValueError(f"Invalid density level: {density_level}. Must be one of {ObstacleDensity.ALL_LEVELS}")
    current_obstacle_density = density_level
    regenerate_obstacles(density_level=density_level)

def get_obstacle_density():
    return current_obstacle_density

def set_obstacle_jitter(jitter_px=0, seed=0):
    # 保留接口，不使用
    global _jitter_px, _jitter_seed
    _jitter_px = int(max(0, jitter_px))
    _jitter_seed = int(seed)

# 智能体颜色与尺寸（再调一版，更高对比）
base_color_inner = (46, 160, 67)
base_color_outer = (22, 122, 39)

# Tracker 用偏紫蓝，Target 用偏橙红
tracker_color = (90, 110, 255)    # 亮蓝偏紫
target_color = (255, 120, 80)     # 橙红

agent_radius = 8          # 半径稍大一点，让造型更明显
base_radius_draw = 12

# 抗锯齿与超级采样
ssaa = 1  # Disable SSAA by default for speed
enable_aa = False
draw_grid = False

# 训练相关（保持接口）
test_flag = False
mask_flag = False
success_reward = 20
max_loss_step = 50
total_steps = 500


agent_spawn_min_gap = 150.0
dynamic_obstacles = False

def update_dynamic_obstacles():
    pass


def set_render_quality(mode: str):
    """运行时切换渲染质量: 'fast' 或 'quality'"""
    global FAST, ssaa, enable_aa, draw_grid, trail_max_len, trail_width
    FAST = (mode.lower() == 'fast')
    if FAST:
        ssaa = 1
        enable_aa = False
        draw_grid = False
        trail_max_len = 80
        trail_width = 1
    else:
        ssaa = 2
        enable_aa = True
        draw_grid = False
        trail_max_len = 400
        trail_width = 2

def set_speeds(tracker: float = None, target: float = None):
    global tracker_speed, target_speed
    if tracker is not None:
        tracker_speed = float(tracker)
    if target is not None:
        target_speed = float(target)

def set_capture_params(radius: float = None, sector_angle_deg: float = None, required_steps: int = None):
    global capture_radius, capture_sector_angle_deg, capture_required_steps
    if radius is not None:
        capture_radius = float(radius)
    if sector_angle_deg is not None:
        capture_sector_angle_deg = float(sector_angle_deg)
    if required_steps is not None:
        capture_required_steps = int(required_steps)