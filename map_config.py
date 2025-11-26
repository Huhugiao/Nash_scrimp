import os
import random, math

# ============================================================================
# Environment Parameters
# ============================================================================

class EnvParameters:
    # 观测 / 动作接口保持不变
    N_ACTIONS = 48
    EPISODE_LEN = 650
    NUM_TARGET_POLICIES = 5  # Updated to match CONTEXT_LEN (Greedy, APF, DWA, Hiding, Random)

    # 视场与观测配置（重新设计）
    # Tracker：有限视场、雷达 360°；Target：全局视角 + 360° 雷达
    FOV_ANGLE = 120         # 追踪者视场角（度）
    FOV_RANGE = 250         # 追踪者最大可见距离（像素）
    RADAR_RAYS = 16            # 360° 雷达射线数
    MAX_UNOBSERVED_STEPS = 80  # 最长未观测时间归一化上限

# ============================================================================
# Obstacle Density Levels
# ============================================================================

class ObstacleDensity:
    """障碍物密度级别（线段墙为主）"""
    NONE = "none"
    SPARSE = "sparse"
    MEDIUM = "medium"
    DENSE = "dense"
    ULTRA = "ultra"

    ALL_LEVELS = [NONE, SPARSE, MEDIUM, DENSE, ULTRA]

DEFAULT_OBSTACLE_DENSITY = ObstacleDensity.MEDIUM
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
tracker_speed = 3

# 转向限制（度/step）
max_turn_deg = 10.0
target_max_turn_deg = 12.0
tracker_max_turn_deg = 8

# 捕获逻辑（保持接口，但参数更“紧凑”）
capture_radius = 42.0
capture_sector_angle_deg = 90.0
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
trail_max_len = 120
trail_width = 2

# 增大墙体厚度：视觉更清晰，也让边界遮挡更合理
wall_thickness = 6  

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

# 手动设定的均匀分布障碍物列表

_SPARSE_OBSTACLES = [
    # 左上
    {'type': 'rect', 'x': 100, 'y': 100, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    # 右上
    {'type': 'segment', 'x1': 480, 'y1': 120, 'x2': 540, 'y2': 180, 'thick': 8, 'color': OBSTACLE_COLOR},
    # 左下
    {'type': 'segment', 'x1': 120, 'y1': 480, 'x2': 180, 'y2': 540, 'thick': 8, 'color': OBSTACLE_COLOR},
    # 右下
    {'type': 'rect', 'x': 500, 'y': 500, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    # 中间偏离一点
    {'type': 'circle', 'cx': 320, 'cy': 240, 'r': 20, 'color': OBSTACLE_COLOR},
]

_MEDIUM_OBSTACLES = [
    # 第一行
    {'type': 'rect', 'x': 80, 'y': 80, 'w': 50, 'h': 50, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 280, 'y1': 60, 'x2': 360, 'y2': 60, 'thick': 8, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 510, 'y': 80, 'w': 50, 'h': 50, 'color': OBSTACLE_COLOR},
    
    # 第二行
    {'type': 'segment', 'x1': 60, 'y1': 240, 'x2': 140, 'y2': 320, 'thick': 8, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 320, 'cy': 320, 'r': 25, 'color': OBSTACLE_COLOR}, # 中心圆
    {'type': 'segment', 'x1': 500, 'y1': 320, 'x2': 580, 'y2': 240, 'thick': 8, 'color': OBSTACLE_COLOR},

    # 第三行
    {'type': 'rect', 'x': 80, 'y': 510, 'w': 50, 'h': 50, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 280, 'y1': 580, 'x2': 360, 'y2': 580, 'thick': 8, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 510, 'y': 510, 'w': 50, 'h': 50, 'color': OBSTACLE_COLOR},
    
    # 补充一些中间的
    {'type': 'segment', 'x1': 200, 'y1': 200, 'x2': 200, 'y2': 440, 'thick': 8, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 440, 'y1': 200, 'x2': 440, 'y2': 440, 'thick': 8, 'color': OBSTACLE_COLOR},
]

_DENSE_OBSTACLES = [
    # 4x4 网格布局基础，去掉中心
    # Row 1
    {'type': 'rect', 'x': 60, 'y': 60, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 200, 'y1': 50, 'x2': 200, 'y2': 110, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 440, 'y1': 50, 'x2': 440, 'y2': 110, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 540, 'y': 60, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},

    # Row 2
    {'type': 'segment', 'x1': 50, 'y1': 200, 'x2': 110, 'y2': 200, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 220, 'cy': 220, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 420, 'cy': 220, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 530, 'y1': 200, 'x2': 590, 'y2': 200, 'thick': 10, 'color': OBSTACLE_COLOR},

    # Row 3
    {'type': 'segment', 'x1': 50, 'y1': 440, 'x2': 110, 'y2': 440, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 220, 'cy': 420, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 420, 'cy': 420, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 530, 'y1': 440, 'x2': 590, 'y2': 440, 'thick': 10, 'color': OBSTACLE_COLOR},

    # Row 4
    {'type': 'rect', 'x': 60, 'y': 540, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 200, 'y1': 530, 'x2': 200, 'y2': 590, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 440, 'y1': 530, 'x2': 440, 'y2': 590, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 540, 'y': 540, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    
    # Center Cross
    {'type': 'segment', 'x1': 280, 'y1': 320, 'x2': 360, 'y2': 320, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 320, 'y1': 280, 'x2': 320, 'y2': 360, 'thick': 10, 'color': OBSTACLE_COLOR},
]

_ULTRA_OBSTACLES = [
    # 密集矩阵分布
    # 外部一圈
    {'type': 'rect', 'x': 50, 'y': 50, 'w': 60, 'h': 60, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 290, 'y': 40, 'w': 60, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 530, 'y': 50, 'w': 60, 'h': 60, 'color': OBSTACLE_COLOR},
    
    {'type': 'rect', 'x': 40, 'y': 290, 'w': 40, 'h': 60, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 560, 'y': 290, 'w': 40, 'h': 60, 'color': OBSTACLE_COLOR},
    
    {'type': 'rect', 'x': 50, 'y': 530, 'w': 60, 'h': 60, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 290, 'y': 560, 'w': 60, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 530, 'y': 530, 'w': 60, 'h': 60, 'color': OBSTACLE_COLOR},

    # 内部斜线阵列
    {'type': 'segment', 'x1': 160, 'y1': 160, 'x2': 220, 'y2': 220, 'thick': 12, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 420, 'y1': 160, 'x2': 480, 'y2': 220, 'thick': 12, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 160, 'y1': 480, 'x2': 220, 'y2': 420, 'thick': 12, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 420, 'y1': 480, 'x2': 480, 'y2': 420, 'thick': 12, 'color': OBSTACLE_COLOR},

    # 中心复杂结构
    {'type': 'circle', 'cx': 320, 'cy': 320, 'r': 30, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 260, 'y1': 260, 'x2': 260, 'y2': 380, 'thick': 12, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 380, 'y1': 260, 'x2': 380, 'y2': 380, 'thick': 12, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 260, 'y1': 260, 'x2': 380, 'y2': 260, 'thick': 12, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 260, 'y1': 380, 'x2': 380, 'y2': 380, 'thick': 12, 'color': OBSTACLE_COLOR},
]

OBSTACLE_PRESETS = {
    ObstacleDensity.NONE: [],
    ObstacleDensity.SPARSE: _SPARSE_OBSTACLES,
    ObstacleDensity.MEDIUM: _MEDIUM_OBSTACLES,
    ObstacleDensity.DENSE: _DENSE_OBSTACLES,
    ObstacleDensity.ULTRA: _ULTRA_OBSTACLES,
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
ssaa = 1
enable_aa = True
draw_grid = False

# 训练相关（保持接口）
test_flag = False
mask_flag = False
success_reward = 20
max_loss_step = 50
total_steps = 500

agent_spawn_min_gap = 150.0

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

def set_turn_limits(tracker_deg: float = None, target_deg: float = None, default_deg: float = None):
    global tracker_max_turn_deg, target_max_turn_deg, max_turn_deg
    if tracker_deg is not None:
        tracker_max_turn_deg = float(tracker_deg)
    if target_deg is not None:
        target_max_turn_deg = float(target_deg)
    if default_deg is not None:
        max_turn_deg = float(default_deg)

def set_capture_params(radius: float = None, sector_angle_deg: float = None, required_steps: int = None):
    global capture_radius, capture_sector_angle_deg, capture_required_steps
    if radius is not None:
        capture_radius = float(radius)
    if sector_angle_deg is not None:
        capture_sector_angle_deg = float(sector_angle_deg)
    if required_steps is not None:
        capture_required_steps = int(required_steps)