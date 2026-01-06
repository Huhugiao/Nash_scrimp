"""
Residual RL 训练参数配置 (独立)
条件式残差网络：基础模型冻结，残差网络在危险时激活进行避障修正
"""
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from map_config import ObstacleDensity


class SetupParameters:
    """系统设置参数"""
    SEED = 1234
    NUM_GPU = 1
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = False
    
    # 障碍物密度等级 (none, sparse, medium, dense)
    OBSTACLE_DENSITY = ObstacleDensity.DENSE


class TrainingParameters:
    """训练超参数"""
    # --- 优化器设置 ---
    lr = 1e-4                         # 初始学习率 (比MLP低)
    LR_FINAL = 1e-5                   # 最终学习率
    LR_SCHEDULE = 'cosine'            # 学习率调度方式
    
    # --- 训练流程设置 ---
    N_ENVS = 4                        # 并行环境数量
    N_STEPS = 2048                    # 每个环境采样的步数
    N_MAX_STEPS = 10e6                # 最大训练总步数
    LOG_EPOCH_STEPS = int(1e4)        # 日志间隔
    
    MINIBATCH_SIZE = 64               # Mini-batch大小
    N_EPOCHS_INITIAL = 10             # N_EPOCHS 初始值
    N_EPOCHS_FINAL = 10               # N_EPOCHS 最终值
    
    # --- PPO 核心参数 ---
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    RATIO_CLAMP_MAX = 4.0
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    GAMMA = 0.99
    LAM = 0.95
    
    # --- 对手策略 (与MLP一致) ---
    OPPONENT_TYPE = "random"
    ADAPTIVE_SAMPLING = False
    ADAPTIVE_SAMPLING_WINDOW = 200
    ADAPTIVE_SAMPLING_MIN_GAMES = 10
    ADAPTIVE_SAMPLING_STRENGTH = 1.0
    
    RANDOM_OPPONENT_WEIGHTS = {
        "target": {
            "Greedy": 1.0,
            # "CoverSeeker": 1.0,
            # "ZigZag": 0.5,
            # "Orbiter": 0.5,
        }
    }
    
    # --- 观测噪声 ---
    OBS_NOISE_STD = 0.0


class NetParameters:
    """网络结构参数"""
    # Radar Encoding (与 MLP 一致)
    RADAR_DIM = 64
    RADAR_EMBED_DIM = 8
    
    # Inputs (移除了 normalized_heading，全部第一视角观测)
    ACTOR_SCALAR_LEN = 10
    PRIVILEGED_SCALAR_LEN = 8
    
    # Input Vectors (RAW dimensions)
    ACTOR_RAW_LEN = ACTOR_SCALAR_LEN + RADAR_DIM            # 74
    PRIVILEGED_RAW_LEN = PRIVILEGED_SCALAR_LEN + RADAR_DIM  # 72
    CRITIC_RAW_LEN = ACTOR_RAW_LEN + PRIVILEGED_RAW_LEN     # 146
    
    ACTION_DIM = 2
    CONTEXT_WINDOW = 32
    
    # --- 残差网络特有参数 ---
    VELOCITY_DIM = 2  # 线速度 + 角速度
    RESIDUAL_INPUT_DIM = RADAR_DIM + ACTION_DIM + VELOCITY_DIM  # 64 + 2 + 2 = 68
    RESIDUAL_HIDDEN_DIM = 64          # 隐藏层维度 (比MLP小)
    RESIDUAL_NUM_LAYERS = 2           # 隐藏层数
    RESIDUAL_MAX_SCALE = 0.5          # Residual 最大幅度 [-0.5, 0.5]
    
    # Log std bounds for residual actor (stability)
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0


class ResidualRLConfig:
    """残差RL主配置"""
    ENABLED = True
    EXPERIMENT_NAME = "residual_avoidance"
    
    # 基础模型路径 (冻结的预训练 Tracker)
    BASE_MODEL_PATH = "./models/rl_Greedy_collision_medium_01-05-09-21/best_model/checkpoint.pth"
    
    # 惩罚系数
    ACTION_PENALTY_COEF = 0.002       # L2 惩罚 (鼓励小 residual)


class RecordingParameters:
    """日志与记录参数"""
    EXPERIMENT_PROJECT = "AvoidMaker_Residual"
    EXPERIMENT_NAME = "residual_avoidance"
    
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    # 路径设置 (包含障碍物密度等级)
    _DENSITY_TAG = f'_{SetupParameters.OBSTACLE_DENSITY}'
    SUMMARY_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    MODEL_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    GIFS_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}/gifs'
    
    # 频率设置
    SAVE_INTERVAL = 300000
    EVAL_INTERVAL = 100000
    GIF_INTERVAL = 500000
    EVAL_EPISODES = 32
    
    TENSORBOARD = True
    
    # Loss 名称
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std',
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean'
    ]
