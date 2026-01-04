import datetime
from map_config import EnvParameters, ObstacleDensity

class SetupParameters:
    """
    系统设置参数
    """
    SEED = 1234              # 随机种子
    NUM_GPU = 1              # 使用的GPU数量
    USE_GPU_LOCAL = False    # 是否在本地（Runner）使用GPU
    USE_GPU_GLOBAL = False    # 是否在全局（Driver/Learner）使用GPU
    PRETRAINED_TARGET_PATH = None  # 预训练Target模型路径
    PRETRAINED_TRACKER_PATH = None # 预训练Tracker模型路径
    
    # 障碍物密度等级 (none, sparse, medium, dense)
    OBSTACLE_DENSITY = ObstacleDensity.DENSE

class TrainingParameters:
    """
    训练超参数
    """
    # --- 优化器设置 ---
    lr = 3e-4                # 初始学习率
    LR_FINAL = 1e-5          # 最终学习率
    LR_SCHEDULE = 'cosine'   # 学习率调度方式 ('cosine', 'linear', 'constant')
    
    # --- 训练流程设置 ---
    N_ENVS = 4               # 并行环境数量
    N_STEPS = 2048           # 每个环境采样的步数 (PPO Rollout Length)
    N_MAX_STEPS = 30e7        # 最大训练总步数
    LOG_EPOCH_STEPS = int(1e4) # 每隔多少步记录一次日志
    
    MINIBATCH_SIZE = 64      # PPO更新的Mini-batch大小
    N_EPOCHS_INITIAL = 10    # N_EPOCHS 初始值
    N_EPOCHS_FINAL = 10       # N_EPOCHS 最终值 (线性衰减)

    # --- 序列长度设置 (MLP也使用TBPTT进行数据切分) ---
    TBPTT_STEPS = 32         # 截断反向传播的时间步长 (也是Context Window大小)
    
    # --- PPO 核心参数 ---
    VALUE_CLIP_RANGE = 0.2   # Value Loss的Clip范围
    CLIP_RANGE = 0.2         # Policy Loss的Clip范围 (PPO Clip)
    RATIO_CLAMP_MAX = 4.0    # Importance Sampling Ratio的最大值
    EX_VALUE_COEF = 0.5      # Value Loss的系数
    ENTROPY_COEF = 0.01      # Entropy Bonus的系数
    MAX_GRAD_NORM = 0.5      # 梯度裁剪阈值
    GAMMA = 0.99             # 折扣因子
    LAM = 0.95               # GAE参数 lambda
    
    # --- 模仿学习 (IL) 设置 ---
    IL_TYPE = "expert"       # 模仿类型 (目前仅支持 'expert')
    
    # 训练模式: 'mixed' (IL+RL), 'rl' (Pure RL), 'il' (Pure IL)
    TRAINING_MODE = "rl"
    
    # IL 概率调度
    IL_INITIAL_PROB = 1.0    # 初始模仿概率
    IL_FINAL_PROB = 0.05     # 最终模仿概率
    IL_DECAY_STEPS = 3e6     # 衰减步数

    if TRAINING_MODE == "rl":
        IL_INITIAL_PROB = 0.0
        IL_FINAL_PROB = 0.0
        IL_DECAY_STEPS = 1.0
    elif TRAINING_MODE == "il":
        IL_INITIAL_PROB = 1.0
        IL_FINAL_PROB = 1.0
        IL_DECAY_STEPS = 1.0
    
    # IL 数据混合比例
    # 既然只用新数据，这里不再需要比例设置，或者可以理解为 1.0
    
    # Q-Filter 设置
    USE_Q_CRITIC = False      # 是否使用Q-Critic进行专家数据筛选
    Q_LOSS_COEF = 0.5        # Q Loss的系数
    IL_FILTER_THRESHOLD = -0 # Q-Filter 阈值 (Q_expert - V > threshold)
    
    # --- 对手策略设置 ---
    # 纯 RL 训练，随机抽样四个规则 target
    OPPONENT_TYPE = "random" # 对手类型 ('random', 'policy', 'random_nonexpert')
    
    # 自适应采样 (Adaptive Sampling)
    ADAPTIVE_SAMPLING = False           # 是否开启自适应采样
    ADAPTIVE_SAMPLING_WINDOW = 200      # 统计胜率的窗口大小
    ADAPTIVE_SAMPLING_MIN_GAMES = 10   # 最小对局数
    ADAPTIVE_SAMPLING_STRENGTH = 1.0   # 采样强度 (越大越倾向于选择弱势对手)
    
    # 随机对手权重 (初始权重)
    RANDOM_OPPONENT_WEIGHTS = {
        "target": {
            "Greedy": 1.0,
            # "CoverSeeker": 1.0,
            # "ZigZag": 0.5,
            # "Orbiter": 0.5,
        }
    }

class NetParameters:
    """
    网络结构参数
    """
    # Radar Encoding
    RADAR_DIM = 64           # 原始雷达维度
    RADAR_EMBED_DIM = 8      # 雷达编码后维度
    
    # Inputs
    ACTOR_SCALAR_LEN = 11    # Actor 标量部分
    PRIVILEGED_SCALAR_LEN = 8 # Target 标量部分
    
    # Input Vectors (Scalar + Embedded Radar)
    # RAW dimensions (for buffers and env interaction)
    ACTOR_RAW_LEN = ACTOR_SCALAR_LEN + RADAR_DIM           # 11 + 64 = 75
    PRIVILEGED_RAW_LEN = PRIVILEGED_SCALAR_LEN + RADAR_DIM # 8 + 64 = 72
    CRITIC_RAW_LEN = ACTOR_RAW_LEN + PRIVILEGED_RAW_LEN    # 75 + 72 = 147

    # ENCODED dimensions (for network internal processing)
    ACTOR_VECTOR_LEN = ACTOR_SCALAR_LEN + RADAR_EMBED_DIM      # 11 + 8 = 19
    PRIVILEGED_LEN = PRIVILEGED_SCALAR_LEN + RADAR_EMBED_DIM   # 8 + 8 = 16
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + PRIVILEGED_LEN      # 19 + 16 = 35
    
    ACTION_DIM = 2           # 动作维度 (Angle, Speed)
    
    # MLP 特有参数
    HIDDEN_DIM = 128         # 隐藏层维度
    NUM_HIDDEN_LAYERS = 5    # 隐藏层层数
    CONTEXT_WINDOW = TrainingParameters.TBPTT_STEPS # 上下文窗口长度 (用于数据处理)

class RecordingParameters:
    """
    日志与记录参数
    """
    EXPERIMENT_PROJECT = "AvoidMaker_MLP"
    
    # 环境安全层开关 (训练时)
    ENABLE_SAFETY_LAYER = False   # True: 环境辅助避障, False: 纯 RL 自主学习避障
    # 根据激活的 targets 自动命名
    _targets = list(TrainingParameters.RANDOM_OPPONENT_WEIGHTS.get("target", {}).keys())
    EXPERIMENT_NAME = f"rl_{_targets[0] if len(_targets) == 1 else 'all'}"
    EXPERIMENT_NAME += "_collision" if not ENABLE_SAFETY_LAYER else ""
    
    ENTITY = "user"
    EXPERIMENT_NOTE = "MLP PPO training with separate radar encoding"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False          # 是否继续训练 (加载权重和进度)
    FRESH_RETRAIN = False     # 仅加载模型权重，重置训练进度和学习率调度
    RESTORE_DIR = "./models/rl_all_12-23-12-19/latest_model/checkpoint.pth"       # 恢复模型的目录
    

    TENSORBOARD = True       # 是否使用TensorBoard
    TXT_LOG = True           # 是否记录TXT日志
    
    # 路径设置 (包含障碍物密度等级)
    _DENSITY_TAG = f'_{SetupParameters.OBSTACLE_DENSITY}'
    SUMMARY_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    MODEL_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    GIFS_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}/gifs'
    
    # 频率设置
    EVAL_INTERVAL = 100000   # 评估间隔 (步数)
    SAVE_INTERVAL = 300000   # 保存模型间隔 (步数)
    BEST_INTERVAL = 0        # (未使用)
    GIF_INTERVAL = 500000    # 保存GIF间隔 (步数)
    EVAL_EPISODES = 48       # 评估时的对局数
    
    # Loss 名称列表 (用于日志记录)
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std', 
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean'
    ]

