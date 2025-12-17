import datetime
from map_config import EnvParameters

class SetupParameters:
    """
    系统设置参数
    """
    SEED = 1234              # 随机种子
    NUM_GPU = 1              # 使用的GPU数量
    USE_GPU_LOCAL = False    # 是否在本地（Runner）使用GPU
    USE_GPU_GLOBAL = True    # 是否在全局（Driver/Learner）使用GPU
    PRETRAINED_TARGET_PATH = None  # 预训练Target模型路径
    PRETRAINED_TRACKER_PATH = None # 预训练Tracker模型路径

class TrainingParameters:
    """
    训练超参数
    """
    # --- 优化器设置 ---
    lr = 3e-4                # 初始学习率
    LR_FINAL = 3e-5          # 最终学习率
    LR_SCHEDULE = 'cosine'   # 学习率调度方式 ('cosine', 'linear', 'constant')
    
    # --- 训练流程设置 ---
    N_ENVS = 4               # 并行环境数量
    N_STEPS = 2048           # 每个环境采样的步数 (PPO Rollout Length)
    
    # 纯 RL 训练步数
    N_MAX_STEPS = int(8e6)   # 800万步
    
    LOG_EPOCH_STEPS = int(1e4) # 每隔多少步记录一次日志
    
    MINIBATCH_SIZE = 64      # PPO更新的Mini-batch大小
    N_EPOCHS_INITIAL = 12    # 初始Epoch数 (会衰减)
    N_EPOCHS_FINAL = 4       # 最终Epoch数

    # --- 序列长度设置 (MLP也使用TBPTT进行数据切分) ---
    TBPTT_STEPS = 32         # 截断反向传播的时间步长 (也是Context Window大小)
    
    # --- PPO 核心参数 ---
    VALUE_CLIP_RANGE = 0.2   # Value Loss的Clip范围
    CLIP_RANGE = 0.1         # Policy Loss的Clip范围 (PPO Clip)
    RATIO_CLAMP_MAX = 4.0    # Importance Sampling Ratio的最大值
    EX_VALUE_COEF = 0.6      # Value Loss的系数
    ENTROPY_COEF = 0.01      # Entropy Bonus的系数
    MAX_GRAD_NORM = 0.5      # 梯度裁剪阈值
    GAMMA = 0.99             # 折扣因子
    LAM = 0.95               # GAE参数 lambda
    
    # --- 观测噪声 (Domain Randomization) ---
    OBS_NOISE_STD = 0.01     # 观测噪声标准差 (0=无噪声, 建议0.01~0.05)
    
    # ====== 训练模式 ======
    # 'rl'     - 纯 RL 训练 (推荐配合 SAFETY_LAYER_ENABLED)
    # 'phase1' - IL-only Actor + RL Critic/Q
    # 'phase2' - Q-filtered IL + RL
    # 'mixed'  - 传统混合训练
    # 'il'     - 纯 IL 训练
    TRAINING_MODE = "rl"
    
    # ====== Safety Layer (环境辅助避障) ======
    # 开启后，环境会在执行动作前修正动作以避免碰撞
    # 让 RL 专注于追踪目标，避障由环境处理
    # 后续可用 Residual Learning 学习这个避障行为
    SAFETY_LAYER_ENABLED = True
    
    # IL 概率 (RL 模式下自动设为 0)
    IL_INITIAL_PROB = 0.0
    IL_FINAL_PROB = 0.0
    IL_PROB_DECAY_ENABLED = False
    IL_DECAY_STEPS = 1.0
    
    # Q-Critic 设置 (RL 模式不需要)
    USE_Q_CRITIC = False
    Q_LOSS_COEF = 0.5
    IL_FILTER_THRESHOLD = 0.0
    
    # --- 对手策略设置 ---
    OPPONENT_TYPE = "random"
    
    # 随机对手权重
    RANDOM_OPPONENT_WEIGHTS = {
        "target": {
            "CoverSeeker": 1.0,
            # "Greedy": 1.0,
            # "ZigZag": 1.0,
            # "Orbiter": 1.0,
        }
    }

class NetParameters:
    """
    网络结构参数
    """
    # ====== 雷达编码配置（统一开关）======
    RADAR_DIM = 64           # 原始雷达维度（射线数量）
    RADAR_EMBED_DIM = 8      # 雷达编码后维度（可调整：8/16/32）
    
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
    EXPERIMENT_NAME = "mlp_rl_coverseeker"
    ENTITY = "user"
    EXPERIMENT_NOTE = "MLP PPO training"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False          # 是否继续训练
    RESTORE_DIR = "./models/mlp_rl_greedy_12-12-14-27/best_model/checkpoint.pth"       # 恢复模型的目录
    
    WANDB = False            # 是否使用WandB
    TENSORBOARD = True       # 是否使用TensorBoard
    TXT_LOG = True           # 是否记录TXT日志
    
    # 路径设置
    SUMMARY_PATH = f'./models/{EXPERIMENT_NAME}{TIME}'
    MODEL_PATH = f'./models/{EXPERIMENT_NAME}{TIME}'
    GIFS_PATH = f'./models/{EXPERIMENT_NAME}{TIME}/gifs'
    
    # 频率设置
    EVAL_INTERVAL = 100000   # 评估间隔 (步数)
    SAVE_INTERVAL = 300000   # 保存模型间隔 (步数)
    BEST_INTERVAL = 0        # (未使用)
    GIF_INTERVAL = 300000    # 保存GIF间隔 (步数)
    EVAL_EPISODES = 50       # 评估时的对局数
    
    # Loss 名称列表 (用于日志记录)
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std', 
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean'
    ]