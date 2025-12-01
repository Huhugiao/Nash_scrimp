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
    N_MAX_STEPS = 3e7        # 最大训练总步数
    LOG_EPOCH_STEPS = int(1e4) # 每隔多少步记录一次日志
    
    MINIBATCH_SIZE = 64      # PPO更新的Mini-batch大小
    N_EPOCHS = 10            # PPO更新的Epoch数

    # --- 序列长度设置 (MLP也使用TBPTT进行数据切分) ---
    TBPTT_STEPS = 32         # 截断反向传播的时间步长 (也是Context Window大小)
    
    # --- PPO 核心参数 ---
    VALUE_CLIP_RANGE = 0.2   # Value Loss的Clip范围
    CLIP_RANGE = 0.2         # Policy Loss的Clip范围 (PPO Clip)
    RATIO_CLAMP_MAX = 4.0    # Importance Sampling Ratio的最大值
    EX_VALUE_COEF = 0.5      # Value Loss的系数
    ENTROPY_COEF = 0.02      # Entropy Bonus的系数
    MAX_GRAD_NORM = 0.5      # 梯度裁剪阈值
    GAMMA = 0.99             # 折扣因子
    LAM = 0.95               # GAE参数 lambda
    
    # --- 模仿学习 (IL) 设置 ---
    IL_TYPE = "expert"       # 模仿类型 (目前仅支持 'expert')
    
    # 纯RL开关: 1 = 开启纯RL (关闭IL), 0 = 开启混合训练 (IL + RL)
    PURE_RL_SWITCH = 0   
    
    # IL 概率调度 (仅当 PURE_RL_SWITCH = 0 时生效)
    # 如果 PURE_RL_SWITCH = 1，这些值会被自动覆盖为 0
    IL_INITIAL_PROB = 1.0    # 初始模仿概率
    IL_FINAL_PROB = 0.05     # 最终模仿概率
    IL_DECAY_STEPS = 3e6     # 衰减步数

    if PURE_RL_SWITCH:
        IL_INITIAL_PROB = 0.0
        IL_FINAL_PROB = 0.0
        IL_DECAY_STEPS = 1.0
    
    # IL 数据混合比例
    # 既然只用新数据，这里不再需要比例设置，或者可以理解为 1.0
    
    # Q-Filter 设置
    USE_Q_CRITIC = True      # 是否使用Q-Critic进行专家数据筛选
    Q_LOSS_COEF = 0.5        # Q Loss的系数
    IL_FILTER_THRESHOLD = -0 # Q-Filter 阈值 (Q_expert - V > threshold)
    
    # --- 对手策略设置 ---
    OPPONENT_TYPE = "random" # 对手类型 ('random', 'policy', 'random_nonexpert')
    
    # 自适应采样 (Adaptive Sampling)
    ADAPTIVE_SAMPLING = True           # 是否开启自适应采样
    ADAPTIVE_SAMPLING_WINDOW = 200      # 统计胜率的窗口大小
    ADAPTIVE_SAMPLING_MIN_GAMES = 10   # 最小对局数
    ADAPTIVE_SAMPLING_STRENGTH = 1.0   # 采样强度 (越大越倾向于选择弱势对手)
    
    # 随机对手权重 (初始权重)
    RANDOM_OPPONENT_WEIGHTS = {
        # "target": {"Greedy": 1.0, "APF": 0, "DWA": 0, "Hiding": 0, "Random": 0}
        "target": {"Greedy": 1.0, "APF": 1.0, "DWA": 1.0, "Hiding": 1.0, "Random": 1.0}
    }

class NetParameters:
    """
    网络结构参数
    """
    ACTOR_VECTOR_LEN = 27    # Actor输入维度 (Tracker Observation)
    PRIVILEGED_LEN = 24      # 特权信息维度 (Target Observation)
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + PRIVILEGED_LEN # Critic输入维度
    ACTION_DIM = 2           # 动作维度 (Angle, Speed)
    
    # MLP 特有参数
    HIDDEN_DIM = 256         # 隐藏层维度
    CONTEXT_WINDOW = TrainingParameters.TBPTT_STEPS # 上下文窗口长度 (用于数据处理)

class RecordingParameters:
    """
    日志与记录参数
    """
    EXPERIMENT_PROJECT = "AvoidMaker_MLP"
    EXPERIMENT_NAME = "mlp_ppo_oneop_rl"
    ENTITY = "user"
    EXPERIMENT_NOTE = "MLP PPO training with separate radar encoding"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = True          # 是否继续训练
    RESTORE_DIR = "./models/mlp_ppo_oneop_rl_12-01-17-05/best_model/checkpoint.pth"       # 恢复模型的目录
    
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