import datetime
from map_config import EnvParameters

class SetupParameters:
    SEED = 1234
    NUM_GPU = 1
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    # 预训练模型路径
    PRETRAINED_TARGET_PATH = None
    PRETRAINED_TRACKER_PATH = None

class TrainingParameters:
    lr = 3e-4
    LR_FINAL = 3e-5
    LR_SCHEDULE = 'cosine'
    
    N_ENVS = 4
    N_STEPS = 512        # 每个环境采样的步数
    N_MAX_STEPS = 3e7    # 总训练步数
    
    MINIBATCH_SIZE = 512
    TBPTT_STEPS = 32     # LSTM截断反向传播步长
    N_EPOCHS = 4
    
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    RATIO_CLAMP_MAX = 4.0
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    GAMMA = 0.99
    LAM = 0.95
    
    # 训练配置 - 固定为训练tracker
    OPPONENT_TYPE = "random"    # "random", "policy", "random_nonexpert"
    IL_TYPE = "expert"          # 模仿学习类型
    
    # 多对手自适应采样配置
    ADAPTIVE_SAMPLING = True
    ADAPTIVE_SAMPLING_WINDOW = 50
    ADAPTIVE_SAMPLING_MIN_GAMES = 10
    ADAPTIVE_SAMPLING_STRENGTH = 1.0
    
    # 随机对手权重 (target策略)
    RANDOM_OPPONENT_WEIGHTS = {
        "target": {"Greedy": 1.0, "APF": 1.0, "DWA": 1.0, "Hiding": 1.0, "Random": 0.5}
    }

class NetParameters:
    # 观测维度
    ACTOR_VECTOR_LEN = 27    # Tracker: 27, Target: 24. 取最大值
    CONTEXT_LEN = EnvParameters.NUM_TARGET_POLICIES          # 对手策略ID的one-hot长度 (对应NUM_TARGET_POLICIES)
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + CONTEXT_LEN
    ACTION_DIM = 2

class RecordingParameters:
    EXPERIMENT_PROJECT = "AvoidMaker_LSTM"
    EXPERIMENT_NAME = "lstm_ppo"
    ENTITY = "user"
    EXPERIMENT_NOTE = "LSTM PPO training with multi-opponent"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    # NOTE:
    # - MODEL_PATH: 保存 checkpoint 和 tfevents（在 driver_lstm 里将 SummaryWriter 指到 MODEL_PATH/tfevents）
    # - GIFS_PATH: 专门保存 GIF，便于在 git 中忽略
    RETRAIN = False
    RESTORE_DIR = None
    
    WANDB = False
    TENSORBOARD = True
    TXT_LOG = True
    
    SUMMARY_PATH = f'./runs/TrackingEnv/{EXPERIMENT_NAME}{TIME}'
    MODEL_PATH = f'./models/TrackingEnv/{EXPERIMENT_NAME}{TIME}'
    GIFS_PATH = f'./models/TrackingEnv/{EXPERIMENT_NAME}{TIME}/gifs'
    
    EVAL_INTERVAL = 50000
    SAVE_INTERVAL = 500000
    BEST_INTERVAL = 0
    GIF_INTERVAL = 200000
    EVAL_EPISODES = 16
    
    # 按 Model.train 返回顺序命名（只影响 tensorboard / 打印标签，不影响训练）
    LOSS_NAME = [
        'total',          # 0
        'policy',         # 1
        'entropy',        # 2
        'value',          # 3
        'adv_std',        # 4
        'approx_kl',      # 5
        'value_clip_frac',# 6
        'clipfrac',       # 7
        'grad_norm',      # 8
        'adv_mean'        # 9
    ]