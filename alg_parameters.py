import datetime
from map_config import EnvParameters

""" Hyperparameters for Tracking Environment"""


class TrainingParameters:
    lr = 1e-4  # learning rate
    GAMMA = 0.99  # discount factor
    LAM = 0.95  # GAE lambda (for future use)
    CLIP_RANGE = 0.2  # PPO clipping range
    MAX_GRAD_NORM = 20  # gradient clipping
    ENTROPY_COEF = 0.01  # entropy coefficient
    EX_VALUE_COEF = 0.5  # value loss coefficient
    N_EPOCHS = 4  # number of training epochs
    N_ENVS = 20  # number of parallel environments
    N_MAX_STEPS = 12e7  # maximum training steps
    N_STEPS = 2 ** 10  # steps per environment per collection
    MINIBATCH_SIZE = int(2 ** 8)  # minibatch size for training
    DEMONSTRATION_PROB = 0.1  # probability of imitation learning
    TBPTT_STEPS = 32  # sequence length for truncated BPTT
    LR_FINAL = 5e-5  # final learning rate for scheduling
    LR_SCHEDULE = "cosine"  # learning rate decay schedule
    VALUE_CLIP_RANGE = 0.2  # PPO value clipping range

    # Adaptive opponent sampling parameters
    ADAPTIVE_SAMPLING = True  # 是否启用自适应对手采样
    ADAPTIVE_SAMPLING_WINDOW = 400  # 用于计算胜率的最近回合数
    ADAPTIVE_SAMPLING_MIN_GAMES = 48  # 最小历史游戏数，用于自适应采样的保护窗口
    ADAPTIVE_SAMPLING_STRENGTH = 1.8  # 采样强度，值越大，胜率低对手的采样权重越高

    # Agent training configuration
    AGENT_TO_TRAIN = "tracker"  # "tracker" or "target"
    OPPONENT_TYPE = "random"  # "expert", "policy", "random", or "random_nonexpert"

    # Random opponent weights - 只保留Greedy target策略
    RANDOM_OPPONENT_WEIGHTS = {
        "tracker": {
            "expert_tracker": 1.0  # 只使用专家tracker策略
        },
        "target": {
            "Greedy": 1.0  # 只使用Greedy target策略
        }
    }

    # IL configuration
    IL_TYPE = "expert"  # "expert" or "policy"
    IL_TEACHER_TRACKER_PATH = "./models/TrackingEnv/DualAgent23-09-252041/best_model/tracker_net_checkpoint.pkl"
    IL_TEACHER_TARGET_PATH = "./models/pretrained/target_teacher.pkl"


class NetParameters:
    # 新的观测维度设计：
    # 
    # Tracker观测 (27维) - 局部感知，受FOV限制:
    #   自身状态 (3): 线速度, 角速度, 朝向
    #   目标相对状态 (5): 距离d, 方位角α, 相对线速度, 相对角速度, FOV边界角
    #   目标可见性 (3): in_FOV标志, occluded标志, 未观测步数
    #   雷达 (16): 360度全向扫描的障碍物距离（16个方向）
    # 
    # Target观测 (24维) - 全局感知，无FOV限制（全知视角）:
    #   自身绝对位置 (2): x, y
    #   自身朝向 (1): heading
    #   tracker绝对位置 (2): x, y（始终可见）
    #   tracker朝向 (1): heading（始终可见）
    #   自身速度 (1): velocity magnitude
    #   tracker速度 (1): velocity magnitude（始终可见）
    #   雷达 (16): 360度全向扫描的障碍物距离（16个方向）
    ACTOR_VECTOR_LEN = 27  # Tracker使用27维，Target使用24维

    # 对手策略ID的维度
    CONTEXT_LEN = EnvParameters.NUM_TARGET_POLICIES

    # Critic使用的完整观测维度 (Actor观测 + 对手ID)
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + CONTEXT_LEN

    # 动作维度
    ACTION_DIM = 2


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = True
    USE_GPU_GLOBAL = True
    NUM_GPU = 1

    # Pre-trained model paths (for policy opponents)
    PRETRAINED_TRACKER_PATH = "./models/TrackingEnv/pretrained_tracker/tracker_net_checkpoint.pkl"
    PRETRAINED_TARGET_PATH = "./models/TrackingEnv/pretrained_target/target_net_checkpoint.pkl"


class RecordingParameters:
    RETRAIN = False
    RESTORE_DIR = "./models/TrackingEnv/DualAgent08-10-251922/final"  # 设置为你要恢复的模型路径

    WANDB = False
    TENSORBOARD = True
    TXT_WRITER = True
    ENTITY = 'yutong'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'TrackingEnv'
    EXPERIMENT_NAME = 'DualAgent'
    EXPERIMENT_NOTE = ''

    # Intervals
    SAVE_INTERVAL = 5e5
    BEST_INTERVAL = 0
    GIF_INTERVAL = 2e6
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
    EVAL_EPISODES = 8

    # Paths
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    TXT_NAME = 'alg.txt'

    # Loss names for logging
    LOSS_NAME = ['total_loss', 'policy_loss', 'entropy', 'value_loss',
                 'adv_std', 'approx_kl', 'value_clip_frac', 'clipfrac',
                 'grad_norm', 'advantages_mean']