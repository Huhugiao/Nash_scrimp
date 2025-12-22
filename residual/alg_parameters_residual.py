"""
Residual RL 模块参数配置

独立于 mlp/alg_parameters_mlp.py，专门用于残差学习的参数。
复用部分通用参数从 mlp 模块导入。
"""
import datetime
from map_config import EnvParameters

# 从 mlp 模块复用基础参数
from mlp.alg_parameters_mlp import (
    SetupParameters,
    NetParameters,
    RecordingParameters,
    TrainingParameters as BaseTrainingParameters,
)


class ResidualNetParameters:
    """
    残差网络结构参数
    比 MLP 主网络更小，因为只需要学习微调
    """
    # 网络大小（可自定义）
    HIDDEN_DIM = 128          
    NUM_LAYERS = 3 
    
    # 残差缩放系数
    RESIDUAL_SCALE = 0.0     # α: final = base + α * residual
    
    # 输入输出维度（与主网络一致）
    ACTOR_RAW_LEN = NetParameters.ACTOR_RAW_LEN      # 75
    CRITIC_RAW_LEN = NetParameters.CRITIC_RAW_LEN    # 147
    ACTION_DIM = NetParameters.ACTION_DIM            # 2
    
    # 雷达编码（复用主网络配置）
    RADAR_DIM = NetParameters.RADAR_DIM              # 64
    RADAR_EMBED_DIM = NetParameters.RADAR_EMBED_DIM  # 8
    ACTOR_SCALAR_LEN = NetParameters.ACTOR_SCALAR_LEN  # 11


class ResidualTrainingParameters(BaseTrainingParameters):
    """
    残差训练超参数
    相比主 RL 训练，使用更保守的设置
    """
    # 优化器设置（更小的学习率用于微调）
    lr = 1e-4
    LR_FINAL = 1e-5
    LR_SCHEDULE = 'cosine'
    
    # 训练流程
    N_ENVS = 4               # 并行环境数
    N_STEPS = 2048           # 每个环境采样步数
    N_MAX_STEPS = int(10e6)   # 总训练步数（比主 RL 少）
    
    LOG_EPOCH_STEPS = int(1e4)
    
    # Mini-batch 设置
    MINIBATCH_SIZE = 64
    N_EPOCHS_INITIAL = 8     # 比主 RL 少
    N_EPOCHS_FINAL = 3
    
    # PPO 参数（略微保守）
    CLIP_RANGE = 0.1
    VALUE_CLIP_RANGE = 0.2
    ENTROPY_COEF = 0.01     
    EX_VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    GAMMA = 0.99
    LAM = 0.95
    
    # 观测噪声（可选，用于鲁棒性训练）
    OBS_NOISE_STD = 0.0
    
    # 禁用环境 Safety Layer（CBF 基础策略已提供安全保障）
    SAFETY_LAYER_ENABLED = False
    
    # 对手设置（复用主训练配置）
    OPPONENT_TYPE = "random"
    RANDOM_OPPONENT_WEIGHTS = {
        "target": {
            "CoverSeeker": 1.0,
        }
    }
    
    # 是否在 loss 中加入残差惩罚（鼓励小残差）
    RESIDUAL_PENALTY_COEF = 0.01
    
    # 残差动作在环境奖励中的惩罚（传递给 env.reward_calculate）
    ACTION_PENALTY_COEF = 0.0


class ResidualRecordingParameters:
    """
    残差训练日志参数
    """
    EXPERIMENT_PROJECT = "AvoidMaker_Residual"
    EXPERIMENT_NAME = "residual_cbf"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    # 路径
    SUMMARY_PATH = f'./models/{EXPERIMENT_NAME}{TIME}'
    MODEL_PATH = f'./models/{EXPERIMENT_NAME}{TIME}'
    
    # 频率
    EVAL_INTERVAL = 50000
    SAVE_INTERVAL = 100000
    GIF_INTERVAL = 100000
    EVAL_EPISODES = 30
    
    # Loss 名称
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'residual_penalty',
        'approx_kl', 'clipfrac', 'grad_norm', 'adv_mean', 'adv_std'
    ]


class ResidualRLParameters:
    """
    运行期参数桥接：兼容现有 runner/train 逻辑。
    """
    ENABLED = True
    EXPERIMENT_NAME = ResidualRecordingParameters.EXPERIMENT_NAME
    RESIDUAL_LR = ResidualTrainingParameters.lr
    RESIDUAL_MAX_STEPS = ResidualTrainingParameters.N_MAX_STEPS
    ACTION_PENALTY_COEF = ResidualTrainingParameters.ACTION_PENALTY_COEF
    # 基础策略使用 CBF，无需 checkpoint
    BASE_MODEL_PATH = None
