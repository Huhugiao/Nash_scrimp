import datetime
from map_config import EnvParameters

class SetupParameters:
    SEED = 1234
    NUM_GPU = 1
    USE_GPU_LOCAL = True
    USE_GPU_GLOBAL = True
    PRETRAINED_TARGET_PATH = None
    PRETRAINED_TRACKER_PATH = None

class TrainingParameters:
    lr = 3e-4
    LR_FINAL = 3e-5
    LR_SCHEDULE = 'cosine'
    
    N_ENVS = 3
    N_STEPS = 1024        # Steps per environment rollout
    N_MAX_STEPS = 7e7    # Total training steps
    LOG_EPOCH_STEPS = 5e4
    
    MINIBATCH_SIZE = 64 
    
    # For MHA, this defines the sequence length for training (Context Window)
    TBPTT_STEPS = 32     
    N_EPOCHS = 4
    
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    RATIO_CLAMP_MAX = 4.0
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    GAMMA = 0.99
    LAM = 0.95
    
    OPPONENT_TYPE = "random"
    IL_TYPE = "expert"
    
    ADAPTIVE_SAMPLING = True
    ADAPTIVE_SAMPLING_WINDOW = 50
    ADAPTIVE_SAMPLING_MIN_GAMES = 10
    ADAPTIVE_SAMPLING_STRENGTH = 1.0
    
    RANDOM_OPPONENT_WEIGHTS = {
        "target": {"Greedy": 1.0, "APF": 1.0, "DWA": 1.0, "Hiding": 1.0, "Random": 0.5}
    }

class NetParameters:
    ACTOR_VECTOR_LEN = 27
    TARGET_OBS_LEN = 24
    CRITIC_VECTOR_LEN = TARGET_OBS_LEN
    ACTION_DIM = 2
    
    # MHA Specific Configuration
    HIDDEN_DIM = 128
    N_HEADS = 4
    N_LAYERS = 2
    CONTEXT_WINDOW = 32  # Sliding window size for inference and training

class RecordingParameters:
    EXPERIMENT_PROJECT = "AvoidMaker_MHA"
    EXPERIMENT_NAME = "mha_ppo"
    ENTITY = "user"
    EXPERIMENT_NOTE = "MHA PPO training"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False
    RESTORE_DIR = './models/TrackingEnv/mha_ppo_11-25-22-10/best_model'
    
    WANDB = False
    TENSORBOARD = True
    TXT_LOG = True
    
    SUMMARY_PATH = f'./runs/TrackingEnv/{EXPERIMENT_NAME}{TIME}'
    MODEL_PATH = f'./models/TrackingEnv/{EXPERIMENT_NAME}{TIME}'
    GIFS_PATH = f'./models/TrackingEnv/{EXPERIMENT_NAME}{TIME}/gifs'
    
    EVAL_INTERVAL = 100000   # 降低评估间隔
    SAVE_INTERVAL = 200000   # 降低保存间隔
    BEST_INTERVAL = 0
    GIF_INTERVAL = 200000    # 添加 GIF 间隔
    EVAL_EPISODES = 16
    
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std', 
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean'
    ]
