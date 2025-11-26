import datetime
from map_config import EnvParameters

class SetupParameters:
    SEED = 1234
    NUM_GPU = 1
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    PRETRAINED_TARGET_PATH = None
    PRETRAINED_TRACKER_PATH = None

class TrainingParameters:
    lr = 3e-4
    LR_FINAL = 3e-5
    LR_SCHEDULE = 'cosine'
    
    N_ENVS = 4
    N_STEPS = 2048       # Standard PPO rollout length for MLP
    N_MAX_STEPS = 3e7
    LOG_EPOCH_STEPS = int(2e5)
    
    MINIBATCH_SIZE = 64
    N_EPOCHS = 10
    
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
    CONTEXT_LEN = EnvParameters.NUM_TARGET_POLICIES
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + CONTEXT_LEN
    ACTION_DIM = 2
    
    # MLP Specific
    HIDDEN_DIM = 256

class RecordingParameters:
    EXPERIMENT_PROJECT = "AvoidMaker_MLP"
    EXPERIMENT_NAME = "mlp_ppo"
    ENTITY = "user"
    EXPERIMENT_NOTE = "MLP PPO training with separate radar encoding"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False
    RESTORE_DIR = None
    
    WANDB = False
    TENSORBOARD = True
    TXT_LOG = True
    
    SUMMARY_PATH = f'./runs/TrackingEnv/{EXPERIMENT_NAME}{TIME}'
    MODEL_PATH = f'./models/TrackingEnv/{EXPERIMENT_NAME}{TIME}'
    GIFS_PATH = f'./models/TrackingEnv/{EXPERIMENT_NAME}{TIME}/gifs'
    
    EVAL_INTERVAL = 100000
    SAVE_INTERVAL = 500000
    BEST_INTERVAL = 0
    GIF_INTERVAL = 200000
    EVAL_EPISODES = 8
    
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std', 
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean'
    ]
