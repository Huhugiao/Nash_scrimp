import datetime
from map_config import EnvParameters, ObstacleDensity

class SetupParameters:
    SEED = 1234
    NUM_GPU = 1
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    PRETRAINED_SKILL_PATH = None
    OBSTACLE_DENSITY = ObstacleDensity.SPARSE

class TrainingParameters:
    lr = 3e-4
    LR_FINAL = 1e-5
    LR_SCHEDULE = 'cosine'
    
    N_ENVS = 4
    N_STEPS = 2048
    N_MAX_STEPS = 30e7
    LOG_EPOCH_STEPS = int(1e4)
    
    MINIBATCH_SIZE = 64
    N_EPOCHS = 10
    
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    GAMMA = 0.99
    LAM = 0.95
    
    PRETRAIN_SKILLS = False
    PRETRAIN_STEPS = 1e6
    
    FREEZE_SKILLS = True
    HIGH_LEVEL_LR = 1e-3
    
    HARD_SKILL_SELECTION = False
    
    OPPONENT_TYPE = "random"
    RANDOM_OPPONENT_WEIGHTS = {
        "target": {
            "Greedy": 1.0,
        }
    }

class NetParameters:
    RADAR_DIM = 64
    RADAR_EMBED_DIM = 8
    
    ACTOR_SCALAR_LEN = 10
    PRIVILEGED_SCALAR_LEN = 8
    
    ACTOR_RAW_LEN = ACTOR_SCALAR_LEN + RADAR_DIM
    PRIVILEGED_RAW_LEN = PRIVILEGED_SCALAR_LEN + RADAR_DIM
    CRITIC_RAW_LEN = ACTOR_RAW_LEN + PRIVILEGED_RAW_LEN
    
    ACTOR_VECTOR_LEN = ACTOR_SCALAR_LEN + RADAR_EMBED_DIM
    PRIVILEGED_LEN = PRIVILEGED_SCALAR_LEN + RADAR_EMBED_DIM 
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + PRIVILEGED_LEN
    
    ACTION_DIM = 2
    
    HIDDEN_DIM = 256
    NUM_HIDDEN_LAYERS = 3
    FEATURE_DIM = 128
    
    SKILL_TAG_DIM = 2
    
    HIGH_LEVEL_HIDDEN = [128, 64]

class RecordingParameters:
    EXPERIMENT_PROJECT = "AvoidMaker_StarHRL"
    EXPERIMENT_NAME = "star_hrl"
    
    ENABLE_SAFETY_LAYER = False
    
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False
    RESTORE_DIR = None
    
    TENSORBOARD = True
    TXT_LOG = True
    
    _DENSITY_TAG = f'_{SetupParameters.OBSTACLE_DENSITY}'
    SUMMARY_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    MODEL_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    
    EVAL_INTERVAL = 100000
    SAVE_INTERVAL = 300000
    EVAL_EPISODES = 48
    
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std',
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean',
        'high_level_loss'
    ]
