import datetime

class TargetSetupParameters:
    SEED = 1234
    NUM_GPU = 1
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True

class TargetTrainingParameters:
    # Training Flow
    N_ENVS = 4
    N_STEPS = 2048 # PPO Standard Horizon
    N_MAX_STEPS = 1e7 # Total interaction steps
    
    # PPO Params
    BATCH_SIZE = 64 # Mini-batch size
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    LAM = 0.95 # GAE Lambda
    CLIP_RANGE = 0.2
    ENT_COEF = 0.001
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    EPOCHS = 10 # PPO Update Epochs
    
    # Reward Styles
    STYLES = ["survival", "stealth", "taunt"]
    NUM_STYLES = len(STYLES)
    STYLE_MAP = {name: i for i, name in enumerate(STYLES)}

class TargetNetParameters:
    # Radar
    RADAR_DIM = 64
    RADAR_EMBED_DIM = 8
    
    # Target Observation (Actor Input)
    TARGET_SCALAR_LEN = 8
    TARGET_RAW_LEN = 72
    
    # Tracker Observation (For Critic Context)
    TRACKER_SCALAR_LEN = 11
    TRACKER_RAW_LEN = 75
    
    # Critic Input (State)
    # Critic sees Full State
    STATE_DIM = TARGET_RAW_LEN + TRACKER_RAW_LEN
    ACTION_DIM = 2
    
    # Encoded Dimensions
    TARGET_VECTOR_LEN = TARGET_SCALAR_LEN + RADAR_EMBED_DIM # 16
    TRACKER_VECTOR_LEN = TRACKER_SCALAR_LEN + RADAR_EMBED_DIM # 19
    STATE_VECTOR_LEN = TARGET_VECTOR_LEN + TRACKER_VECTOR_LEN # 35
    
    # MLP
    HIDDEN_DIM = 128
    NUM_HIDDEN_LAYERS = 3

class TargetRecordingParameters:
    EXPERIMENT_PROJECT = "AvoidMaker_Target"
    EXPERIMENT_NAME = "target_ppo"
    ENTITY = "user"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False
    RESTORE_DIR = ""
    
    TENSORBOARD = True
    
    SUMMARY_PATH = f'./target_models/{EXPERIMENT_NAME}{TIME}'
    MODEL_PATH = f'./target_models/{EXPERIMENT_NAME}{TIME}'
    GIFS_PATH = f'./target_models/{EXPERIMENT_NAME}{TIME}/gifs'
    
    EVAL_INTERVAL = 50000
    SAVE_INTERVAL = 200000
    GIF_INTERVAL = 150000
