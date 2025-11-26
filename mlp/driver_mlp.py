import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import os.path as osp
import math
import numpy as np
import torch
import ray
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import setproctitle
except Exception:
    setproctitle = None

from torch.utils.tensorboard import SummaryWriter
from map_config import EnvParameters
from mlp.alg_parameters_mlp import *
from env import TrackingEnv
from mlp.model_mlp import Model
from mlp.runner_mlp import Runner
from util import set_global_seeds, make_gif, get_opponent_id_one_hot
from rule_policies import TARGET_POLICY_REGISTRY
from policymanager import PolicyManager

IL_INITIAL_PROB = 0.8
IL_FINAL_PROB = 0.1
IL_DECAY_STEPS = 1e7

if not ray.is_initialized():
    ray.init(num_gpus=SetupParameters.NUM_GPU)

def_attr = lambda name, default: getattr(RecordingParameters, name, default)
SUMMARY_PATH = def_attr('SUMMARY_PATH', './runs')
MODEL_PATH = def_attr('MODEL_PATH', './models')
GIFS_PATH = def_attr('GIFS_PATH', osp.join(MODEL_PATH, 'gifs'))

def main():
    if setproctitle:
        setproctitle.setproctitle("AvoidMaker_MLP")
    set_global_seeds(SetupParameters.SEED)
    
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    training_model = Model(global_device, True)
    
    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    
    curr_steps = 0
    
    global_summary = SummaryWriter(osp.join(MODEL_PATH, "tfevents")) if RecordingParameters.TENSORBOARD else None
    
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            il_prob = 0.1 # Simplified
            do_il = (np.random.rand() < il_prob)
            
            weights = training_model.get_weights()
            
            if do_il:
                jobs = [e.imitation.remote(weights, None, curr_steps) for e in envs]
                results = ray.get(jobs)
                curr_steps += TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
            else:
                jobs = [e.run.remote(weights, None, curr_steps, None) for e in envs]
                results = ray.get(jobs)
                
                # Flatten results for MLP training
                actor_obs = []
                critic_obs = []
                returns = []
                values = []
                actions = []
                old_log_probs = []
                
                steps_batch = 0
                for r in results:
                    data = r[0]
                    actor_obs.append(data['actor_obs'])
                    critic_obs.append(data['critic_obs'])
                    returns.append(data['returns'])
                    values.append(data['values'])
                    actions.append(data['actions'])
                    old_log_probs.append(data['logp'])
                    steps_batch += r[1]
                
                # Concatenate all env data
                actor_obs = np.concatenate(actor_obs, axis=0)
                critic_obs = np.concatenate(critic_obs, axis=0)
                returns = np.concatenate(returns, axis=0)
                values = np.concatenate(values, axis=0)
                actions = np.concatenate(actions, axis=0)
                old_log_probs = np.concatenate(old_log_probs, axis=0)
                
                total_samples = actor_obs.shape[0]
                indices = np.arange(total_samples)
                
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(indices)
                    for start in range(0, total_samples, TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        idx = indices[start:end]
                        
                        training_model.train(
                            actor_obs[idx], critic_obs[idx], returns[idx],
                            values[idx], actions[idx], old_log_probs[idx]
                        )
                
                curr_steps += steps_batch
                
                if curr_steps % 10000 == 0:
                    print(f"Step {curr_steps}")
                    
    except KeyboardInterrupt:
        print("Stopping...")
        ray.shutdown()

if __name__ == "__main__":
    main()
