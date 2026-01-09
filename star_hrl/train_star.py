import os
import sys
import math
import random
import datetime
from collections import deque
import numpy as np
import torch
import ray
from torch.utils.tensorboard import SummaryWriter
import setproctitle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from star_hrl.alg_parameters_star import *
from star_hrl.model_star import StarModel
from star_hrl.runner_star import StarRunner
from mlp.util_mlp import set_global_seeds, write_to_tensorboard
from mlp.policymanager_mlp import PolicyManager
from map_config import set_obstacle_density

# Set global obstacle density
set_obstacle_density(SetupParameters.OBSTACLE_DENSITY)

def extract_rl_data_from_rollout(rollout_data):
    data = rollout_data['data']
    return {
        'actor_obs': data['actor_obs'],
        'critic_obs': data['critic_obs'],
        'returns': data['returns'],
        'values': data['values'],
        'actions': data['actions'],
        'logp': data['logp'],
        'dones': data['dones'],
        'episode_starts': data['episode_starts'],
        'high_weights': data['high_weights'],
    }

def build_segments_from_rollout(rollout, window_size=None):
    # Note: Window size is not strictly used for simple PPO buffer, 
    # but we keep the structure for consistency if we add RNNs later.
    # For MLP/Star, we usually just take full episodes or chunks.
    # Here we simulate the segment structure.
    
    segments = []
    total_steps = rollout['actor_obs'].shape[0]
    
    # For non-recurrent policies, we can treat the whole rollout as one "segment"
    # or chop it up. Let's just wrap the whole thing or chop by episodes.
    # To keep it simple and consistent with common PPO impls:
    
    segments.append({
        'actor_obs': rollout['actor_obs'],
        'critic_obs': rollout['critic_obs'],
        'returns': rollout['returns'],
        'values': rollout['values'],
        'actions': rollout['actions'],
        'old_log_probs': rollout['logp'],
        'high_weights': rollout['high_weights'],
        'mask': np.ones(total_steps, dtype=np.float32)
    })
    return segments

def collate_segments(batch_segments):
    # Since we might have variable lengths if we chopped by episode, 
    # but here we just took the whole n_steps block.
    # Actually, let's flatten everything here for the PPO update.
    
    # Inspect first segment to get shapes
    first_seg = batch_segments[0]
    
    # Calculate total samples
    total_samples = sum(len(s['actor_obs']) for s in batch_segments)
    
    batch = {
        'actor_obs': np.zeros((total_samples, NetParameters.ACTOR_RAW_LEN), dtype=np.float32),
        'critic_obs': np.zeros((total_samples, NetParameters.CRITIC_RAW_LEN), dtype=np.float32),
        'returns': np.zeros((total_samples,), dtype=np.float32),
        'values': np.zeros((total_samples,), dtype=np.float32),
        'actions': np.zeros((total_samples, NetParameters.ACTION_DIM), dtype=np.float32),
        'old_log_probs': np.zeros((total_samples,), dtype=np.float32),
        'mask': np.zeros((total_samples,), dtype=np.float32),
        'high_weights': np.zeros((total_samples, 2), dtype=np.float32),
    }
    
    cursor = 0
    for seg in batch_segments:
        l = len(seg['actor_obs'])
        batch['actor_obs'][cursor:cursor+l] = seg['actor_obs']
        batch['critic_obs'][cursor:cursor+l] = seg['critic_obs']
        batch['returns'][cursor:cursor+l] = seg['returns']
        batch['values'][cursor:cursor+l] = seg['values']
        batch['actions'][cursor:cursor+l] = seg['actions']
        batch['old_log_probs'][cursor:cursor+l] = seg['old_log_probs']
        batch['mask'][cursor:cursor+l] = seg['mask']
        batch['high_weights'][cursor:cursor+l] = seg['high_weights']
        cursor += l
        
    return batch

def compute_performance_stats(performance_dict):
    stats = {}
    for key, values in performance_dict.items():
        if values:
            stats[f'{key}_mean'] = float(np.nanmean(values))
            stats[f'{key}_std'] = float(np.nanstd(values))
        else:
            stats[f'{key}_mean'] = 0.0
            stats[f'{key}_std'] = 0.0
    return stats

def get_scheduled_lr(current_step):
    """Cosine annealing learning rate scheduler."""
    initial_lr = TrainingParameters.lr
    final_lr = TrainingParameters.LR_FINAL
    max_steps = TrainingParameters.N_MAX_STEPS
    
    if TrainingParameters.LR_SCHEDULE == 'linear':
        progress = min(max(current_step / max_steps, 0.0), 1.0)
        return initial_lr - (initial_lr - final_lr) * progress
    elif TrainingParameters.LR_SCHEDULE == 'cosine':
        progress = min(max(current_step / max_steps, 0.0), 1.0)
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (initial_lr - final_lr) * weight
    else:
        return initial_lr

def run_evaluation(envs, model_weights, opponent_weights, num_episodes):
    """
    Run evaluation episodes with greedy policy (no exploration).
    """
    eval_results = {'per_r': [], 'per_episode_len': [], 'win': []}
    episodes_collected = 0
    
    # Need to dispatch enough jobs to get num_episodes
    # Simple strategy: run all envs until we have enough
    while episodes_collected < num_episodes:
        jobs = [
            envs[i].run.remote(model_weights, opponent_weights, 0, None)
            for i in range(len(envs))
        ]
        results = ray.get(jobs)
        
        for result in results:
            perf = result['performance']
            eval_results['per_r'].extend(perf['per_r'])
            eval_results['per_episode_len'].extend(perf['per_episode_len'])
            eval_results['win'].extend(perf['win'])
            episodes_collected += result['episodes']
            
            if episodes_collected >= num_episodes:
                break
                
    return eval_results

def main():
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=SetupParameters.NUM_GPU)
        
    set_global_seeds(SetupParameters.SEED)
    if setproctitle:
        setproctitle.setproctitle(f"AvoidMaker_{RecordingParameters.EXPERIMENT_NAME}")
        
    # Paths
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    EXP_NAME = RecordingParameters.EXPERIMENT_NAME
    DENSITY_TAG = f'_{SetupParameters.OBSTACLE_DENSITY}'
    MODEL_PATH = f'./models/{EXP_NAME}{DENSITY_TAG}{TIME}'
    SUMMARY_PATH = MODEL_PATH
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Tensorboard
    global_summary = SummaryWriter(log_dir=SUMMARY_PATH)
    
    # Device
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    
    # Initialize Global Star Model
    training_model = StarModel(global_device, global_model=True)
    
    # Load Pretrained Skills if defined and we are in Phase 2
    if TrainingParameters.FREEZE_SKILLS and SetupParameters.PRETRAINED_SKILL_PATH:
        training_model.load_skill_weights(SetupParameters.PRETRAINED_SKILL_PATH)
        print(f"Pretrained skills loaded from {SetupParameters.PRETRAINED_SKILL_PATH}")
    
    # Initialize Policy Manager for Opponents
    global_pm = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
    
    # Initialize Runners
    envs = [StarRunner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    
    # Training Loop State
    curr_steps = 0
    curr_episodes = 0
    best_perf = -1e9
    
    epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
    epoch_loss_buffer = []
    
    last_train_log_t = 0
    last_save_t = 0
    last_eval_t = 0
    
    try:
        print(f"Starting Star HRL Training: {EXP_NAME}")
        print(f"Max Steps: {TrainingParameters.N_MAX_STEPS}")
        print(f"Skills Frozen: {TrainingParameters.FREEZE_SKILLS}")
        
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            # Scheduler
            if TrainingParameters.FREEZE_SKILLS:
                # Use HIGH_LEVEL_LR, maybe constant or scheduled?
                # For now let's keep it constant as per TrainingParameters or add scheduler logic
                # Usually high level policy converges faster or uses specific LR
                pass 
            else:
                new_lr = get_scheduled_lr(curr_steps)
                training_model.update_learning_rate(new_lr)
            
            # Get weights
            model_weights = training_model.get_weights()
            opponent_weights = None # Not used for 'random' opponent in runner currently
            
            pm_state = {k: list(v) for k, v in global_pm.win_history.items()} if global_pm else None
            
            # Launch Rollouts
            jobs = [
                envs[i].run.remote(model_weights, opponent_weights, curr_steps, pm_state)
                for i in range(TrainingParameters.N_ENVS)
            ]
            results = ray.get(jobs)
            
            all_segments = []
            total_new_episodes = 0
            
            for result in results:
                rl_data = extract_rl_data_from_rollout(result)
                all_segments.extend(build_segments_from_rollout(rl_data))
                
                perf = result['performance']
                epoch_perf_buffer['per_r'].extend(perf['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(perf['per_episode_len'])
                epoch_perf_buffer['win'].extend(perf['win'])
                
                total_new_episodes += result['episodes']
                
                if global_pm and result['policy_manager_state']:
                    for name, history in result['policy_manager_state'].items():
                        global_pm.win_history[name] = deque(history, maxlen=100) # Hardcoded maxlen for now or use param

            curr_steps += TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
            curr_episodes += total_new_episodes
            
            # PPO Update
            if all_segments:
                # Flat batching
                batch = collate_segments(all_segments)
                
                total_samples = len(batch['actor_obs'])
                indices = np.arange(total_samples)
                
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(indices)
                    for start in range(0, total_samples, TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        idx = indices[start:end]
                        
                        train_args = {
                            'writer': global_summary,
                            'global_step': curr_steps,
                            'actor_obs': batch['actor_obs'][idx],
                            'critic_obs': batch['critic_obs'][idx],
                            'returns': batch['returns'][idx],
                            'values': batch['values'][idx],
                            'actions': batch['actions'][idx],
                            'old_log_probs': batch['old_log_probs'][idx],
                            'mask': batch['mask'][idx],
                        }
                        
                        result = training_model.train(**train_args)
                        if isinstance(result, dict):
                            epoch_loss_buffer.append(result.get('losses', []))

            # Logging
            if curr_steps - last_train_log_t >= TrainingParameters.LOG_EPOCH_STEPS:
                last_train_log_t = curr_steps
                train_stats = compute_performance_stats(epoch_perf_buffer) if epoch_perf_buffer['per_r'] else {}
                
                # Check mean high weights
                mean_high_w = np.mean(batch['high_weights'], axis=0) if 'high_weights' in batch else [0,0]
                
                current_lr = training_model.current_lr
                log_str = f"[TRAIN] step={curr_steps:,} | ep={curr_episodes:,} | LR={current_lr:.2e}"
                if train_stats:
                    log_str += f" | Rew={train_stats.get('per_r_mean', 0):.2f} | Win={train_stats.get('win_mean', 0)*100:.1f}%"
                log_str += f" | W_track={mean_high_w[0]:.2f} W_safe={mean_high_w[1]:.2f}"
                print(log_str)
                
                if global_summary:
                    write_to_tensorboard(global_summary, curr_steps, performance_dict=epoch_perf_buffer, 
                                        mb_loss=epoch_loss_buffer, imitation_loss=[0.0, 0.0], q_loss=0.0, evaluate=False)
                    global_summary.add_scalar('Train/W_track', mean_high_w[0], curr_steps)
                    global_summary.add_scalar('Train/W_safe', mean_high_w[1], curr_steps)

                epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
                epoch_loss_buffer = []

            # Periodic Evaluation
            if curr_steps - last_eval_t >= RecordingParameters.EVAL_INTERVAL:
                last_eval_t = curr_steps
                print(f"[EVAL] Running evaluation at step={curr_steps}...")
                
                eval_results = run_evaluation(envs, training_model.get_weights(), None, RecordingParameters.EVAL_EPISODES)
                eval_stats = compute_performance_stats(eval_results)
                
                eval_reward = eval_stats.get('per_r_mean', 0.0)
                eval_win_rate = eval_stats.get('win_mean', 0.0) * 100
                
                print(f"[EVAL] step={curr_steps:,} | Reward={eval_reward:.2f} | WinRate={eval_win_rate:.1f}%")
                
                if global_summary:
                    global_summary.add_scalar('Eval/Reward', eval_reward, curr_steps)
                    global_summary.add_scalar('Eval/WinRate', eval_win_rate, curr_steps)
                
                # Best Model
                if eval_reward > best_perf:
                    best_perf = eval_reward
                    best_path = os.path.join(MODEL_PATH, 'best_model.pth')
                    training_model.save_checkpoint(best_path)
                    print(f"[BEST] New best model saved! Reward={eval_reward:.2f}")

            # Save Latest
            if curr_steps - last_save_t >= RecordingParameters.SAVE_INTERVAL:
                last_save_t = curr_steps
                latest_path = os.path.join(MODEL_PATH, 'latest_model.pth')
                training_model.save_checkpoint(latest_path)
                print(f"[SAVE] Latest model saved at step={curr_steps}")

    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        final_path = os.path.join(MODEL_PATH, 'final_model.pth')
        training_model.save_checkpoint(final_path)
        print(f"Final model saved to {final_path}")
        if global_summary:
            global_summary.close()
        ray.shutdown()

if __name__ == "__main__":
    main()
