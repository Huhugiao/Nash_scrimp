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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import setproctitle

from residual.alg_parameters_residual import *
from residual.model_residual import ResidualModel
from residual.runner_residual import ResidualRunner
from mlp.util_mlp import set_global_seeds, make_gif, write_to_tensorboard
from mlp.policymanager_mlp import PolicyManager
from map_config import set_obstacle_density

# 设置障碍物密度等级
set_obstacle_density(SetupParameters.OBSTACLE_DENSITY)

def extract_rl_data_from_rollout(rollout_data):
    data = rollout_data['data']
    return {
        'actor_obs': data['actor_obs'],
        'critic_obs': data['critic_obs'],
        'radar_obs': data['radar_obs'],
        'velocity_obs': data['velocity_obs'],
        'base_actions': data['base_actions'],
        'returns': data['returns'],
        'values': data['values'],
        'actions': data['actions'],
        'logp': data['logp'],
        'dones': data['dones'],
        'episode_starts': data['episode_starts'],
    }

def build_segments_from_rollout(rollout, window_size):
    segments = []
    total_steps = rollout['actor_obs'].shape[0]
    current_start = 0
    for i in range(total_steps):
        if rollout['episode_starts'][i] and i > current_start:
            _flush_segments(rollout, current_start, i, window_size, segments)
            current_start = i
    if total_steps > current_start:
        _flush_segments(rollout, current_start, total_steps, window_size, segments)
    return segments

def _flush_segments(rollout, start, end, window_size, segments):
    cursor = start
    while cursor < end:
        seg_len = min(window_size, end - cursor)
        seg_end = cursor + seg_len
        segments.append({
            'actor_obs': rollout['actor_obs'][cursor:seg_end],
            'critic_obs': rollout['critic_obs'][cursor:seg_end],
            'radar_obs': rollout['radar_obs'][cursor:seg_end],
            'velocity_obs': rollout['velocity_obs'][cursor:seg_end],
            'base_actions': rollout['base_actions'][cursor:seg_end],
            'returns': rollout['returns'][cursor:seg_end],
            'values': rollout['values'][cursor:seg_end],
            'actions': rollout['actions'][cursor:seg_end],
            'old_log_probs': rollout['logp'][cursor:seg_end],
            'mask': np.ones(seg_len, dtype=np.float32)
        })
        cursor = seg_end

def collate_segments(batch_segments):
    batch_size = len(batch_segments)
    if batch_size == 0:
        raise ValueError("Empty segment batch")
    max_len = max(len(s['actor_obs']) for s in batch_segments)
    batch = {
        'actor_obs': np.zeros((batch_size, max_len, NetParameters.ACTOR_RAW_LEN), dtype=np.float32),
        'critic_obs': np.zeros((batch_size, max_len, NetParameters.CRITIC_RAW_LEN), dtype=np.float32),
        'radar_obs': np.zeros((batch_size, max_len, NetParameters.RADAR_DIM), dtype=np.float32),
        'velocity_obs': np.zeros((batch_size, max_len, NetParameters.VELOCITY_DIM), dtype=np.float32),
        'base_actions': np.zeros((batch_size, max_len, NetParameters.ACTION_DIM), dtype=np.float32),
        'returns': np.zeros((batch_size, max_len), dtype=np.float32),
        'values': np.zeros((batch_size, max_len), dtype=np.float32),
        'actions': np.zeros((batch_size, max_len, NetParameters.ACTION_DIM), dtype=np.float32),
        'old_log_probs': np.zeros((batch_size, max_len), dtype=np.float32),
        'mask': np.zeros((batch_size, max_len), dtype=np.float32),
    }
    for i, seg in enumerate(batch_segments):
        l = len(seg['actor_obs'])
        batch['actor_obs'][i, :l] = seg['actor_obs']
        batch['critic_obs'][i, :l] = seg['critic_obs']
        batch['radar_obs'][i, :l] = seg['radar_obs']
        batch['velocity_obs'][i, :l] = seg['velocity_obs']
        batch['base_actions'][i, :l] = seg['base_actions']
        batch['returns'][i, :l] = seg['returns']
        batch['values'][i, :l] = seg['values']
        batch['actions'][i, :l] = seg['actions']
        batch['old_log_probs'][i, :l] = seg['old_log_probs']
        batch['mask'][i, :l] = seg['mask']
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
    progress = min(max(current_step / max_steps, 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (initial_lr - final_lr) * weight

def get_scheduled_n_epochs(current_step):
    initial = TrainingParameters.N_EPOCHS_INITIAL
    final = TrainingParameters.N_EPOCHS_FINAL
    max_steps = TrainingParameters.N_MAX_STEPS
    progress = min(max(current_step / max_steps, 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    n_epochs = final + (initial - final) * weight
    return max(int(round(n_epochs)), 1)

def run_evaluation(envs, model_weights, opponent_weights, num_episodes):
    """
    Run evaluation episodes with greedy policy (no exploration).
    Returns aggregated performance statistics.
    """
    eval_results = {'per_r': [], 'per_episode_len': [], 'win': []}
    episodes_collected = 0
    
    while episodes_collected < num_episodes:
        # Run one batch of rollouts
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
    if not ResidualRLConfig.ENABLED:
        print("Residual RL is disabled in parameters.")
        return

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=SetupParameters.NUM_GPU)
        
    set_global_seeds(SetupParameters.SEED)
    if setproctitle:
        setproctitle.setproctitle(f"AvoidMaker_{ResidualRLConfig.EXPERIMENT_NAME}")
        
    # Paths (包含障碍物密度等级)
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    EXP_NAME = ResidualRLConfig.EXPERIMENT_NAME
    DENSITY_TAG = f'_{SetupParameters.OBSTACLE_DENSITY}'
    MODEL_PATH = f'./models/{EXP_NAME}{DENSITY_TAG}{TIME}'
    SUMMARY_PATH = MODEL_PATH
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Tensorboard
    global_summary = SummaryWriter(log_dir=SUMMARY_PATH)
    
    # Device
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    
    # Initialize Global Residual Model
    training_model = ResidualModel(global_device, True)
    
    # Global Opponent Model (Standard)
    opponent_model, opponent_weights = None, None
    if TrainingParameters.OPPONENT_TYPE == "policy":
        from mlp.model_mlp import Model # Import standard model for opponent
        opponent_model = Model(global_device, False)
        if SetupParameters.PRETRAINED_TARGET_PATH:
            opp_dict = torch.load(SetupParameters.PRETRAINED_TARGET_PATH, map_location='cpu')
            opponent_model.set_weights(opp_dict['model'])
        opponent_weights = opponent_model.get_weights()
        
    global_pm = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} and TrainingParameters.ADAPTIVE_SAMPLING else None

    # Initialize Residual Runners
    envs = [ResidualRunner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    
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
        print(f"Starting Residual RL Training: {EXP_NAME}")
        print(f"Base Model: {ResidualRLConfig.BASE_MODEL_PATH}")
        print(f"Max Steps: {TrainingParameters.N_MAX_STEPS}")
        
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            new_lr = get_scheduled_lr(curr_steps)
            training_model.update_learning_rate(new_lr)
            
            # Get weights (trainable residual weights)
            model_weights = training_model.get_weights()
            
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
                # Extract RL data (Residual Actions)
                rl_data = extract_rl_data_from_rollout(result)
                all_segments.extend(build_segments_from_rollout(rl_data, NetParameters.CONTEXT_WINDOW))
                
                # Stats
                perf = result['performance']
                epoch_perf_buffer['per_r'].extend(perf['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(perf['per_episode_len'])
                epoch_perf_buffer['win'].extend(perf['win'])
                total_new_episodes += result['episodes']
                
                if global_pm and result['policy_manager_state']:
                    for name, history in result['policy_manager_state'].items():
                        global_pm.win_history[name] = deque(history, maxlen=TrainingParameters.ADAPTIVE_SAMPLING_WINDOW)

            curr_steps += TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
            curr_episodes += total_new_episodes
            
            # PPO Update
            if all_segments:
                random.shuffle(all_segments)
                n_epochs = get_scheduled_n_epochs(curr_steps)
                
                for _ in range(n_epochs):
                    for mb_start in range(0, len(all_segments), TrainingParameters.MINIBATCH_SIZE):
                        mb_end = min(mb_start + TrainingParameters.MINIBATCH_SIZE, len(all_segments))
                        batch_segments = all_segments[mb_start:mb_end]
                        if not batch_segments:
                            continue
                            
                        batch = collate_segments(batch_segments)
                        
                        # Flatten
                        actor_flat = batch['actor_obs'].reshape(-1, NetParameters.ACTOR_RAW_LEN)
                        critic_flat = batch['critic_obs'].reshape(-1, NetParameters.CRITIC_RAW_LEN)
                        radar_flat = batch['radar_obs'].reshape(-1, NetParameters.RADAR_DIM)
                        velocity_flat = batch['velocity_obs'].reshape(-1, NetParameters.VELOCITY_DIM)
                        base_actions_flat = batch['base_actions'].reshape(-1, NetParameters.ACTION_DIM)
                        returns_flat = batch['returns'].reshape(-1)
                        values_flat = batch['values'].reshape(-1)
                        actions_flat = batch['actions'].reshape(-1, NetParameters.ACTION_DIM)
                        old_logp_flat = batch['old_log_probs'].reshape(-1)
                        mask_flat = batch['mask'].reshape(-1)
                        
                        train_args = {
                            'writer': global_summary,
                            'global_step': curr_steps,
                            'actor_obs': actor_flat,
                            'critic_obs': critic_flat,
                            'radar_obs': radar_flat,
                            'velocity_obs': velocity_flat,
                            'base_actions': base_actions_flat,
                            'returns': returns_flat,
                            'values': values_flat,
                            'actions': actions_flat,
                            'old_log_probs': old_logp_flat,
                            'mask': mask_flat,
                            'il_batch': None
                        }
                        
                        result = training_model.train(**train_args)
                        
                        if isinstance(result, dict):
                            epoch_loss_buffer.append(result.get('losses', [])) # list of losses
                            
            # Logging
            if curr_steps - last_train_log_t >= TrainingParameters.LOG_EPOCH_STEPS:
                last_train_log_t = curr_steps
                train_stats = compute_performance_stats(epoch_perf_buffer) if epoch_perf_buffer['per_r'] else {}
                
                log_str = f"[TRAIN] step={curr_steps:,} | ep={curr_episodes:,} | LR={new_lr:.2e}"
                if train_stats:
                    log_str += f" | Rew={train_stats.get('per_r_mean', 0):.2f} | Win={train_stats.get('win_mean', 0)*100:.1f}%"
                print(log_str)
                
                if global_summary:
                     write_to_tensorboard(global_summary, curr_steps, performance_dict=epoch_perf_buffer, 
                                         mb_loss=epoch_loss_buffer, imitation_loss=[0.0, 0.0], q_loss=0.0, evaluate=False)
                     global_summary.add_scalar('Train/LR', new_lr, curr_steps)

                epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
                epoch_loss_buffer = []

            # Periodic Evaluation
            if curr_steps - last_eval_t >= RecordingParameters.EVAL_INTERVAL:
                last_eval_t = curr_steps
                print(f"[EVAL] Running evaluation at step={curr_steps}...")
                
                model_weights = training_model.get_weights()
                eval_results = run_evaluation(envs, model_weights, opponent_weights, RecordingParameters.EVAL_EPISODES)
                eval_stats = compute_performance_stats(eval_results)
                
                eval_reward = eval_stats.get('per_r_mean', 0.0)
                eval_win_rate = eval_stats.get('win_mean', 0.0) * 100
                
                print(f"[EVAL] step={curr_steps:,} | Reward={eval_reward:.2f} | WinRate={eval_win_rate:.1f}%")
                
                if global_summary:
                    global_summary.add_scalar('Eval/Reward', eval_reward, curr_steps)
                    global_summary.add_scalar('Eval/WinRate', eval_win_rate, curr_steps)
                    global_summary.add_scalar('Eval/EpisodeLen', eval_stats.get('per_episode_len_mean', 0.0), curr_steps)
                
                # Best Model Saving (based on evaluation reward)
                if eval_reward > best_perf:
                    best_perf = eval_reward
                    best_path = os.path.join(MODEL_PATH, 'best_model')
                    os.makedirs(best_path, exist_ok=True)
                    torch.save({
                        'model': training_model.get_weights(),
                        'step': curr_steps,
                        'episode': curr_episodes,
                        'eval_reward': eval_reward,
                        'eval_win_rate': eval_win_rate
                    }, os.path.join(best_path, 'checkpoint.pth'))
                    print(f"[BEST] New best model saved! Reward={eval_reward:.2f} WinRate={eval_win_rate:.1f}%")

            # Save Latest Model
            if curr_steps - last_save_t >= RecordingParameters.SAVE_INTERVAL:
                last_save_t = curr_steps
                latest_path = os.path.join(MODEL_PATH, 'latest_model')
                os.makedirs(latest_path, exist_ok=True)
                torch.save({
                    'model': training_model.get_weights(), 
                    'step': curr_steps, 
                    'episode': curr_episodes
                }, os.path.join(latest_path, 'checkpoint.pth'))
                print(f"[SAVE] Latest model saved at step={curr_steps}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        final_path = os.path.join(MODEL_PATH, 'final_model')
        os.makedirs(final_path, exist_ok=True)
        torch.save({'model': training_model.get_weights(), 'step': curr_steps, 'episode': curr_episodes},
                   os.path.join(final_path, 'checkpoint.pth'))
        print(f"Final model saved to {final_path}")
        if global_summary:
            global_summary.close()
        ray.shutdown()

if __name__ == "__main__":
    main()
