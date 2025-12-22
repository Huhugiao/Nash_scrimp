import os
import sys
import math
import random
import numpy as np
import torch
import ray
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import setproctitle
except Exception:
    setproctitle = None

from mlp.alg_parameters_mlp import SetupParameters, NetParameters
from mlp.model_mlp import Model as BaseModel
from mlp.util_mlp import set_global_seeds, make_gif, write_to_tensorboard
from mlp.policymanager_mlp import PolicyManager

from residual.alg_parameters_residual import (
    ResidualTrainingParameters as TP,
    ResidualRecordingParameters as RP,
    ResidualRLParameters,
)
from residual.model_residual import ResidualModel
from residual.runner_residual import ResidualRunner
from residual.nets_residual import ResidualPolicyNetwork
from cbf_controller import CBFTracker
from env import TrackingEnv
from map_config import EnvParameters


def _parse_eval_obs(obs_result):
    """Split tracker/target observations."""
    obs = obs_result
    if isinstance(obs_result, tuple):
        if len(obs_result) == 2 and isinstance(obs_result[1], dict):
            obs = obs_result[0]
        else:
            obs = obs_result
    if isinstance(obs, tuple) and len(obs) == 2:
        tracker_obs, target_obs = obs
    else:
        tracker_obs = obs
        target_obs = obs
    return np.asarray(tracker_obs, dtype=np.float32), np.asarray(target_obs, dtype=np.float32)


def residual_evaluate(residual_model, device, num_episodes=None):
    """
    评估残差策略：CBF + Residual
    """
    num_episodes = num_episodes or RP.EVAL_EPISODES
    eval_env = TrackingEnv(safety_layer_enabled=False)
    cbf_base = CBFTracker(env=eval_env)
    eval_pm = PolicyManager() if TP.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
    
    performance = {'per_r': [], 'per_episode_len': [], 'win': []}
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs_result = eval_env.reset()
            tracker_obs, target_obs = _parse_eval_obs(obs_result)
            done = False
            ep_r = 0.0
            ep_len = 0
            info = {}
            
            current_policy = None
            if eval_pm:
                current_policy, _ = eval_pm.sample_policy("target")
                eval_pm.reset()
            
            while not done and ep_len < EnvParameters.EPISODE_LEN:
                # CBF base action with privileged state for full QP
                privileged_state = eval_env.get_privileged_state() if hasattr(eval_env, 'get_privileged_state') else None
                cbf_action = cbf_base.get_action(tracker_obs, privileged_state=privileged_state)
                base_action = np.asarray(cbf_action, dtype=np.float32)
                
                # Residual action (deterministic for eval)
                obs_tensor = torch.from_numpy(tracker_obs).float().to(device).unsqueeze(0)
                raw_mean, _ = residual_model.actor.forward_raw(obs_tensor)
                residual_action = torch.tanh(raw_mean) * residual_model.actor.residual_scale
                residual_action_np = residual_action.cpu().numpy()[0]
                
                # Fuse with proper scaling
                import map_config
                max_turn = float(getattr(map_config, 'tracker_max_turn_deg', 5.0))
                residual_scaled = np.array([
                    residual_action_np[0] * max_turn,
                    residual_action_np[1]
                ], dtype=np.float32)
                fused = base_action + residual_scaled
                final_action = np.array([
                    np.clip(fused[0], -max_turn, max_turn),
                    np.clip(fused[1], 0.0, 1.0)
                ], dtype=np.float32)
                
                # Target action
                if eval_pm and current_policy:
                    target_action = eval_pm.get_action(current_policy, target_obs)
                else:
                    target_action = np.zeros(2, dtype=np.float32)
                
                obs_result, reward, terminated, truncated, info = eval_env.step((final_action, target_action))
                done = terminated or truncated
                ep_r += float(reward)
                ep_len += 1
                tracker_obs, target_obs = _parse_eval_obs(obs_result)
            
            performance['per_r'].append(ep_r)
            performance['per_episode_len'].append(ep_len)
            performance['win'].append(1 if isinstance(info, dict) and info.get('reason') == 'tracker_caught_target' else 0)
    
    stats = compute_performance_stats(performance)
    return stats.get('per_r_mean', 0.0), stats.get('win_mean', 0.0)


def record_residual_gif(residual_model, device, gif_path):
    """
    录制残差策略 GIF
    """
    eval_env = TrackingEnv(safety_layer_enabled=False)
    cbf_base = CBFTracker(env=eval_env)
    eval_pm = PolicyManager() if TP.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
    
    frames = []
    obs_result = eval_env.reset()
    tracker_obs, target_obs = _parse_eval_obs(obs_result)
    done = False
    ep_len = 0
    
    current_policy = None
    if eval_pm:
        current_policy, _ = eval_pm.sample_policy("target")
        eval_pm.reset()
    
    try:
        frame = eval_env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    except Exception:
        pass
    
    with torch.no_grad():
        while not done and ep_len < EnvParameters.EPISODE_LEN:
            privileged_state = eval_env.get_privileged_state() if hasattr(eval_env, 'get_privileged_state') else None
            cbf_action = cbf_base.get_action(tracker_obs, privileged_state=privileged_state)
            base_action = np.asarray(cbf_action, dtype=np.float32)
            
            obs_tensor = torch.from_numpy(tracker_obs).float().to(device).unsqueeze(0)
            raw_mean, _ = residual_model.actor.forward_raw(obs_tensor)
            residual_action = torch.tanh(raw_mean) * residual_model.actor.residual_scale
            residual_action_np = residual_action.cpu().numpy()[0]
            
            # Fuse with proper scaling
            import map_config
            max_turn = float(getattr(map_config, 'tracker_max_turn_deg', 5.0))
            residual_scaled = np.array([
                residual_action_np[0] * max_turn,
                residual_action_np[1]
            ], dtype=np.float32)
            fused = base_action + residual_scaled
            final_action = np.array([
                np.clip(fused[0], -max_turn, max_turn),
                np.clip(fused[1], 0.0, 1.0)
            ], dtype=np.float32)
            
            if eval_pm and current_policy:
                target_action = eval_pm.get_action(current_policy, target_obs)
            else:
                target_action = np.zeros(2, dtype=np.float32)
            
            obs_result, _, terminated, truncated, _ = eval_env.step((final_action, target_action))
            done = terminated or truncated
            ep_len += 1
            tracker_obs, target_obs = _parse_eval_obs(obs_result)
            
            try:
                frame = eval_env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
            except Exception:
                break
    
    if len(frames) > 1:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        make_gif(frames, gif_path, fps=EnvParameters.RENDER_FPS)

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
    final_lr = TP.LR_FINAL
    initial_lr = TP.lr
    max_steps = TP.N_MAX_STEPS
    progress = min(max(current_step / max_steps, 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (initial_lr - final_lr) * weight

def get_scheduled_n_epochs(current_step):
    initial = TP.N_EPOCHS_INITIAL
    final = TP.N_EPOCHS_FINAL
    max_steps = TP.N_MAX_STEPS
    progress = min(max(current_step / max_steps, 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    n_epochs = final + (initial - final) * weight
    return max(int(round(n_epochs)), 1)

def main():
    if not ResidualRLParameters.ENABLED:
        print("Residual RL is disabled in parameters.")
        return

    if not ray.is_initialized():
        ray.init(num_gpus=SetupParameters.NUM_GPU)

    set_global_seeds(SetupParameters.SEED)
    if setproctitle:
        setproctitle.setproctitle(f"AvoidMaker_{ResidualRLParameters.EXPERIMENT_NAME}")

    MODEL_PATH = RP.MODEL_PATH
    SUMMARY_PATH = RP.SUMMARY_PATH
    os.makedirs(MODEL_PATH, exist_ok=True)

    global_summary = SummaryWriter(log_dir=SUMMARY_PATH)

    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')

    training_model = ResidualModel(global_device, True)

    opponent_model, opponent_weights = None, None
    if TP.OPPONENT_TYPE == "policy":
        opponent_model = BaseModel(global_device, False)
        if SetupParameters.PRETRAINED_TARGET_PATH:
            opp_dict = torch.load(SetupParameters.PRETRAINED_TARGET_PATH, map_location='cpu')
            opponent_model.set_weights(opp_dict['model'])
        opponent_weights = opponent_model.get_weights()

    global_pm = PolicyManager() if TP.OPPONENT_TYPE in {"random", "random_nonexpert"} and getattr(TP, 'ADAPTIVE_SAMPLING', False) else None

    envs = [ResidualRunner.remote(i + 1) for i in range(TP.N_ENVS)]

    curr_steps = 0
    curr_episodes = 0
    best_perf = -1e9

    epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
    epoch_loss_buffer = []

    last_train_log_t = 0
    last_save_t = 0
    last_eval_t = 0
    last_gif_t = 0
    
    GIFS_PATH = os.path.join(MODEL_PATH, 'gifs')
    os.makedirs(GIFS_PATH, exist_ok=True)

    try:
        print(f"Starting Residual RL Training: {ResidualRLParameters.EXPERIMENT_NAME}")
        print(f"Max Steps: {ResidualRLParameters.RESIDUAL_MAX_STEPS}")

        while curr_steps < ResidualRLParameters.RESIDUAL_MAX_STEPS:
            new_lr = get_scheduled_lr(curr_steps)
            training_model.update_learning_rate(new_lr)

            model_weights = training_model.get_weights()
            pm_state = {k: list(v) for k, v in global_pm.win_history.items()} if global_pm else None

            jobs = [
                envs[i].run.remote(model_weights, opponent_weights, curr_steps, pm_state)
                for i in range(TP.N_ENVS)
            ]
            results = ray.get(jobs)

            all_segments = []
            total_new_episodes = 0

            for result in results:
                rl_data = extract_rl_data_from_rollout(result)
                all_segments.extend(build_segments_from_rollout(rl_data, NetParameters.CONTEXT_WINDOW))

                perf = result['performance']
                epoch_perf_buffer['per_r'].extend(perf['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(perf['per_episode_len'])
                epoch_perf_buffer['win'].extend(perf['win'])
                total_new_episodes += result['episodes']

                if global_pm and result['policy_manager_state']:
                    from collections import deque
                    for name, history in result['policy_manager_state'].items():
                        global_pm.win_history[name] = deque(history, maxlen=getattr(TP, 'ADAPTIVE_SAMPLING_WINDOW', 1))

            curr_steps += TP.N_ENVS * TP.N_STEPS
            curr_episodes += total_new_episodes

            if all_segments:
                random.shuffle(all_segments)
                n_epochs = get_scheduled_n_epochs(curr_steps)

                for _ in range(n_epochs):
                    for mb_start in range(0, len(all_segments), TP.MINIBATCH_SIZE):
                        mb_end = min(mb_start + TP.MINIBATCH_SIZE, len(all_segments))
                        batch_segments = all_segments[mb_start:mb_end]
                        if not batch_segments:
                            continue

                        batch = collate_segments(batch_segments)

                        actor_flat = batch['actor_obs'].reshape(-1, NetParameters.ACTOR_RAW_LEN)
                        critic_flat = batch['critic_obs'].reshape(-1, NetParameters.CRITIC_RAW_LEN)
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
                            'returns': returns_flat,
                            'values': values_flat,
                            'actions': actions_flat,
                            'old_log_probs': old_logp_flat,
                            'mask': mask_flat,
                            'il_batch': None,
                        }

                        result = training_model.train(**train_args)
                        if isinstance(result, dict):
                            epoch_loss_buffer.append(result.get('losses', []))

            if curr_steps - last_train_log_t >= TP.LOG_EPOCH_STEPS:
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

            if curr_steps - last_save_t >= RP.SAVE_INTERVAL:
                last_save_t = curr_steps
                latest_path = os.path.join(MODEL_PATH, 'latest_model')
                os.makedirs(latest_path, exist_ok=True)
                torch.save({
                    'model': training_model.get_weights(),
                    'step': curr_steps,
                    'episode': curr_episodes
                }, os.path.join(latest_path, 'checkpoint.pth'))
                print(f"[SAVE] Latest model saved at step={curr_steps}")

            # Evaluation
            if curr_steps - last_eval_t >= RP.EVAL_INTERVAL:
                last_eval_t = curr_steps
                eval_reward, eval_win = residual_evaluate(
                    training_model.network, global_device, RP.EVAL_EPISODES
                )
                print(f"[EVAL] step={curr_steps:,} | Rew={eval_reward:.2f} | Win={eval_win*100:.1f}%")
                
                if global_summary:
                    global_summary.add_scalar('Eval/Reward', eval_reward, curr_steps)
                    global_summary.add_scalar('Eval/WinRate', eval_win, curr_steps)
                
                # Save best model
                if eval_win > best_perf:
                    best_perf = eval_win
                    best_path = os.path.join(MODEL_PATH, 'best_model')
                    os.makedirs(best_path, exist_ok=True)
                    torch.save({
                        'model': training_model.get_weights(),
                        'step': curr_steps,
                        'episode': curr_episodes,
                        'win_rate': eval_win
                    }, os.path.join(best_path, 'checkpoint.pth'))
                    print(f"[BEST] New best model! Win={eval_win*100:.1f}%")

            # GIF Recording
            if curr_steps - last_gif_t >= RP.GIF_INTERVAL:
                last_gif_t = curr_steps
                gif_path = os.path.join(GIFS_PATH, f'step_{curr_steps}.gif')
                record_residual_gif(training_model.network, global_device, gif_path)

    except KeyboardInterrupt:
        print("Training interrupted")
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
