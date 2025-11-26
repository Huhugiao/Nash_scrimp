import os, sys
# 新增：防止 Ray Worker 中 cvxpy/numpy 的 OpenMP 线程死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import os.path as osp
import math
import numpy as np
import torch
import ray
import random
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import setproctitle
except Exception:
    setproctitle = None

from torch.utils.tensorboard import SummaryWriter

from map_config import EnvParameters
from mha.alg_parameters_mha import *
from env import TrackingEnv
from mha.model_mha import Model
from mha.runner_mha import Runner
from util import set_global_seeds, make_gif, get_opponent_id_one_hot
from rule_policies import TRACKER_POLICY_REGISTRY, TARGET_POLICY_REGISTRY
from policymanager import PolicyManager

IL_INITIAL_PROB = 0.8
IL_FINAL_PROB = 0.1
IL_DECAY_STEPS = 1e7

PURE_RL_SWITCH = 0
if PURE_RL_SWITCH:
    IL_INITIAL_PROB = 0
    IL_FINAL_PROB = 0
    IL_DECAY_STEPS = 1

if not ray.is_initialized():
    ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to SCRIMP with MHA (Transformer) - Training Tracker!\n")
print(f"Opponent type: {TrainingParameters.OPPONENT_TYPE}")

if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
    available = sorted(list(TARGET_POLICY_REGISTRY.keys()))
    weights = TrainingParameters.RANDOM_OPPONENT_WEIGHTS.get("target", {})
    
    print(f"\nMulti-Opponent Configuration:")
    print(f"  Available target policies: {', '.join(available)}")
    print(f"  Sampling weights:")
    for policy_name in available:
        weight = weights.get(policy_name, 1.0)
        print(f"    - {policy_name}: {weight}")
    
    if TrainingParameters.ADAPTIVE_SAMPLING:
        print(f"  Adaptive sampling: ENABLED")
    else:
        print(f"  Adaptive sampling: DISABLED")

print(f"IL type: {getattr(TrainingParameters, 'IL_TYPE', 'expert')}")
print(f"IL probability will cosine anneal from {IL_INITIAL_PROB*100:.1f}% to {IL_FINAL_PROB*100:.1f}% over {IL_DECAY_STEPS} steps")
if getattr(TrainingParameters, 'IL_TYPE', 'expert') == "expert":
    print("Imitation teacher: CBF tracker policy")

def_attr = lambda name, default: getattr(RecordingParameters, name, default)
SUMMARY_PATH = def_attr('SUMMARY_PATH', f'./runs/TrackingEnv/{RecordingParameters.EXPERIMENT_NAME}{RecordingParameters.TIME}')
MODEL_PATH = def_attr('MODEL_PATH', f'./models/TrackingEnv/{RecordingParameters.EXPERIMENT_NAME}{RecordingParameters.TIME}')
GIFS_PATH = def_attr('GIFS_PATH', osp.join(MODEL_PATH, 'gifs'))
EVAL_INTERVAL = int(def_attr('EVAL_INTERVAL', 20000))
SAVE_INTERVAL = int(def_attr('SAVE_INTERVAL', 5e5))
BEST_INTERVAL = int(def_attr('BEST_INTERVAL', 0))
EVAL_EPISODES = int(def_attr('EVAL_EPISODES', 8))

def get_cosine_annealing_il_prob(current_step):
    if current_step >= IL_DECAY_STEPS:
        return IL_FINAL_PROB
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / IL_DECAY_STEPS))
    return IL_FINAL_PROB + (IL_INITIAL_PROB - IL_FINAL_PROB) * cosine_decay

def get_scheduled_lr(current_step):
    final_lr = getattr(TrainingParameters, 'LR_FINAL', TrainingParameters.lr)
    schedule = getattr(TrainingParameters, 'LR_SCHEDULE', 'cosine')
    progress = min(max(current_step / TrainingParameters.N_MAX_STEPS, 0.0), 1.0)
    if schedule == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        weight = 1.0 - progress
    return final_lr + (TrainingParameters.lr - final_lr) * weight

def build_segments_from_rollout(rollout, window_size):
    segments = []
    total_steps = rollout['actor_obs'].shape[0]
    
    current_start = 0
    for i in range(total_steps):
        # If episode starts at i (and it's not the very first step of the buffer), flush previous segment
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
        
        seg = {
            'actor_obs': rollout['actor_obs'][cursor:seg_end],
            'critic_obs': rollout['critic_obs'][cursor:seg_end],
            'returns': rollout['returns'][cursor:seg_end],
            'values': rollout['values'][cursor:seg_end],
            'actions': rollout['actions'][cursor:seg_end],
            'old_log_probs': rollout['logp'][cursor:seg_end],
            'mask': np.ones(seg_len, dtype=np.float32)
        }
        segments.append(seg)
        cursor += seg_len

def collate_segments(batch_segments):
    batch_size = len(batch_segments)
    if batch_size == 0:
        raise ValueError("Empty segment batch passed to collate.")
    max_len = max(len(s['actor_obs']) for s in batch_segments)
    
    actor_obs = np.zeros((batch_size, max_len, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32)
    critic_obs = np.zeros((batch_size, max_len, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32)
    returns = np.zeros((batch_size, max_len), dtype=np.float32)
    values = np.zeros((batch_size, max_len), dtype=np.float32)
    actions = np.zeros((batch_size, max_len, NetParameters.ACTION_DIM), dtype=np.float32)
    old_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)
    
    for i, seg in enumerate(batch_segments):
        l = len(seg['actor_obs'])
        actor_obs[i, :l] = seg['actor_obs']
        critic_obs[i, :l] = seg['critic_obs']
        returns[i, :l] = seg['returns']
        values[i, :l] = seg['values']
        actions[i, :l] = seg['actions']
        old_log_probs[i, :l] = seg['old_log_probs']
        mask[i, :l] = seg['mask']
        
    return {
        'actor_obs': actor_obs,
        'critic_obs': critic_obs,
        'returns': returns,
        'values': values,
        'actions': actions,
        'old_log_probs': old_log_probs,
        'mask': mask
    }

def compute_performance_stats(performance_dict):
    stats = {}
    for key, values in performance_dict.items():
        if len(values) > 0:
            stats[f'{key}_mean'] = float(np.nanmean(values))
            stats[f'{key}_std'] = float(np.nanstd(values))
        else:
            stats[f'{key}_mean'] = 0.0
            stats[f'{key}_std'] = 0.0
    return stats

def format_train_log(curr_steps, curr_episodes, phase, il_prob,
                     train_stats, avg_losses, recent_window_stats, il_loss=None):
    parts = [
        f"[{phase}] step={curr_steps:,}",
        f"ep={curr_episodes:,}",
        f"ILp={il_prob*100:5.1f}%"
    ]
    if train_stats:
        r_mean = train_stats.get('per_r_mean', 0.0)
        r_std = train_stats.get('per_r_std', 0.0)
        l_mean = train_stats.get('per_episode_len_mean', 0.0)
        l_std = train_stats.get('per_episode_len_std', 0.0)
        parts.append(f"Rew={r_mean:.2f}±{r_std:.2f}")
        parts.append(f"Len={l_mean:.2f}±{l_std:.2f}")

    return " | ".join(parts)

def main():
    model_dict = None
    if def_attr('RETRAIN', False):
        restore_path = def_attr('RESTORE_DIR', None)
        if restore_path:
            model_path = restore_path + "/tracker_net_checkpoint.pkl"
            if os.path.exists(model_path):
                model_dict = torch.load(model_path, map_location='cpu')
                
                # Fix: Inherit IL probability from the loaded step
                prev_step = model_dict.get("step", 0)
                global IL_INITIAL_PROB
                if prev_step >= IL_DECAY_STEPS:
                    IL_INITIAL_PROB = IL_FINAL_PROB
                else:
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * prev_step / IL_DECAY_STEPS))
                    IL_INITIAL_PROB = IL_FINAL_PROB + (IL_INITIAL_PROB - IL_FINAL_PROB) * cosine_decay
                print(f"RETRAIN=True: Updated IL_INITIAL_PROB to {IL_INITIAL_PROB:.4f} based on loaded step {prev_step}")

    global_summary = None
    if def_attr('TENSORBOARD', True):
        tb_dir = osp.join(MODEL_PATH, "tfevents")
        os.makedirs(tb_dir, exist_ok=True)
        global_summary = SummaryWriter(tb_dir)
        print(f'Launching tensorboard at: {tb_dir}\n')

    if setproctitle is not None:
        setproctitle.setproctitle(
            RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + getattr(RecordingParameters, 'ENTITY', 'user'))
    set_global_seeds(SetupParameters.SEED)

    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    training_model = Model(global_device, True)

    if model_dict is not None:
        training_model.network.load_state_dict(model_dict['model'])
        if 'optimizer' in model_dict and training_model.net_optimizer:
            training_model.net_optimizer.load_state_dict(model_dict['optimizer'])

    opponent_model = None
    opponent_weights = None
    if TrainingParameters.OPPONENT_TYPE == "policy":
        opponent_model = Model(global_device, False)
        opp_path = SetupParameters.PRETRAINED_TARGET_PATH
        if opp_path and os.path.exists(opp_path):
            opponent_dict = torch.load(opp_path, map_location='cpu')
            opponent_model.network.load_state_dict(opponent_dict['model'])
            opponent_weights = opponent_model.get_weights()

    global_policy_manager = None
    if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} and TrainingParameters.ADAPTIVE_SAMPLING:
        global_policy_manager = PolicyManager()

    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    eval_env = TrackingEnv()

    curr_steps = int(model_dict.get("step", 0)) if model_dict is not None else 0
    curr_episodes = int(model_dict.get("episode", 0)) if model_dict is not None else 0
    best_perf = float(model_dict.get("reward", -1e9)) if model_dict is not None else -1e9
    avg_perf_for_best = best_perf

    RECENT_WINDOW_ITERS = 20
    recent_rewards = deque(maxlen=RECENT_WINDOW_ITERS)
    recent_wins = deque(maxlen=RECENT_WINDOW_ITERS)
    il_segments = deque(maxlen=20000) # Buffer for IL data

    last_test_t = -int(EVAL_INTERVAL) - 1
    last_model_t = -int(SAVE_INTERVAL) - 1
    last_best_t = -int(BEST_INTERVAL) - 1

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(GIFS_PATH, exist_ok=True)

    last_train_log_t = 0
    epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
    epoch_loss_buffer = []
    epoch_il_loss_buffer = []

    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            il_prob = get_cosine_annealing_il_prob(curr_steps)
            scheduled_lr = get_scheduled_lr(curr_steps)
            training_model.update_learning_rate(scheduled_lr)
            if global_summary:
                global_summary.add_scalar('Train/lr', scheduled_lr, curr_steps)
            do_il = (np.random.rand() < il_prob)

            weights = training_model.get_weights()
            performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}

            if do_il:
                # ---------------- IL ----------------
                jobs = [e.imitation.remote(weights, opponent_weights, curr_steps) for e in envs]
                results = ray.get(jobs)
                
                total_il_episodes = 0
                mb_loss = []
                
                # Process IL results
                for r in results:
                    # Store for RL projection
                    n = len(r['actor_obs'])
                    dummy_rollout = {
                        'actor_obs': r['actor_obs'],
                        'critic_obs': r['critic_obs'],
                        'actions': r['actions'],
                        'dones': r['dones'],
                        'episode_starts': r['episode_starts'],
                        'returns': np.zeros(n, dtype=np.float32),
                        'values': np.zeros(n, dtype=np.float32),
                        'logp': np.zeros(n, dtype=np.float32)
                    }
                    new_segs = build_segments_from_rollout(dummy_rollout, TrainingParameters.TBPTT_STEPS)
                    il_segments.extend(new_segs)
                    
                    perf = r['performance']
                    performance_dict['per_r'].extend(perf.get('per_r', []))
                    performance_dict['per_episode_len'].extend(perf.get('per_episode_len', []))
                    performance_dict['win'].extend(perf.get('win', []))
                    total_il_episodes += r.get('episodes', 0)

                # Train on IL data immediately
                if len(il_segments) > 0:
                    for _ in range(3):
                        k = min(len(il_segments), TrainingParameters.MINIBATCH_SIZE)
                        il_sample = random.sample(il_segments, k)
                        il_batch_raw = collate_segments(il_sample)
                        loss_result = training_model.imitation_train(
                            il_batch_raw['actor_obs'], il_batch_raw['critic_obs'], il_batch_raw['actions']
                        )
                        mb_loss.append(loss_result)

                if global_summary and mb_loss:
                    avg_il_loss = np.nanmean([loss[0] for loss in mb_loss])
                    global_summary.add_scalar('Train/imitation_loss', avg_il_loss, curr_steps)
                    epoch_il_loss_buffer.append(avg_il_loss)

                curr_steps += int(TrainingParameters.N_ENVS * TrainingParameters.N_STEPS)
                curr_episodes += total_il_episodes

                train_stats = compute_performance_stats(performance_dict)
                if len(performance_dict['per_r']) > 0:
                    recent_rewards.append(train_stats['per_r_mean'])
                if len(performance_dict.get('win', [])) > 0:
                    recent_wins.append(train_stats['win_mean'])

                epoch_perf_buffer['per_r'].extend(performance_dict['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(performance_dict['per_episode_len'])
                epoch_perf_buffer['win'].extend(performance_dict['win'])

            else:
                # ---------------- RL ----------------
                pm_state = global_policy_manager.win_history if global_policy_manager else None
                if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
                    jobs = [e.run.remote(weights, opponent_weights, curr_steps, pm_state) for e in envs]
                else:
                    jobs = [e.run.remote(weights, opponent_weights, curr_steps, None) for e in envs]
                results = ray.get(jobs)

                if global_policy_manager:
                    all_pm_states = [r[4] for r in results if r[4] is not None]
                    for pm_state in all_pm_states:
                        for policy_name, history in pm_state.items():
                            if policy_name in global_policy_manager.win_history:
                                global_policy_manager.win_history[policy_name].extend(list(history))

                segments = []
                steps_batch = 0
                episodes_batch = 0
                for r in results:
                    rollout = r[0]
                    segments.extend(build_segments_from_rollout(rollout, TrainingParameters.TBPTT_STEPS))
                    steps_batch += r[1]
                    episodes_batch += r[2]
                    perf = r[3]
                    performance_dict['per_r'].extend(perf.get('per_r', []))
                    performance_dict['per_episode_len'].extend(perf.get('per_episode_len', []))
                    performance_dict['win'].extend(perf.get('win', []))

                if not segments:
                    curr_steps += steps_batch
                    curr_episodes += episodes_batch
                    continue

                if global_summary:
                    adv_concat = np.concatenate([seg['returns'] - seg['values'] for seg in segments])
                    if adv_concat.size > 0:
                        global_summary.add_scalar('Train/adv_mean', float(np.mean(adv_concat)), curr_steps)

                mb_loss = []
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(segments)
                    for start in range(0, len(segments), TrainingParameters.MINIBATCH_SIZE):
                        batch_segs = segments[start:start + TrainingParameters.MINIBATCH_SIZE]
                        batch = collate_segments(batch_segs)
                        
                        # Prepare IL batch for gradient projection
                        il_batch = None
                        if len(il_segments) > 0:
                            k = min(len(il_segments), TrainingParameters.MINIBATCH_SIZE)
                            il_sample = random.sample(il_segments, k)
                            il_batch_raw = collate_segments(il_sample)
                            il_batch = {
                                'actor_obs': il_batch_raw['actor_obs'],
                                'critic_obs': il_batch_raw['critic_obs'],
                                'actions': il_batch_raw['actions'],
                                'mask': il_batch_raw['mask']
                            }
                        
                        loss_result = training_model.train(
                            batch['actor_obs'], batch['critic_obs'], batch['returns'],
                            batch['values'], batch['actions'], batch['old_log_probs'],
                            batch['mask'], il_batch=il_batch
                        )
                        if not np.all(np.isfinite(loss_result)):
                            continue
                        mb_loss.append(loss_result)

                valid_losses = [loss for loss in mb_loss if np.all(np.isfinite(loss))]
                if global_summary and valid_losses:
                    avg_losses = np.nanmean(valid_losses, axis=0)
                    names = RecordingParameters.LOSS_NAME
                    for idx, val in enumerate(avg_losses):
                        if idx < len(names):
                            global_summary.add_scalar(f'Train/{names[idx]}', val, curr_steps)

                curr_steps += steps_batch
                curr_episodes += episodes_batch

                train_stats = compute_performance_stats(performance_dict)
                avg_perf_for_best = train_stats.get('per_r_mean', -1e9)

                if len(performance_dict['per_r']) > 0:
                    recent_rewards.append(train_stats['per_r_mean'])
                if len(performance_dict.get('win', [])) > 0:
                    recent_wins.append(train_stats['win_mean'])

                epoch_perf_buffer['per_r'].extend(performance_dict['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(performance_dict['per_episode_len'])
                epoch_perf_buffer['win'].extend(performance_dict['win'])
                if valid_losses:
                    epoch_loss_buffer.extend(valid_losses)

            # ------------ Common Logic ------------
            if global_summary and 'train_stats' in locals():
                for key, value in train_stats.items():
                    global_summary.add_scalar(f'Train/{key}', value, curr_steps)

            if curr_steps - last_train_log_t >= TrainingParameters.LOG_EPOCH_STEPS:
                last_train_log_t = curr_steps
                epoch_stats = compute_performance_stats(epoch_perf_buffer)
                avg_epoch_losses = None
                if epoch_loss_buffer:
                    avg_epoch_losses = np.nanmean(epoch_loss_buffer, axis=0)
                
                avg_epoch_il_loss = None
                if epoch_il_loss_buffer:
                    avg_epoch_il_loss = float(np.mean(epoch_il_loss_buffer))

                recent_window_stats = None
                if len(recent_rewards) > 0:
                    recent_window_stats = {
                        'window': len(recent_rewards),
                        'per_r_mean': float(np.mean(recent_rewards)),
                        'win_mean': float(np.mean(recent_wins)) if recent_wins else None,
                    }

                phase = "IL" if il_prob > 0.5 else "RL"
                print(format_train_log(
                    curr_steps, curr_episodes, phase=phase,
                    il_prob=il_prob,
                    train_stats=epoch_stats,
                    avg_losses=avg_epoch_losses,
                    recent_window_stats=recent_window_stats,
                    il_loss=avg_epoch_il_loss
                ))

                epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
                epoch_loss_buffer = []
                epoch_il_loss_buffer = []

            if curr_steps - last_model_t >= SAVE_INTERVAL:
                last_model_t = curr_steps
                model_path = osp.join(MODEL_PATH, 'latest')
                os.makedirs(model_path, exist_ok=True)
                save_path = model_path + "/tracker_net_checkpoint.pkl"
                checkpoint = {"model": training_model.network.state_dict(),
                              "optimizer": training_model.net_optimizer.state_dict(),
                              "step": curr_steps, "episode": curr_episodes, "reward": avg_perf_for_best}
                torch.save(checkpoint, save_path)

            if 'avg_perf_for_best' in locals() and \
               avg_perf_for_best > best_perf and (curr_steps - last_best_t >= BEST_INTERVAL):
                best_perf = avg_perf_for_best
                last_best_t = curr_steps
                model_path = osp.join(MODEL_PATH, 'best_model')
                os.makedirs(model_path, exist_ok=True)
                save_path = model_path + "/tracker_net_checkpoint.pkl"
                checkpoint = {"model": training_model.network.state_dict(),
                              "optimizer": training_model.net_optimizer.state_dict(),
                              "step": curr_steps, "episode": curr_episodes, "reward": best_perf}
                torch.save(checkpoint, save_path)

            if curr_steps - last_test_t >= EVAL_INTERVAL:
                last_test_t = curr_steps
                eval_stats = evaluate_single_agent(eval_env, training_model, opponent_model, global_device)
                
                r_mean = eval_stats.get('per_r_mean', 0.0)
                r_std = eval_stats.get('per_r_std', 0.0)
                l_mean = eval_stats.get('per_episode_len_mean', 0.0)
                l_std = eval_stats.get('per_episode_len_std', 0.0)
                
                print(f"[EVAL] step={curr_steps:,} | Rew={r_mean:.2f}±{r_std:.2f} | Len={l_mean:.2f}±{l_std:.2f}")
                
                if global_summary:
                    for key, value in eval_stats.items():
                        global_summary.add_scalar(f'Eval/{key}', value, curr_steps)

                generate_one_episode_gif(eval_env, training_model, opponent_model, global_device, curr_steps)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        print('Saving Final Model!')
        model_path = MODEL_PATH + '/final'
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        save_path = model_path + "/tracker_net_checkpoint.pkl"
        checkpoint = {"model": training_model.network.state_dict(),
                      "optimizer": training_model.net_optimizer.state_dict(),
                      "step": curr_steps, "episode": curr_episodes, "reward": best_perf}
        torch.save(checkpoint, save_path)
        print(f"Final model saved to {save_path}")
        ray.shutdown()

def get_opponent_action_for_eval(target_obs, opponent_type, opponent_model,
                                  policy_manager, current_policy_name, current_policy_id, opponent_history=None,
                                  privileged_state=None, tracker_obs=None):
    if opponent_type == "policy":
        if opponent_model is None:
            raise RuntimeError("OPPONENT_TYPE=policy but opponent_model is None")
        # Critic input for MHA: Just Target
        critic_obs = target_obs
        opp_action, _, new_history, _, _ = opponent_model.evaluate(target_obs, critic_obs, opponent_history, greedy=True)
        return opp_action, new_history
        
    elif opponent_type in {"random", "random_nonexpert"}:
        if policy_manager and current_policy_name:
            opp_pair = policy_manager.get_action(current_policy_name, target_obs, privileged_state)
        else:
            default_target = sorted(TARGET_POLICY_REGISTRY.keys())[0]
            policy_cls = TARGET_POLICY_REGISTRY.get(default_target)
            policy_obj = policy_cls()
            opp_pair = policy_obj.get_action(target_obs)
        return opp_pair, None
    else:
        raise ValueError(f"Unsupported OPPONENT_TYPE: {opponent_type}")

def evaluate_single_agent(eval_env, agent_model, opponent_model, device):
    eval_performance_dict = {'per_r': [], 'per_episode_len': [], 'win': []}
    eval_policy_manager = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None

    for _ in range(EVAL_EPISODES):
        obs_tuple = eval_env.reset()
        if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
            try:
                tracker_obs, target_obs = obs_tuple[0]
            except Exception:
                tracker_obs = obs_tuple[0]
                target_obs = obs_tuple[0]
        else:
            tracker_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            target_obs = tracker_obs
        
        done = False
        ep_r = 0.0
        ep_len = 0

        agent_history = None
        opponent_history = None

        current_policy_name, current_policy_id = (None, -1)
        if eval_policy_manager:
            current_policy_name, current_policy_id = eval_policy_manager.sample_policy("target")
            eval_policy_manager.reset()

        while not done and ep_len < EnvParameters.EPISODE_LEN:
            # Agent(tracker) action
            critic_obs_full = target_obs
            agent_action, _, agent_history, _, _ = agent_model.evaluate(tracker_obs, critic_obs_full, agent_history, greedy=True)

            # Opponent(target) action
            target_action, opponent_history = get_opponent_action_for_eval(
                target_obs, TrainingParameters.OPPONENT_TYPE,
                opponent_model, eval_policy_manager, current_policy_name, current_policy_id, opponent_history,
                tracker_obs=tracker_obs
            )

            obs_result, reward, terminated, truncated, info = eval_env.step((agent_action, target_action))
            done = terminated or truncated
            ep_r += float(reward)
            ep_len += 1
            
            if isinstance(obs_result, tuple) and len(obs_result) == 2:
                try:
                    tracker_obs, target_obs = obs_result
                except Exception:
                    tracker_obs = obs_result
                    target_obs = obs_result
            else:
                tracker_obs = obs_result
                target_obs = obs_result

        eval_performance_dict['per_r'].append(ep_r)
        eval_performance_dict['per_episode_len'].append(ep_len)
        win = 1 if info.get('reason') == 'tracker_caught_target' else 0
        eval_performance_dict['win'].append(win)

    return compute_performance_stats(eval_performance_dict)

def generate_one_episode_gif(eval_env, agent_model, opponent_model, device, curr_steps):
    episode_frames = []

    obs_tuple = eval_env.reset()
    if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2:
        try:
            tracker_obs, target_obs = obs_tuple[0]
        except Exception:
            tracker_obs = obs_tuple[0]
            target_obs = obs_tuple[0]
    else:
        tracker_obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        target_obs = tracker_obs
    
    done = False
    ep_len = 0

    agent_history = None
    opponent_history = None

    eval_policy_manager = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
    current_policy_name, current_policy_id = (None, -1)
    if eval_policy_manager:
        current_policy_name, current_policy_id = eval_policy_manager.sample_policy("target")
        eval_policy_manager.reset()

    while not done and ep_len < EnvParameters.EPISODE_LEN:
        frame = eval_env.render(mode='rgb_array')
        if frame is not None:
            episode_frames.append(frame)

        critic_obs_full = target_obs
        agent_action, _, agent_history, _, _ = agent_model.evaluate(tracker_obs, critic_obs_full, agent_history, greedy=True)

        target_action, opponent_history = get_opponent_action_for_eval(
            target_obs, TrainingParameters.OPPONENT_TYPE,
            opponent_model, eval_policy_manager, current_policy_name, current_policy_id, opponent_history,
            tracker_obs=tracker_obs
        )

        obs_result, reward, terminated, truncated, info = eval_env.step((agent_action, target_action))
        done = terminated or truncated
        ep_len += 1
        
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            try:
                tracker_obs, target_obs = obs_result
            except Exception:
                tracker_obs = obs_result
                target_obs = obs_result
        else:
            tracker_obs = obs_result
            target_obs = obs_result

    if len(episode_frames) > 0:
        gif_path = osp.join(GIFS_PATH, f"eval_{int(curr_steps)}.gif")
        os.makedirs(GIFS_PATH, exist_ok=True)
        make_gif(episode_frames, gif_path, fps=30)

if __name__ == "__main__":
    main()