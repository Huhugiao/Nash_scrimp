import os, sys
# 新增：防止 Ray Worker 中 cvxpy/numpy 的 OpenMP 线程死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import os.path as osp
import math
import numpy as np
import torch
import ray
from collections import deque   # <-- 新增

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import setproctitle
except Exception:
    setproctitle = None

from torch.utils.tensorboard import SummaryWriter

from map_config import EnvParameters
from lstm.alg_parameters import *  # 修改导入
from env import TrackingEnv
from lstm.model_lstm import Model
from lstm.runner_lstm import Runner
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

LSTM_HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 1

if not ray.is_initialized():
    ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to SCRIMP with LSTM - Training Tracker!\n")
print(f"Opponent type: {TrainingParameters.OPPONENT_TYPE}")
print(f"LSTM Configuration: Hidden Size={LSTM_HIDDEN_SIZE}, Layers={NUM_LSTM_LAYERS}")

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
        print(f"    - Window size: {TrainingParameters.ADAPTIVE_SAMPLING_WINDOW}")
        print(f"    - Min games: {TrainingParameters.ADAPTIVE_SAMPLING_MIN_GAMES}")
        print(f"    - Strength: {TrainingParameters.ADAPTIVE_SAMPLING_STRENGTH}")
    else:
        print(f"  Adaptive sampling: DISABLED")

print(f"IL type: {getattr(TrainingParameters, 'IL_TYPE', 'expert')}")
print(f"IL probability will cosine anneal from {IL_INITIAL_PROB*100:.1f}% to {IL_FINAL_PROB*100:.1f}% over {IL_DECAY_STEPS} steps")
if getattr(TrainingParameters, 'IL_TYPE', 'expert') == "expert":
    # 原来是 APF，这里改成 CBF
    print("Imitation teacher: CBF tracker policy")

def_attr = lambda name, default: getattr(RecordingParameters, name, default)
SUMMARY_PATH = def_attr('SUMMARY_PATH', f'./runs/TrackingEnv/{RecordingParameters.EXPERIMENT_NAME}{RecordingParameters.TIME}')
MODEL_PATH = def_attr('MODEL_PATH', f'./models/TrackingEnv/{RecordingParameters.EXPERIMENT_NAME}{RecordingParameters.TIME}')
GIFS_PATH = def_attr('GIFS_PATH', osp.join(MODEL_PATH, 'gifs'))
EVAL_INTERVAL = int(def_attr('EVAL_INTERVAL', 20000))
SAVE_INTERVAL = int(def_attr('SAVE_INTERVAL', 5e5))
BEST_INTERVAL = int(def_attr('BEST_INTERVAL', 0))
GIF_INTERVAL = int(def_attr('GIF_INTERVAL', 1e5))
EVAL_EPISODES = int(def_attr('EVAL_EPISODES', 8))

all_args = {
    'seed': SetupParameters.SEED,
    'n_envs': TrainingParameters.N_ENVS,
    'n_steps': TrainingParameters.N_STEPS,
    'learning_rate': TrainingParameters.lr,
    'max_steps': TrainingParameters.N_MAX_STEPS,
    'episode_len': EnvParameters.EPISODE_LEN,
    'n_actions': EnvParameters.N_ACTIONS,
    'opponent_type': TrainingParameters.OPPONENT_TYPE,
    'il_type': getattr(TrainingParameters, 'IL_TYPE', 'expert'),
    'il_initial_prob': IL_INITIAL_PROB,
    'il_final_prob': IL_FINAL_PROB,
    'il_decay_steps': IL_DECAY_STEPS,
    'lstm_hidden_size': LSTM_HIDDEN_SIZE,
    'num_lstm_layers': NUM_LSTM_LAYERS
}


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


def build_segments_from_rollout(rollout, tbptt_steps):
    segments = []
    current = None

    def start_segment(idx):
        return {
            'actor_obs': [],
            'critic_obs': [],
            'returns': [],
            'values': [],
            'actions': [],
            'old_log_probs': [],
            'episode_starts': [],
            'actor_hidden_h': rollout['actor_hidden_h'][idx].copy(),
            'actor_hidden_c': rollout['actor_hidden_c'][idx].copy(),
            'critic_hidden_h': rollout['critic_hidden_h'][idx].copy(),
            'critic_hidden_c': rollout['critic_hidden_c'][idx].copy()
        }

    def flush(seg):
        if seg is None or len(seg['actor_obs']) == 0:
            return
        segment_dict = {
            'actor_obs': np.stack(seg['actor_obs'], axis=0).astype(np.float32),
            'critic_obs': np.stack(seg['critic_obs'], axis=0).astype(np.float32),
            'returns': np.array(seg['returns'], dtype=np.float32),
            'values': np.array(seg['values'], dtype=np.float32),
            'actions': np.array(seg['actions'], dtype=np.float32),
            'old_log_probs': np.array(seg['old_log_probs'], dtype=np.float32),
            'episode_starts': np.array(seg['episode_starts'], dtype=np.bool_),
            'actor_hidden_h': seg['actor_hidden_h'],
            'actor_hidden_c': seg['actor_hidden_c'],
            'critic_hidden_h': seg['critic_hidden_h'],
            'critic_hidden_c': seg['critic_hidden_c']
        }
        segments.append(segment_dict)

    total_steps = rollout['actor_obs'].shape[0]
    for idx in range(total_steps):
        if current is None:
            current = start_segment(idx)
        elif rollout['episode_starts'][idx]:
            flush(current)
            current = start_segment(idx)

        current['actor_obs'].append(rollout['actor_obs'][idx])
        current['critic_obs'].append(rollout['critic_obs'][idx])
        current['returns'].append(rollout['returns'][idx])
        current['values'].append(rollout['values'][idx])
        current['actions'].append(rollout['actions'][idx])
        current['old_log_probs'].append(rollout['logp'][idx])
        current['episode_starts'].append(rollout['episode_starts'][idx])

        if len(current['actor_obs']) >= tbptt_steps:
            flush(current)
            current = None

    flush(current)
    return segments


def collate_segments(batch_segments):
    batch_size = len(batch_segments)
    if batch_size == 0:
        raise ValueError("Empty segment batch passed to collate.")
    max_len = max(seg['actor_obs'].shape[0] for seg in batch_segments)
    actor_obs = np.zeros((batch_size, max_len, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32)
    critic_obs = np.zeros((batch_size, max_len, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32)
    returns = np.zeros((batch_size, max_len), dtype=np.float32)
    values = np.zeros((batch_size, max_len), dtype=np.float32)
    actions = np.zeros((batch_size, max_len, getattr(NetParameters, 'ACTION_DIM', 2)), dtype=np.float32)
    old_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)
    episode_starts = np.zeros((batch_size, max_len), dtype=np.bool_)
    actor_hidden_h = np.zeros((batch_size, NUM_LSTM_LAYERS, LSTM_HIDDEN_SIZE), dtype=np.float32)
    actor_hidden_c = np.zeros((batch_size, NUM_LSTM_LAYERS, LSTM_HIDDEN_SIZE), dtype=np.float32)
    critic_hidden_h = np.zeros((batch_size, NUM_LSTM_LAYERS, LSTM_HIDDEN_SIZE), dtype=np.float32)
    critic_hidden_c = np.zeros((batch_size, NUM_LSTM_LAYERS, LSTM_HIDDEN_SIZE), dtype=np.float32)

    for idx, seg in enumerate(batch_segments):
        length = seg['actor_obs'].shape[0]
        actor_obs[idx, :length] = seg['actor_obs']
        critic_obs[idx, :length] = seg['critic_obs']
        returns[idx, :length] = seg['returns']
        values[idx, :length] = seg['values']
        actions[idx, :length] = seg['actions']
        old_log_probs[idx, :length] = seg['old_log_probs']
        mask[idx, :length] = 1.0
        episode_starts[idx, :length] = seg['episode_starts']
        actor_hidden_h[idx] = seg['actor_hidden_h']
        actor_hidden_c[idx] = seg['actor_hidden_c']
        critic_hidden_h[idx] = seg['critic_hidden_h']
        critic_hidden_c[idx] = seg['critic_hidden_c']

    return {
        'actor_obs': actor_obs,
        'critic_obs': critic_obs,
        'returns': returns,
        'values': values,
        'actions': actions,
        'old_log_probs': old_log_probs,
        'mask': mask,
        'episode_starts': episode_starts,
        'actor_hidden_h': actor_hidden_h,
        'actor_hidden_c': actor_hidden_c,
        'critic_hidden_h': critic_hidden_h,
        'critic_hidden_c': critic_hidden_c
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
    """
    生成单行训练日志字符串，便于在 terminal 中观察。
    phase: 'IL' or 'RL'
    train_stats: compute_performance_stats 返回的 dict
    avg_losses:  PPO loss 向量或 None（IL 阶段为 None）
    recent_window_stats: 最近若干 iteration 的滑动平均（dict），可为 None
    il_loss: IL 阶段的 loss (新增)
    """
    parts = [
        f"[{phase}] step={curr_steps:,}",
        f"ep={curr_episodes:,}",
        f"ILp={il_prob*100:5.1f}%"
    ]

    # 性能指标（本 iteration）
    if train_stats:
        r_mean = train_stats.get('per_r_mean', 0.0)
        r_std = train_stats.get('per_r_std', 0.0)
        win_mean = train_stats.get('win_mean', 0.0)
        parts.append(f"R={r_mean:7.3f}±{r_std:5.2f}")
        if win_mean > 0.0:
            parts.append(f"Win={win_mean*100:5.1f}%")

    # 损失（仅 RL / PPO）
    if avg_losses is not None:
        names = RecordingParameters.LOSS_NAME
        for idx, val in enumerate(avg_losses):
            if idx < len(names):
                parts.append(f"{names[idx]}={val:7.4f}")
    
    # 新增：IL Loss
    if il_loss is not None:
        parts.append(f"IL_Loss={il_loss:7.4f}")

    # 滑动平均（更稳定的观测）
    if recent_window_stats:
        r_w = recent_window_stats.get('per_r_mean', None)
        win_w = recent_window_stats.get('win_mean', None)
        if r_w is not None:
            parts.append(f"[R@{recent_window_stats['window']}it]={r_w:7.3f}")
        if win_w is not None:
            parts.append(f"[Win@{recent_window_stats['window']}it]={win_w*100:5.1f}%")

    return " | ".join(parts)


def main():
    model_dict = None

    if def_attr('RETRAIN', False):
        restore_path = def_attr('RESTORE_DIR', None)
        if restore_path:
            model_path = restore_path + "/tracker_net_checkpoint.pkl"
            if os.path.exists(model_path):
                model_dict = torch.load(model_path, map_location='cpu')

    global_summary = None
    if def_attr('TENSORBOARD', True):
        # 日志目录强制绑到 MODEL_PATH 下，保证模型与 tfevents 在一起
        tb_dir = osp.join(MODEL_PATH, "tfevents")
        os.makedirs(tb_dir, exist_ok=True)
        global_summary = SummaryWriter(tb_dir)
        print(f'Launching tensorboard at: {tb_dir}\n')

    if setproctitle is not None:
        setproctitle.setproctitle(
            RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + getattr(RecordingParameters, 'ENTITY', 'user'))
    set_global_seeds(SetupParameters.SEED)

    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    training_model = Model(global_device, True, lstm_hidden_size=LSTM_HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS)

    if model_dict is not None:
        training_model.network.load_state_dict(model_dict['model'])
        training_model.net_optimizer.load_state_dict(model_dict['optimizer'])

    opponent_model = None
    opponent_weights = None
    if TrainingParameters.OPPONENT_TYPE == "policy":
        opponent_model = Model(global_device, False, lstm_hidden_size=LSTM_HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS)
        opp_path = SetupParameters.PRETRAINED_TARGET_PATH
        if opp_path and os.path.exists(opp_path):
            opponent_dict = torch.load(opp_path, map_location='cpu')
            opponent_model.network.load_state_dict(opponent_dict['model'])
            opponent_weights = opponent_model.get_weights()

    # 固定mission=0 (tracker训练)
    global_policy_manager = None
    if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} and TrainingParameters.ADAPTIVE_SAMPLING:
        global_policy_manager = PolicyManager()

    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    eval_env = TrackingEnv()

    curr_steps = int(model_dict.get("step", 0)) if model_dict is not None else 0
    curr_episodes = int(model_dict.get("episode", 0)) if model_dict is not None else 0
    best_perf = float(model_dict.get("reward", -1e9)) if model_dict is not None else -1e9
    avg_perf_for_best = best_perf

    # 最近若干 iteration 的滑动窗口，用于更平滑的统计
    RECENT_WINDOW_ITERS = 20
    recent_rewards = deque(maxlen=RECENT_WINDOW_ITERS)
    recent_wins = deque(maxlen=RECENT_WINDOW_ITERS)

    last_test_t = -int(EVAL_INTERVAL) - 1
    last_model_t = -int(SAVE_INTERVAL) - 1
    last_best_t = -int(BEST_INTERVAL) - 1
    last_gif_t = -int(GIF_INTERVAL) - 1

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(GIFS_PATH, exist_ok=True)

    # 新增：控制 train 打印频率
    last_train_log_t = 0
    # 累积一个“逻辑 epoch”内的统计
    epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
    epoch_loss_buffer = []   # 累积 RL loss（IL 不一定有）
    epoch_il_loss_buffer = [] # 新增：累积 IL loss

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
                # ---------------- IL / 模仿阶段 ----------------
                jobs = [e.imitation.remote(weights, opponent_weights, curr_steps) for e in envs]
                il_batches = ray.get(jobs)
                actor_vec = np.concatenate([b['actor_obs'] for b in il_batches], axis=0)
                critic_vec = np.concatenate([b['critic_obs'] for b in il_batches], axis=0)
                lbl = np.concatenate([b['actions'] for b in il_batches], axis=0)
                total_il_episodes = 0
                for batch in il_batches:
                    perf = batch['performance']
                    performance_dict['per_r'].extend(perf.get('per_r', []))
                    performance_dict['per_episode_len'].extend(perf.get('per_episode_len', []))
                    performance_dict['win'].extend(perf.get('win', [])) # 新增：提取 win
                    total_il_episodes += batch.get('episodes', 0)
                idx = np.random.permutation(len(actor_vec))
                actor_vec, critic_vec, lbl = actor_vec[idx], critic_vec[idx], lbl[idx]
                mb_loss = []
                for _ in range(3):
                    for start in range(0, len(actor_vec), TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        loss_result = training_model.imitation_train(
                            actor_vec[start:end], critic_vec[start:end], lbl[start:end]
                        )
                        if not np.all(np.isfinite(loss_result)):
                            continue
                        mb_loss.append(loss_result)
                if global_summary and mb_loss:
                    valid_il_losses = [loss for loss in mb_loss if np.all(np.isfinite(loss))]
                    if valid_il_losses:
                        avg_il_loss = np.nanmean([loss[0] for loss in valid_il_losses])
                        global_summary.add_scalar('Train/imitation_loss', avg_il_loss, curr_steps)
                        epoch_il_loss_buffer.append(avg_il_loss) # 新增：记录 IL loss
                curr_steps += int(TrainingParameters.N_ENVS * TrainingParameters.N_STEPS)
                curr_episodes += total_il_episodes

                # 只更新统计缓冲，不立即打印
                train_stats = compute_performance_stats(performance_dict)
                if len(performance_dict['per_r']) > 0:
                    recent_rewards.append(train_stats['per_r_mean'])
                if len(performance_dict.get('win', [])) > 0:
                    recent_wins.append(train_stats['win_mean'])

                # 累积到 epoch buffer
                epoch_perf_buffer['per_r'].extend(performance_dict['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(performance_dict['per_episode_len'])
                epoch_perf_buffer['win'].extend(performance_dict['win']) # 新增：累积 win

            else:
                # ---------------- RL / PPO 阶段 ----------------
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
                        global_summary.add_scalar('Train/adv_std', float(np.std(adv_concat)), curr_steps)
                        global_summary.add_scalar('Train/adv_min', float(np.min(adv_concat)), curr_steps)
                        global_summary.add_scalar('Train/adv_max', float(np.max(adv_concat)), curr_steps)

                mb_loss = []
                seg_indices = np.arange(len(segments))
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(seg_indices)
                    for start in range(0, len(seg_indices), TrainingParameters.MINIBATCH_SIZE):
                        batch_indices = seg_indices[start:start + TrainingParameters.MINIBATCH_SIZE]
                        batch_segments = [segments[i] for i in batch_indices]
                        batch = collate_segments(batch_segments)
                        loss_result = training_model.train(
                            batch['actor_obs'], batch['critic_obs'], batch['returns'],
                            batch['values'], batch['actions'], batch['old_log_probs'],
                            (batch['actor_hidden_h'], batch['actor_hidden_c']),
                            (batch['critic_hidden_h'], batch['critic_hidden_c']),
                            batch['mask'], batch['episode_starts']
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
                elif global_summary and not valid_losses:
                    global_summary.add_scalar('Train/invalid_batch_ratio', 1.0, curr_steps)

                curr_steps += steps_batch
                curr_episodes += episodes_batch

                train_stats = compute_performance_stats(performance_dict)
                avg_perf_for_best = train_stats.get('per_r_mean', -1e9)

                # 维护滑动窗口
                if len(performance_dict['per_r']) > 0:
                    recent_rewards.append(train_stats['per_r_mean'])
                if len(performance_dict.get('win', [])) > 0:
                    recent_wins.append(train_stats['win_mean'])

                # 累积 epoch 统计和 loss（用于低频打印）
                epoch_perf_buffer['per_r'].extend(performance_dict['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(performance_dict['per_episode_len'])
                epoch_perf_buffer['win'].extend(performance_dict['win'])
                if valid_losses:
                    epoch_loss_buffer.extend(valid_losses)

            # ------------ 公共逻辑：TensorBoard / 保存 / Eval / GIF ------------
            # 这里的 train_stats/avg_perf_for_best 只在 RL 分支定义，
            # 对 IL 分支，我们只在 tensorboard 中记录 performance，不参与 best model。
            if global_summary and 'train_stats' in locals():
                for key, value in train_stats.items():
                    global_summary.add_scalar(f'Train/{key}', value, curr_steps)

            # 每个“逻辑 epoch”打印一次 train 概要（混合 IL / RL）
            if curr_steps - last_train_log_t >= TrainingParameters.LOG_EPOCH_STEPS:
                last_train_log_t = curr_steps

                # 统计 epoch 内的平均性能
                epoch_stats = compute_performance_stats(epoch_perf_buffer)
                # 统计 epoch 内的平均 loss（只看 RL，有可能为空）
                avg_epoch_losses = None
                if epoch_loss_buffer:
                    avg_epoch_losses = np.nanmean(epoch_loss_buffer, axis=0)
                
                # 新增：统计 epoch 内的平均 IL loss
                avg_epoch_il_loss = None
                if epoch_il_loss_buffer:
                    avg_epoch_il_loss = float(np.mean(epoch_il_loss_buffer))

                # 使用最近窗口（recent_rewards / recent_wins）给出平滑指标
                recent_window_stats = None
                if len(recent_rewards) > 0:
                    recent_window_stats = {
                        'window': len(recent_rewards),
                        'per_r_mean': float(np.mean(recent_rewards)),
                        'win_mean': float(np.mean(recent_wins)) if recent_wins else None,
                    }

                # phase 只用于标记当前 IL 概率下的大致训练状态
                phase = "IL" if il_prob > 0.5 else "RL"
                print(format_train_log(
                    curr_steps, curr_episodes, phase=phase,
                    il_prob=il_prob,
                    train_stats=epoch_stats,
                    avg_losses=avg_epoch_losses,
                    recent_window_stats=recent_window_stats,
                    il_loss=avg_epoch_il_loss # 新增参数
                ))

                # 清空 epoch 缓冲
                epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
                epoch_loss_buffer = []
                epoch_il_loss_buffer = [] # 新增：清空

            # ---- 模型保存 & best model ----
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
                # 去掉 "New best model saved ..." 的打印
                # ...no console print here...

            # ---- Eval ----
            if curr_steps - last_test_t >= EVAL_INTERVAL:
                last_test_t = curr_steps
                eval_stats = evaluate_single_agent(eval_env, training_model, opponent_model, global_device)
                # 精简 eval 打印
                msg_parts = [f"[EVAL] step={curr_steps:,}"]
                for k, v in eval_stats.items():
                    msg_parts.append(f"{k}={v:7.3f}")
                print(" | ".join(msg_parts))
                if global_summary:
                    for key, value in eval_stats.items():
                        global_summary.add_scalar(f'Eval/{key}', value, curr_steps)

            # ---- GIF ----
            if curr_steps - last_gif_t >= GIF_INTERVAL:
                last_gif_t = curr_steps
                generate_one_episode_gif(eval_env, training_model, opponent_model, global_device, curr_steps)
                print(f"GIF saved for step {curr_steps}")
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
                                  policy_manager, current_policy_name, current_policy_id, opponent_hidden=None,
                                  privileged_state=None):
    """评估时获取对手(target)动作"""
    if opponent_type == "policy":
        if opponent_model is None:
            raise RuntimeError("OPPONENT_TYPE=policy but opponent_model is None")
        dummy_context = np.zeros(NetParameters.CONTEXT_LEN, dtype=np.float32)
        critic_obs = np.concatenate([target_obs, dummy_context])
        opp_action, _, new_opponent_hidden, _, _ = opponent_model.evaluate(target_obs, critic_obs, opponent_hidden, greedy=True)
        return opp_action, new_opponent_hidden
        
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

        agent_hidden = None
        opponent_hidden = None

        current_policy_name, current_policy_id = (None, -1)
        if eval_policy_manager:
            current_policy_name, current_policy_id = eval_policy_manager.sample_policy("target")
            eval_policy_manager.reset()

        while not done and ep_len < EnvParameters.EPISODE_LEN:
            # Agent(tracker)动作
            critic_obs_full = np.concatenate([tracker_obs, get_opponent_id_one_hot(current_policy_id)])
            agent_action, _, agent_hidden, _, _ = agent_model.evaluate(tracker_obs, critic_obs_full, agent_hidden, greedy=True)

            # 对手(target)动作
            target_action, opponent_hidden = get_opponent_action_for_eval(
                target_obs, TrainingParameters.OPPONENT_TYPE,
                opponent_model, eval_policy_manager, current_policy_name, current_policy_id, opponent_hidden
            )

            obs_result, reward, terminated, truncated, info = eval_env.step((agent_action, target_action))
            done = terminated or truncated
            ep_r += float(reward)
            ep_len += 1
            
            # 解析观测
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
    print("Generating GIF for one episode...")
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

    agent_hidden = None
    opponent_hidden = None

    eval_policy_manager = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
    current_policy_name, current_policy_id = (None, -1)
    if eval_policy_manager:
        current_policy_name, current_policy_id = eval_policy_manager.sample_policy("target")
        eval_policy_manager.reset()

    while not done and ep_len < EnvParameters.EPISODE_LEN:
        frame = eval_env.render(mode='rgb_array')
        if frame is not None:
            episode_frames.append(frame)

        critic_obs_full = np.concatenate([tracker_obs, get_opponent_id_one_hot(current_policy_id)])
        agent_action, _, agent_hidden, _, _ = agent_model.evaluate(tracker_obs, critic_obs_full, agent_hidden, greedy=True)

        target_action, opponent_hidden = get_opponent_action_for_eval(
            target_obs, TrainingParameters.OPPONENT_TYPE,
            opponent_model, eval_policy_manager, current_policy_name, current_policy_id, opponent_hidden
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