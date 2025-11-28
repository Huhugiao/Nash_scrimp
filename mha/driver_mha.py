import os
import sys

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
except ImportError:
    setproctitle = None

from torch.utils.tensorboard import SummaryWriter

from map_config import EnvParameters
from mha.alg_parameters_mha import *
from env import TrackingEnv
from mha.model_mha import Model
from mha.runner_mha import Runner
from util import set_global_seeds, make_gif
from rule_policies import TARGET_POLICY_REGISTRY
from policymanager import PolicyManager

# ============ IL配置 ============
IL_INITIAL_PROB = 1.0      # phase2 起始值
IL_FINAL_PROB   = 0.05     # phase2 结束值
IL_DECAY_STEPS  = 3e6      # phase2 持续 3M 步

# 新增：IL 数据复用配置
IL_PHASE1_NEW_RATIO = 0.6   # Phase 1: 60% 新数据 + 40% buffer
IL_PHASE2_NEW_RATIO = 0.3   # Phase 2: 30% 新数据 + 70% buffer

PURE_RL_SWITCH = 0
if PURE_RL_SWITCH:
    IL_INITIAL_PROB = 0
    IL_FINAL_PROB = 0
    IL_DECAY_STEPS = 1

# ============ Ray初始化 ============
if not ray.is_initialized():
    ray.init(num_gpus=SetupParameters.NUM_GPU)

print("Welcome to SCRIMP with MHA (Transformer) - Training Tracker!")

if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
    available = sorted(list(TARGET_POLICY_REGISTRY.keys()))
    weights = TrainingParameters.RANDOM_OPPONENT_WEIGHTS.get("target", {})
    print(f"Multi-Opponent Configuration:")
    print(f"  Available target policies: {', '.join(available)}")
    print(f"  Sampling weights: {weights}")
    print(f"  Adaptive sampling: {'ENABLED' if TrainingParameters.ADAPTIVE_SAMPLING else 'DISABLED'}")

print(f"IL probability will cosine anneal from {IL_INITIAL_PROB*100:.1f}% to {IL_FINAL_PROB*100:.1f}% over {IL_DECAY_STEPS:.0f} steps")

# ============ 路径配置 ============
def_attr = lambda name, default: getattr(RecordingParameters, name, default)

# 模型与日志统一放在同一目录
MODEL_PATH = def_attr('MODEL_PATH', f'./models/TrackingEnv/{RecordingParameters.EXPERIMENT_NAME}{RecordingParameters.TIME}')
# TensorBoard 事件文件直接写在 MODEL_PATH 下
SUMMARY_PATH = MODEL_PATH
# GIF 放在模型目录下的 gifs 子目录
GIFS_PATH = osp.join(MODEL_PATH, 'gifs')

EVAL_INTERVAL = int(def_attr('EVAL_INTERVAL', 100000))  # 降低评估间隔
SAVE_INTERVAL = int(def_attr('SAVE_INTERVAL', 5e5))
BEST_INTERVAL = int(def_attr('BEST_INTERVAL', 0))
GIF_INTERVAL = int(def_attr('GIF_INTERVAL', 200000))  # 添加GIF间隔
EVAL_EPISODES = int(def_attr('EVAL_EPISODES', 16))


# ============ 工具函数 ============
def get_cosine_annealing_il_prob(current_step):
    if current_step >= IL_DECAY_STEPS:
        return IL_FINAL_PROB
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / IL_DECAY_STEPS))
    return IL_FINAL_PROB + (IL_INITIAL_PROB - IL_FINAL_PROB) * cosine_decay


def get_scheduled_lr(current_step):
    final_lr = getattr(TrainingParameters, 'LR_FINAL', TrainingParameters.lr)
    progress = min(max(current_step / TrainingParameters.N_MAX_STEPS, 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (TrainingParameters.lr - final_lr) * weight


def build_segments_from_rollout(rollout, window_size):
    """将rollout按episode边界和窗口大小切分成segments"""
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
        cursor += seg_len


def collate_segments(batch_segments):
    """将多个segment整理成batch"""
    batch_size = len(batch_segments)
    if batch_size == 0:
        raise ValueError("Empty segment batch")
    
    max_len = max(len(s['actor_obs']) for s in batch_segments)
    
    batch = {
        'actor_obs': np.zeros((batch_size, max_len, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32),
        'critic_obs': np.zeros((batch_size, max_len, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32),
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


def collate_il_segments(batch_segments):
    """整理IL数据，将每条segment pad/truncate到 CONTEXT_WINDOW"""
    batch_size = len(batch_segments)
    if batch_size == 0:
        return None

    max_len = NetParameters.CONTEXT_WINDOW

    batch = {
        'actor_obs': np.zeros((batch_size, max_len, NetParameters.ACTOR_VECTOR_LEN), dtype=np.float32),
        'critic_obs': np.zeros((batch_size, max_len, NetParameters.CRITIC_VECTOR_LEN), dtype=np.float32),
        'actions': np.zeros((batch_size, max_len, NetParameters.ACTION_DIM), dtype=np.float32),
        'mask': np.zeros((batch_size, max_len), dtype=np.float32),
    }

    for i, seg in enumerate(batch_segments):
        seg_len = len(seg['actor_obs'])
        use_len = min(seg_len, max_len)
        batch['actor_obs'][i, :use_len] = seg['actor_obs'][:use_len]
        batch['critic_obs'][i, :use_len] = seg['critic_obs'][:use_len]
        batch['actions'][i, :use_len] = seg['expert_actions'][:use_len]
        batch['mask'][i, :use_len] = 1.0

    return batch

def _collect_il_segments_in_range(data, start_idx, end_idx):
    """
    在 [start_idx, end_idx) 区间内：
    - 只看 is_expert_step == True 的步
    - 把连续 expert 区间按 CONTEXT_WINDOW 切成 segments
    """
    il_segments = []
    max_len = NetParameters.CONTEXT_WINDOW

    is_expert = data['is_expert_step']
    n = len(is_expert)
    s = max(0, start_idx)
    e = min(n, end_idx)
    idx = s

    while idx < e:
        # 跳过非 expert 步
        while idx < e and not is_expert[idx]:
            idx += 1
        if idx >= e:
            break

        # 连续 expert 区间 [seg_start, seg_end)
        seg_start = idx
        while idx < e and is_expert[idx]:
            idx += 1
        seg_end = idx

        # 按窗口切 segment
        cursor = seg_start
        while cursor < seg_end:
            win_end = min(cursor + max_len, seg_end)
            if win_end <= cursor:
                break
            il_segments.append({
                'actor_obs': data['actor_obs'][cursor:win_end].copy(),
                'critic_obs': data['critic_obs'][cursor:win_end].copy(),
                'expert_actions': data['expert_actions'][cursor:win_end].copy(),
            })
            cursor = win_end

    return il_segments


def extract_il_data_from_rollout(rollout_data, phase: int):
    """
    从 rollout 提取 IL 数据：
    - phase == 1：无论成功与否，都在所有 episode 上使用 is_expert_step==True 的步
    - phase == 2：仅使用成功 episode 中、is_expert_step==True 的步
    """
    data = rollout_data['data']
    il_segments = []

    if phase == 1:
        # 阶段1：纯模仿 —— agent 自己产生的所有状态上，只要有 expert 标签就学
        n = len(data['is_expert_step'])
        il_segments.extend(_collect_il_segments_in_range(data, 0, n))
    else:
        # 阶段2：只在成功 episode 的 expert 步上做 IL
        for ep in data['episode_success']:
            if not ep['success']:
                continue
            start_idx, end_idx = ep['start_idx'], ep['end_idx']
            if end_idx <= start_idx:
                continue
            il_segments.extend(_collect_il_segments_in_range(data, start_idx, end_idx))

    return il_segments


def extract_rl_data_from_rollout(rollout_data):
    """从rollout提取RL数据（所有数据）"""
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

def sample_il_data_mixed(all_il_segments, il_buffer, batch_size, new_ratio=0.5):
    """
    混合采样：从新数据和 buffer 中按比例采样
    
    Args:
        all_il_segments: 当前 rollout 的新 IL 数据
        il_buffer: 历史 IL 数据缓冲区
        batch_size: 目标采样数量
        new_ratio: 新数据占比 (0.0 ~ 1.0)
    
    Returns:
        采样后的 segment 列表，若数据不足则返回 None
    """
    samples = []
    
    n_new_target = int(batch_size * new_ratio)
    n_buffer_target = batch_size - n_new_target
    
    # 1) 从新数据采样
    if all_il_segments and n_new_target > 0:
        n_new = min(len(all_il_segments), n_new_target)
        samples.extend(random.sample(all_il_segments, n_new))
    
    # 2) 从 buffer 采样（补足剩余）
    remaining = batch_size - len(samples)
    if len(il_buffer) > 0 and remaining > 0:
        n_buffer = min(len(il_buffer), remaining)
        samples.extend(random.sample(list(il_buffer), n_buffer))
    
    # 3) 如果 buffer 不够，再从新数据补
    remaining = batch_size - len(samples)
    if all_il_segments and remaining > 0:
        available = [s for s in all_il_segments if s not in samples]
        if available:
            n_extra = min(len(available), remaining)
            samples.extend(random.sample(available, n_extra))
    
    if len(samples) == 0:
        return None
    
    return samples


def compute_adaptive_new_ratio(recent_il_losses, recent_win_rates, phase):
    """
    根据 IL loss 和胜率自适应调整新旧数据比例
    
    - IL loss 高 → 多用新数据（当前策略产生的状态更相关）
    - 胜率低 → 多复用 buffer（需要更多专家示范）
    """
    if phase == 1:
        base_ratio = IL_PHASE1_NEW_RATIO
    else:
        base_ratio = IL_PHASE2_NEW_RATIO
    
    if len(recent_il_losses) < 10 or len(recent_win_rates) < 10:
        return base_ratio
    
    avg_loss = float(np.mean(recent_il_losses))
    avg_win = float(np.mean(recent_win_rates))
    
    # 动态调整
    adjustment = 0.0
    
    # IL loss 高（>2.0）→ 稍微多用新数据
    if avg_loss > 2.0:
        adjustment += 0.1
    elif avg_loss < 0.5:
        adjustment -= 0.1
    
    # 胜率低（<20%）→ 多复用 buffer
    if avg_win < 0.2:
        adjustment -= 0.15
    elif avg_win > 0.5:
        adjustment += 0.1
    
    return np.clip(base_ratio + adjustment, 0.2, 0.8)


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


def format_train_log(curr_steps, curr_episodes, il_prob, train_stats, il_buffer_size):
    parts = [
        f"[TRAIN] step={curr_steps:,}",
        f"ep={curr_episodes:,}",
        f"ILp={il_prob*100:5.1f}%",
        f"IL_buf={il_buffer_size}"
    ]
    if train_stats:
        parts.append(f"Rew={train_stats.get('per_r_mean', 0):.2f}±{train_stats.get('per_r_std', 0):.2f}")
        parts.append(f"Win={train_stats.get('win_mean', 0)*100:.1f}%")
    return " | ".join(parts)


# ============ 评估函数 ============
def evaluate_single_agent(eval_env, agent_model, opponent_model, device):
    """评估agent性能"""
    performance = {'per_r': [], 'per_episode_len': [], 'win': []}
    eval_pm = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None

    for _ in range(EVAL_EPISODES):
        obs_result, _ = eval_env.reset()
        tracker_obs, target_obs = _parse_eval_obs(obs_result)
        
        done = False
        ep_r, ep_len = 0.0, 0
        agent_history = None
        
        current_policy = None
        if eval_pm:
            current_policy, _ = eval_pm.sample_policy("target")
            eval_pm.reset()

        while not done and ep_len < EnvParameters.EPISODE_LEN:
            # Agent动作
            agent_action, _, agent_history, _, _ = agent_model.evaluate(
                tracker_obs, target_obs, agent_history, greedy=True
            )
            
            # 对手动作
            if TrainingParameters.OPPONENT_TYPE == "policy" and opponent_model:
                target_action, _, _, _, _ = opponent_model.evaluate(
                    target_obs, target_obs, None, greedy=True
                )
            elif eval_pm and current_policy:
                target_action = eval_pm.get_action(current_policy, target_obs)
            else:
                target_action = np.zeros(NetParameters.ACTION_DIM, dtype=np.float32)

            obs_result, reward, terminated, truncated, info = eval_env.step((agent_action, target_action))
            done = terminated or truncated
            ep_r += float(reward)
            ep_len += 1
            
            tracker_obs, target_obs = _parse_eval_obs(obs_result)

        performance['per_r'].append(ep_r)
        performance['per_episode_len'].append(ep_len)
        performance['win'].append(1 if info.get('reason') == 'tracker_caught_target' else 0)

    stats = compute_performance_stats(performance)
    return stats.get('per_r_mean', 0), stats.get('win_mean', 0)


def _parse_eval_obs(obs_result):
    """解析评估时的观测"""
    if isinstance(obs_result, tuple) and len(obs_result) == 2:
        return np.asarray(obs_result[0], dtype=np.float32), np.asarray(obs_result[1], dtype=np.float32)
    obs = np.asarray(obs_result, dtype=np.float32)
    return obs, obs[:NetParameters.CRITIC_VECTOR_LEN]


def _record_eval_gif(eval_env, agent_model, opponent_model, device, gif_path):
    """用当前模型在 eval_env 跑一条 episode 并保存 GIF。"""
    frames = []
    obs_result, _ = eval_env.reset()
    tracker_obs, target_obs = _parse_eval_obs(obs_result)
    done = False
    ep_len = 0
    agent_history = None

    # 尝试渲染首帧
    try:
        frame = eval_env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    except Exception:
        pass

    eval_pm = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} else None
    current_policy = None
    if eval_pm:
        current_policy, _ = eval_pm.sample_policy("target")
        eval_pm.reset()

    with torch.no_grad():
        while not done and ep_len < EnvParameters.EPISODE_LEN:
            agent_action, _, agent_history, _, _ = agent_model.evaluate(
                tracker_obs, target_obs, agent_history, greedy=True
            )

            if TrainingParameters.OPPONENT_TYPE == "policy" and opponent_model:
                target_action, _, _, _, _ = opponent_model.evaluate(
                    target_obs, target_obs, None, greedy=True
                )
            elif eval_pm and current_policy:
                target_action = eval_pm.get_action(current_policy, target_obs)
            else:
                target_action = np.zeros(NetParameters.ACTION_DIM, dtype=np.float32)

            obs_result, reward, terminated, truncated, info = eval_env.step((agent_action, target_action))
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
        os.makedirs(osp.dirname(gif_path), exist_ok=True)
        make_gif(frames, gif_path, fps=EnvParameters.N_ACTIONS // 2)
        print(f"GIF saved: {gif_path}")


# ============ 主函数 ============
def main():
    # 加载检查点
    model_dict = None
    if def_attr('RETRAIN', False):
        checkpoint_path = RecordingParameters.RESTORE_DIR
        if checkpoint_path and os.path.exists(checkpoint_path):
            model_dict = torch.load(checkpoint_path, map_location='cpu')
            print(f"Loaded checkpoint from {checkpoint_path}")

    # Tensorboard
    global_summary = SummaryWriter(log_dir=SUMMARY_PATH) if def_attr('TENSORBOARD', True) else None

    if setproctitle:
        setproctitle.setproctitle(f"mha_train_{RecordingParameters.EXPERIMENT_NAME}")
    set_global_seeds(SetupParameters.SEED)

    # 模型初始化
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    training_model = Model(global_device, True)
    if model_dict:
        training_model.set_weights(model_dict['model'])

    opponent_model, opponent_weights = None, None
    if TrainingParameters.OPPONENT_TYPE == "policy":
        opponent_model = Model(global_device, False)
        if SetupParameters.PRETRAINED_TARGET_PATH:
            opp_dict = torch.load(SetupParameters.PRETRAINED_TARGET_PATH, map_location='cpu')
            opponent_model.set_weights(opp_dict['model'])
        opponent_weights = opponent_model.get_weights()

    # Policy Manager
    global_pm = PolicyManager() if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} and TrainingParameters.ADAPTIVE_SAMPLING else None

    # 环境
    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    eval_env = TrackingEnv()

    # 状态追踪
    curr_steps = int(model_dict.get("step", 0)) if model_dict else 0
    curr_episodes = int(model_dict.get("episode", 0)) if model_dict else 0
    best_perf = float(model_dict.get("reward", -1e9)) if model_dict else -1e9

    il_buffer = deque(maxlen=10000)
    epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
    epoch_loss_buffer = []

    # 修复：初始化为0而不是负数，确保第一次正确触发
    last_test_t = 0
    last_model_t = 0
    last_best_t = 0
    last_gif_t = 0
    last_train_log_t = 0

    # ========= 两阶段 IL 控制 =========
    phase = 1
    IL_LOSS_THRESHOLD = 0.5
    recent_il_losses = deque(maxlen=100)
    recent_win_rates = deque(maxlen=100)  # 新增：追踪胜率

    # 新增：IL 效果追踪
    il_action_errors = deque(maxlen=500)

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(GIFS_PATH, exist_ok=True)

    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            # 阶段控制下的 IL 概率
            if phase == 1:
                il_prob = 1.0
            else:
                il_prob = get_cosine_annealing_il_prob(curr_steps)

            new_lr = get_scheduled_lr(curr_steps)
            training_model.update_learning_rate(new_lr)
            
            model_weights = training_model.get_weights()
            pm_state = {k: list(v) for k, v in global_pm.win_history.items()} if global_pm else None
            
            # 并行数据采集
            job_list = [
                envs[i].run.remote(model_weights, opponent_weights, curr_steps, pm_state, il_prob)
                for i in range(TrainingParameters.N_ENVS)
            ]
            results = ray.get(job_list)
            
            # 聚合数据
            all_rl_segments = []
            all_il_segments = []
            total_new_episodes = 0
            
            for result in results:
                rl_data = extract_rl_data_from_rollout(result)
                rl_segments = build_segments_from_rollout(rl_data, NetParameters.CONTEXT_WINDOW)
                all_rl_segments.extend(rl_segments)
                
                il_segments = extract_il_data_from_rollout(result, phase=phase)
                all_il_segments.extend(il_segments)
                
                perf = result['performance']
                epoch_perf_buffer['per_r'].extend(perf['per_r'])
                epoch_perf_buffer['per_episode_len'].extend(perf['per_episode_len'])
                epoch_perf_buffer['win'].extend(perf['win'])
                total_new_episodes += result['episodes']
                
                # 追踪胜率
                recent_win_rates.extend(perf['win'])
                
                # 计算 IL action error（验证学习效果）
                data = result['data']
                expert_mask = data['is_expert_step']
                if expert_mask.sum() > 0:
                    agent_actions = data['actions'][expert_mask]
                    expert_actions = data['expert_actions'][expert_mask]
                    # 注意：agent_actions 是 pre_tanh，需要转换
                    agent_actions_tanh = np.tanh(agent_actions)
                    errors = np.abs(agent_actions_tanh - expert_actions).mean()
                    il_action_errors.append(errors)
                
                if global_pm and result['policy_manager_state']:
                    for name, history in result['policy_manager_state'].items():
                        if name in global_pm.win_history:
                            global_pm.win_history[name] = deque(history, maxlen=TrainingParameters.ADAPTIVE_SAMPLING_WINDOW)
            
            # 先添加到 buffer（两个阶段都添加）
            il_buffer.extend(all_il_segments)
            curr_steps += TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
            curr_episodes += total_new_episodes

            # 计算自适应新旧数据比例
            adaptive_new_ratio = compute_adaptive_new_ratio(recent_il_losses, recent_win_rates, phase)

            # 训练
            batch_il_losses = []
            if all_rl_segments:
                random.shuffle(all_rl_segments)
                
                for _ in range(TrainingParameters.N_EPOCHS):
                    for mb_start in range(0, len(all_rl_segments), TrainingParameters.MINIBATCH_SIZE):
                        mb_end = min(mb_start + TrainingParameters.MINIBATCH_SIZE, len(all_rl_segments))
                        batch_segments = all_rl_segments[mb_start:mb_end]
                        
                        if not batch_segments:
                            continue
                        
                        batch = collate_segments(batch_segments)
                        
                        # ========== 修复：统一使用混合采样 ==========
                        il_batch = None
                        if phase == 1:
                            # Phase 1: 混合采样，新数据占比较高
                            il_samples = sample_il_data_mixed(
                                all_il_segments, 
                                il_buffer, 
                                TrainingParameters.MINIBATCH_SIZE,
                                new_ratio=adaptive_new_ratio
                            )
                            if il_samples:
                                il_batch = collate_il_segments(il_samples)
                        else:
                            # Phase 2: 混合采样，buffer 占比较高
                            il_samples = sample_il_data_mixed(
                                all_il_segments, 
                                il_buffer, 
                                TrainingParameters.MINIBATCH_SIZE // 2,
                                new_ratio=adaptive_new_ratio
                            )
                            if il_samples:
                                il_batch = collate_il_segments(il_samples)
                        
                        # 训练
                        result = training_model.train(
                            batch['actor_obs'], batch['critic_obs'],
                            batch['returns'], batch['values'],
                            batch['actions'], batch['old_log_probs'],
                            batch['mask'], il_batch=il_batch
                        )
                        
                        if isinstance(result, dict):
                            epoch_loss_buffer.append(result.get('losses', []))
                            if 'il_loss' in result and result['il_loss'] is not None:
                                batch_il_losses.append(float(result['il_loss']))
                        elif isinstance(result, (list, tuple)):
                            epoch_loss_buffer.append(result)
            
            # 如果没有从 train 中获取 IL loss，单独计算
            if not batch_il_losses and (all_il_segments or len(il_buffer) > 0):
                il_samples = sample_il_data_mixed(
                    all_il_segments, 
                    il_buffer, 
                    TrainingParameters.MINIBATCH_SIZE,
                    new_ratio=0.5
                )
                if il_samples:
                    il_batch = collate_il_segments(il_samples)
                    if il_batch is not None:
                        il_result = training_model.imitation_train(
                            il_batch['actor_obs'], 
                            il_batch['critic_obs'], 
                            il_batch['actions']
                        )
                        if isinstance(il_result, (list, tuple)) and len(il_result) > 0:
                            batch_il_losses.append(float(il_result[0]))
            
            # 更新 recent_il_losses
            if batch_il_losses:
                avg_batch_il = float(np.mean(batch_il_losses))
                recent_il_losses.append(avg_batch_il)
            
            # 日志
            if curr_steps - last_train_log_t >= TrainingParameters.LOG_EPOCH_STEPS:
                last_train_log_t = curr_steps
                train_stats = compute_performance_stats(epoch_perf_buffer) if epoch_perf_buffer['per_r'] else {}

                avg_il_loss = float(np.mean(recent_il_losses)) if recent_il_losses else 0.0
                avg_il_error = float(np.mean(il_action_errors)) if il_action_errors else 0.0
                
                log_str = format_train_log(curr_steps, curr_episodes, il_prob, train_stats, len(il_buffer))
                log_str += f" | phase={phase} | IL_loss={avg_il_loss:.4f} | IL_err={avg_il_error:.4f} | new_ratio={adaptive_new_ratio:.2f}"
                print(log_str)
                
                if global_summary:
                    if train_stats:
                        global_summary.add_scalar('perf/reward', train_stats.get('per_r_mean', 0), curr_steps)
                        global_summary.add_scalar('perf/win_rate', train_stats.get('win_mean', 0), curr_steps)
                    global_summary.add_scalar('train/il_prob', il_prob, curr_steps)
                    global_summary.add_scalar('train/il_buffer_size', len(il_buffer), curr_steps)
                    global_summary.add_scalar('train/lr', new_lr, curr_steps)
                    global_summary.add_scalar('train/il_loss', avg_il_loss, curr_steps)
                    global_summary.add_scalar('train/il_action_error', avg_il_error, curr_steps)
                    global_summary.add_scalar('train/il_new_ratio', adaptive_new_ratio, curr_steps)
                    global_summary.add_scalar('train/phase', phase, curr_steps)
                
                # 阶段切换
                if phase == 1 and len(recent_il_losses) >= 20 and avg_il_loss < IL_LOSS_THRESHOLD:
                    phase = 2
                    print(f"[PHASE] Switch to phase 2 (mixed IL+RL) at step={curr_steps}, avg_il_loss={avg_il_loss:.4f}")

                epoch_perf_buffer = {'per_r': [], 'per_episode_len': [], 'win': []}
                epoch_loss_buffer = []
            
            # 评估
            if curr_steps - last_test_t >= EVAL_INTERVAL:
                last_test_t = curr_steps
                eval_reward, eval_win_rate = evaluate_single_agent(eval_env, training_model, opponent_model, global_device)
                print(f"[EVAL] step={curr_steps:,} reward={eval_reward:.2f} win_rate={eval_win_rate:.2%}")
                
                if global_summary:
                    global_summary.add_scalar('eval/reward', eval_reward, curr_steps)
                    global_summary.add_scalar('eval/win_rate', eval_win_rate, curr_steps)
                
                # 更新最优模型
                current_perf = eval_reward
                if current_perf > best_perf:
                    best_perf = current_perf
                    best_path = os.path.join(MODEL_PATH, 'best_model')
                    os.makedirs(best_path, exist_ok=True)
                    torch.save({
                        'model': training_model.get_weights(),
                        'step': curr_steps,
                        'episode': curr_episodes,
                        'reward': best_perf
                    }, os.path.join(best_path, 'checkpoint.pth'))
                    print(f"[BEST] New best model saved: reward={best_perf:.2f}")
            
            # GIF 录制
            if curr_steps - last_gif_t >= GIF_INTERVAL:
                last_gif_t = curr_steps
                gif_path = os.path.join(GIFS_PATH, f'eval_{curr_steps}.gif')
                _record_eval_gif(eval_env, training_model, opponent_model, global_device, gif_path)
            
            # 只保存最新模型（不定时保存多个）
            if curr_steps - last_model_t >= SAVE_INTERVAL:
                last_model_t = curr_steps
                latest_path = os.path.join(MODEL_PATH, 'latest_model')
                os.makedirs(latest_path, exist_ok=True)
                torch.save({
                    'model': training_model.get_weights(),
                    'step': curr_steps,
                    'episode': curr_episodes,
                    'reward': best_perf
                }, os.path.join(latest_path, 'checkpoint.pth'))
                print(f"[SAVE] Latest model saved at step={curr_steps}")

    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        final_path = os.path.join(MODEL_PATH, 'final_model')
        os.makedirs(final_path, exist_ok=True)
        torch.save({
            'model': training_model.get_weights(),
            'step': curr_steps,
            'episode': curr_episodes,
            'reward': best_perf
        }, os.path.join(final_path, 'checkpoint.pth'))
        print(f"Final model saved to {final_path}")
        
        if global_summary:
            global_summary.close()
        ray.shutdown()


if __name__ == "__main__":
    main()

