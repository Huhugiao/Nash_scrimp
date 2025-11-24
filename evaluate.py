import os
# 设置SDL为dummy模式，避免在多进程中初始化图形界面导致死锁或错误
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
import argparse
import pandas as pd
import time
import multiprocessing
import datetime
import map_config
from map_config import EnvParameters, ObstacleDensity
import json
from collections import defaultdict

from env import TrackingEnv
from lstm.model_lstm import Model  # 从lstm模块导入
from lstm.alg_parameters import *  # 使用通用参数
from util import make_gif, get_opponent_id_one_hot
from policymanager import PolicyManager
from rule_policies import CBFTracker, GreedyTarget, TRACKER_POLICY_REGISTRY, TARGET_POLICY_REGISTRY

# 简化：直接定义策略名称
TRACKER_POLICY_NAMES = tuple(TRACKER_POLICY_REGISTRY.keys())
TARGET_POLICY_NAMES = tuple(TARGET_POLICY_REGISTRY.keys())
TRACKER_TYPE_CHOICES = TRACKER_POLICY_NAMES + ("policy", "all")
TARGET_TYPE_CHOICES = TARGET_POLICY_NAMES + ("policy", "all")

DEFAULT_TRACKER = "CBF"
DEFAULT_TARGET = "Greedy"


def get_available_policies(role: str):
    if role == "tracker":
        return list(TRACKER_POLICY_NAMES)
    if role == "target":
        return list(TARGET_POLICY_NAMES)
    raise ValueError(f"Unknown role: {role}")


class BattleConfig:
    def __init__(self,
                 tracker_type=None,
                 target_type=None,
                 tracker_model_path=None,
                 target_model_path=None,
                 episodes=100,
                 save_gif_freq=10,
                 output_dir="./battle_results",
                 seed=1234,
                 state_space="vector",
                 specific_tracker_strategy=None,
                 specific_target_strategy=None,
                 main_output_dir=None,
                 obstacle_density=None):
        self.tracker_type = tracker_type or DEFAULT_TRACKER
        self.target_type = target_type or DEFAULT_TARGET
        self.tracker_model_path = tracker_model_path
        self.target_model_path = target_model_path
        self.episodes = episodes
        self.save_gif_freq = save_gif_freq
        self.output_dir = output_dir
        self.seed = seed
        self.state_space = str(state_space)
        self.specific_tracker_strategy = specific_tracker_strategy
        self.specific_target_strategy = specific_target_strategy
        self.main_output_dir = main_output_dir
        self.obstacle_density = obstacle_density or ObstacleDensity.MEDIUM

        os.makedirs(output_dir, exist_ok=True)
        # 删除 mission 相关
        # self.mission = 0
        self.run_dir = None
        self.run_timestamp = None

        # 设置障碍物密度
        map_config.set_obstacle_density(self.obstacle_density)


def run_battle_batch(args):
    config, episode_indices = args

    # 确保每个进程都设置正确的障碍物密度
    map_config.set_obstacle_density(config.obstacle_density)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tracker_model = None
    target_model = None

    if config.tracker_type == "policy":
        tracker_model = Model(device, global_model=False)
        model_dict = torch.load(config.tracker_model_path, map_location=device)
        tracker_model.network.load_state_dict(model_dict['model'])
        tracker_model.network.eval()

    if config.target_type == "policy":
        target_model = Model(device, global_model=False)
        model_dict = torch.load(config.target_model_path, map_location=device)
        target_model.network.load_state_dict(model_dict['model'])
        target_model.network.eval()

    # PolicyManager is needed for ID lookup in build_critic_observation
    policy_manager = PolicyManager()

    batch_results = []

    for episode_idx in episode_indices:
        try:
            result = run_single_episode(config, episode_idx, tracker_model, target_model, device, policy_manager)
            batch_results.append(result)
        except Exception as e:
            print(f"Error in episode {episode_idx}: {e}")
            batch_results.append({
                "episode_id": episode_idx,
                "steps": 0,
                "reward": 0.0,
                "tracker_caught_target": False,
                "target_reached_exit": False,
                "tracker_type": config.tracker_type,
                "target_type": config.target_type,
                "tracker_strategy": config.specific_tracker_strategy or config.tracker_type,
                "target_strategy": config.specific_target_strategy or config.target_type
            })

    return batch_results


def run_single_episode(config, episode_idx, tracker_model, target_model, device, policy_manager, force_save_gif=False):
    # 确保障碍物密度设置正确
    map_config.set_obstacle_density(config.obstacle_density)
    
    # 删除 mission 参数，直接构造环境
    env = TrackingEnv()
    try:
        obs_result = env.reset()
        # 解包观测
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs = obs_result[0]
            if isinstance(obs, (tuple, list)) and len(obs) == 2:
                tracker_obs, target_obs = obs
            else:
                tracker_obs = target_obs = obs
        else:
            obs = obs_result
            if isinstance(obs, (tuple, list)) and len(obs) == 2:
                tracker_obs, target_obs = obs
            else:
                tracker_obs = target_obs = obs

        done = False
        episode_step = 0
        episode_reward = 0.0
        tracker_caught_target = False
        target_reached_exit = False

        save_gif = force_save_gif or (config.save_gif_freq > 0 and episode_idx % config.save_gif_freq == 0)
        episode_frames = []

        tracker_hidden = None
        target_hidden = None

        # 初始化策略
        tracker_strategy, tracker_policy_fn = _init_tracker_policy(config)
        if hasattr(tracker_policy_fn, 'reset'):
            tracker_policy_fn.reset()

        target_strategy, target_policy_obj = _init_target_policy(config)
        if hasattr(target_policy_obj, 'reset'):
            target_policy_obj.reset()

        if save_gif:
            try:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    episode_frames.append(frame)
            except Exception:
                save_gif = False

        with torch.no_grad():
            while not done and episode_step < EnvParameters.EPISODE_LEN:
                # === 获取特权状态（用于规则策略） ===
                privileged_state = env.get_privileged_state()
                
                # 构建观测
                tracker_actor_obs = np.asarray(tracker_obs, dtype=np.float32)
                target_actor_obs = np.asarray(target_obs, dtype=np.float32)
                
                # 修复：使用正确的对手策略名称
                target_opp_strategy = target_strategy if config.target_type != "policy" else None
                tracker_opp_strategy = tracker_strategy if config.tracker_type != "policy" else None
                
                tracker_critic_obs = build_critic_observation(
                    tracker_actor_obs, target_opp_strategy, policy_manager
                )
                
                target_critic_obs = build_critic_observation(
                    target_actor_obs, tracker_opp_strategy, policy_manager
                )

                # Tracker动作
                t_action = _get_tracker_action(
                    config, tracker_model, tracker_policy_fn,
                    tracker_strategy, tracker_actor_obs,
                    tracker_critic_obs, tracker_hidden, privileged_state
                )
                if isinstance(t_action, tuple) and len(t_action) == 3:
                    tracker_action, _, tracker_hidden = t_action
                else:
                    tracker_action = t_action

                # Target动作
                g_action = _get_target_action(
                    config, target_model, target_policy_obj,
                    target_strategy, target_actor_obs,
                    target_critic_obs, target_hidden, privileged_state
                )
                if isinstance(g_action, tuple) and len(g_action) == 3:
                    target_action, _, target_hidden = g_action
                else:
                    target_action = g_action

                # 确保传入 env.step 的是 2D 连续动作
                tracker_action = np.asarray(tracker_action, dtype=np.float32).reshape(2)
                target_action = np.asarray(target_action, dtype=np.float32).reshape(2)

                # 执行动作
                step_obs, reward, terminated, truncated, info = env.step(
                    (tracker_action, target_action)
                )
                done = terminated or truncated
                episode_reward += float(reward)
                episode_step += 1

                # 更新胜负状态
                if info.get('reason') == 'tracker_caught_target':
                    tracker_caught_target = True
                elif info.get('tracker_collision'):
                    target_reached_exit = True
                elif truncated:
                    target_reached_exit = True

                # 解析新观测
                if isinstance(step_obs, (tuple, list)) and len(step_obs) == 2:
                    tracker_obs, target_obs = step_obs
                else:
                    tracker_obs = target_obs = step_obs

                if save_gif:
                    try:
                        frame = env.render(mode='rgb_array')
                        if frame is not None:
                            episode_frames.append(frame)
                    except Exception:
                        save_gif = False

        if save_gif and len(episode_frames) > 1:
            try:
                # 优先使用 run_dir (具体对战文件夹)，其次 main_output_dir (评估根目录)，最后 output_dir
                save_dir = config.run_dir if config.run_dir else (config.main_output_dir or config.output_dir)
                gif_path = os.path.join(save_dir, f"episode_{episode_idx:03d}.gif")
                make_gif(episode_frames, gif_path, fps=EnvParameters.N_ACTIONS // 2)
            except Exception:
                pass

        return {
            "episode_id": episode_idx,
            "steps": episode_step,
            "reward": episode_reward,
            "tracker_caught_target": tracker_caught_target,
            "target_reached_exit": target_reached_exit,
            "tracker_type": config.tracker_type,
            "target_type": config.target_type,
            "tracker_strategy": tracker_strategy,
            "target_strategy": target_strategy
        }

    finally:
        env.close()


def _init_tracker_policy(config):
    """初始化tracker策略"""
    tracker_strategy = config.specific_tracker_strategy
    
    if config.tracker_type == "policy":
        return tracker_strategy or "policy", None
    elif config.tracker_type == "all":
        if tracker_strategy is None:
             raise ValueError("Tracker type is 'all' but no specific strategy provided")
        policy_cls = TRACKER_POLICY_REGISTRY.get(tracker_strategy, CBFTracker)
        return tracker_strategy, policy_cls()
    elif config.tracker_type in TRACKER_POLICY_REGISTRY:
        tracker_strategy = tracker_strategy or config.tracker_type
        policy_cls = TRACKER_POLICY_REGISTRY[config.tracker_type]
        return tracker_strategy, policy_cls()
    else:
        raise ValueError(f"Unsupported tracker type: {config.tracker_type}")


def _init_target_policy(config):
    """初始化target策略"""
    target_strategy = config.specific_target_strategy
    
    if config.target_type == "policy":
        return target_strategy or "policy", None
    elif config.target_type == "all":
        if target_strategy is None:
             raise ValueError("Target type is 'all' but no specific strategy provided")
        policy_cls = TARGET_POLICY_REGISTRY.get(target_strategy, GreedyTarget)
        return target_strategy, policy_cls()
    elif config.target_type in TARGET_POLICY_REGISTRY:
        target_strategy = target_strategy or config.target_type
        policy_cls = TARGET_POLICY_REGISTRY[config.target_type]
        return target_strategy, policy_cls()
    else:
        raise ValueError(f"Unsupported target type: {config.target_type}")


def _get_tracker_action(config, model, policy_fn, strategy, actor_obs, critic_obs, hidden, privileged_state=None):
    """获取tracker动作"""
    if config.tracker_type == "policy":
        # model.evaluate 返回的是 tanh 后的二维动作（归一化在 [-1,1]）
        eval_result = model.evaluate(actor_obs, critic_obs, hidden, greedy=True)
        if isinstance(eval_result, tuple):
            raw_action, pre_tanh, new_hidden, _, _ = eval_result
        else:
            raw_action, pre_tanh, new_hidden = eval_result, None, hidden
        # 显式转换为 numpy(2,)，避免后续堆叠出错
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
        return raw_action, pre_tanh, new_hidden
    else:
        if policy_fn is None:
            raise ValueError(f"No tracker policy function for strategy {strategy}")
        # 规则策略直接输出 env 控制动作（二维连续）
        if hasattr(policy_fn, 'get_action'):
            action = policy_fn.get_action(actor_obs, privileged_state)
        else:
            action = policy_fn(actor_obs, privileged_state)
        action = np.asarray(action, dtype=np.float32).reshape(2)
        return action, None, None


def _get_target_action(config, model, policy_obj, strategy, actor_obs, critic_obs, hidden, privileged_state=None):
    """获取target动作"""
    if config.target_type == "policy":
        eval_result = model.evaluate(actor_obs, critic_obs, hidden, greedy=True)
        if isinstance(eval_result, tuple):
            raw_action, pre_tanh, new_hidden, _, _ = eval_result
        else:
            raw_action, pre_tanh, new_hidden = eval_result, None, hidden
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
        return raw_action, pre_tanh, new_hidden
    else:
        if policy_obj is None:
            raise ValueError(f"No target policy object for strategy {strategy}")
        action = policy_obj.get_action(actor_obs)
        action = np.asarray(action, dtype=np.float32).reshape(2)
        return action, None, None

def build_critic_observation(actor_obs, opponent_strategy=None, policy_manager=None):
    actor_vec = np.asarray(actor_obs, dtype=np.float32).reshape(-1)
    if actor_vec.shape[0] < NetParameters.ACTOR_VECTOR_LEN:
        pad = NetParameters.ACTOR_VECTOR_LEN - actor_vec.shape[0]
        actor_vec = np.pad(actor_vec, (0, pad), mode="constant")
    elif actor_vec.shape[0] > NetParameters.ACTOR_VECTOR_LEN:
        actor_vec = actor_vec[:NetParameters.ACTOR_VECTOR_LEN]

    context = np.zeros(NetParameters.CONTEXT_LEN, dtype=np.float32)
    if opponent_strategy is not None and policy_manager is not None:
        policy_id = policy_manager.get_policy_id(opponent_strategy)
        if policy_id is not None and policy_id >= 0:
            context = get_opponent_id_one_hot(policy_id)

    return np.concatenate([actor_vec, context], axis=0)


def analyze_strategy_performance(df):
    strategy_stats = defaultdict(lambda: {
        'episodes': 0,
        'tracker_wins': 0,
        'target_wins': 0,
        'draws': 0,
        'avg_steps': 0.0,
        'avg_reward': 0.0
    })

    for _, row in df.iterrows():
        key = f"{row['tracker_strategy']}_vs_{row['target_strategy']}"
        stats = strategy_stats[key]

        stats['episodes'] += 1
        if row['tracker_caught_target']:
            stats['tracker_wins'] += 1
        elif row['target_reached_exit']:
            stats['target_wins'] += 1
        else:
            stats['draws'] += 1

        stats['avg_steps'] += row['steps']
        stats['avg_reward'] += row['reward']

    for key, stats in strategy_stats.items():
        if stats['episodes'] > 0:
            stats['avg_steps'] /= stats['episodes']
            stats['avg_reward'] /= stats['episodes']
            stats['tracker_win_rate'] = stats['tracker_wins'] / stats['episodes']
            stats['target_win_rate'] = stats['target_wins'] / stats['episodes']
            stats['draw_rate'] = stats['draws'] / stats['episodes']

    return dict(strategy_stats)


def run_strategy_evaluation(base_config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_evaluation_dir = os.path.join(
        base_config.output_dir,
        f"evaluation_{base_config.tracker_type}_vs_{base_config.target_type}_{base_config.obstacle_density}_{timestamp}"
    )
    os.makedirs(main_evaluation_dir, exist_ok=True)

    all_results = []
    all_summaries = []

    tracker_strategies = []
    target_strategies = []

    if base_config.tracker_type == "all":
        tracker_strategies = get_available_policies("tracker")
    else:
        tracker_strategies = [base_config.tracker_type]

    if base_config.target_type == "all":
        target_strategies = get_available_policies("target")
    else:
        target_strategies = [base_config.target_type]

    total_combinations = len(tracker_strategies) * len(target_strategies)
    print(f"将评估 {total_combinations} 种策略组合，每种组合 {base_config.episodes} 场对战")
    print(f"所有结果将保存在: {main_evaluation_dir}")

    combination_count = 0
    for tracker_strategy in tracker_strategies:
        for target_strategy in target_strategies:
            combination_count += 1
            print(f"\n进度 [{combination_count}/{total_combinations}] 评估策略组合: {tracker_strategy} vs {target_strategy}")

            strategy_output_dir = os.path.join(main_evaluation_dir, "individual_battles")
            os.makedirs(strategy_output_dir, exist_ok=True)

            config = BattleConfig(
                tracker_type=base_config.tracker_type,
                target_type=base_config.target_type,
                tracker_model_path=base_config.tracker_model_path,
                target_model_path=base_config.target_model_path,
                episodes=base_config.episodes,
                save_gif_freq=base_config.save_gif_freq,
                output_dir=strategy_output_dir,
                seed=base_config.seed + combination_count,
                state_space=base_config.state_space,
                specific_tracker_strategy=tracker_strategy if base_config.tracker_type == "all" else None,
                specific_target_strategy=target_strategy if base_config.target_type == "all" else None,
                main_output_dir=main_evaluation_dir,
                obstacle_density=base_config.obstacle_density)
            results, run_dir = run_battle(config, strategy_name=f"{tracker_strategy}_vs_{target_strategy}")

            if results is not None:
                all_results.extend(results.to_dict('records'))

                summary = {
                    'tracker_strategy': tracker_strategy,
                    'target_strategy': target_strategy,
                    'episodes': len(results),
                    'tracker_win_rate': results['tracker_caught_target'].mean(),
                    'target_win_rate': results['target_reached_exit'].mean(),
                    'draw_rate': 1.0 - results['tracker_caught_target'].mean() - results['target_reached_exit'].mean(),
                    'avg_steps': results['steps'].mean(),
                    'avg_reward': results['reward'].mean()
                }
                all_summaries.append(summary)

    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(main_evaluation_dir, "all_results.csv"), index=False)

        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(os.path.join(main_evaluation_dir, "strategy_summary.csv"), index=False)

        config_info = {
            'tracker_type': base_config.tracker_type,
            'target_type': base_config.target_type,
            'episodes_per_strategy': base_config.episodes,
            'total_combinations': total_combinations,
            'total_episodes': len(all_results),
            'evaluation_time': timestamp,
            'tracker_model_path': base_config.tracker_model_path,
            'target_model_path': base_config.target_model_path,
            'obstacle_density': base_config.obstacle_density
        }

        with open(os.path.join(main_evaluation_dir, "evaluation_config.json"), 'w') as f:
            json.dump(config_info, f, indent=2)

        print(f"\n=== 综合评估结果 ===")
        print(f"总共评估了 {total_combinations} 种策略组合")
        print(f"总计 {len(all_results)} 场对战")
        print(f"结果保存在: {main_evaluation_dir}")
        print("\n策略组合表现排名 (按Tracker胜率排序):")
        print(f"{'Tracker策略':<20} {'Target策略':<20} {'场次':<6} {'Tracker胜率':<12} {'Target胜率':<12} {'平均步数':<10}")
        print("-" * 90)

        summary_df_sorted = summary_df.sort_values('tracker_win_rate', ascending=False)
        for _, row in summary_df_sorted.iterrows():
            print(f"{row['tracker_strategy']:<20} {row['target_strategy']:<20} {row['episodes']:<6.0f} "
                  f"{row['tracker_win_rate']*100:<11.1f}% {row['target_win_rate']*100:<11.1f}% {row['avg_steps']:<10.1f}")

        return all_results_df, main_evaluation_dir

    return None, None


def run_battle(config, strategy_name=None):
    if strategy_name:
        print(f"运行对战: {strategy_name}, {config.episodes} 场")
    else:
        print(f"运行对战: {config.tracker_type} vs {config.target_type}, {config.episodes} 场")

    if strategy_name:
        run_name = f"battle_{strategy_name}"
    else:
        run_name = f"battle_{config.tracker_type}_vs_{config.target_type}"

    config.run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(config.run_dir, exist_ok=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    num_processes = min(multiprocessing.cpu_count() // 2, 6)
    batch_size = max(10, config.episodes // max(num_processes, 1))

    start_time = time.time()
    results = []

    batches = []
    for batch_start in range(0, config.episodes, batch_size):
        batch_end = min(batch_start + batch_size, config.episodes)
        batch_episodes = list(range(batch_start, batch_end))
        batches.append((config, batch_episodes))

    try:
        # 使用 spawn 上下文启动进程池，避免死锁
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_processes) as pool:
            batch_results = pool.map(run_battle_batch, batches)
        for batch in batch_results:
            results.extend(batch)
    except Exception as e:
        print(f"Error in parallel execution: {e}")
        return None, None

    if not results:
        print("No successful episodes completed!")
        return None, None

    df = pd.DataFrame(results)

    avg_steps = float(df["steps"].mean()) if len(df) > 0 else 0.0
    avg_reward = float(df["reward"].mean()) if len(df) > 0 else 0.0
    
    # === 修复：正确计算胜率和平局率 ===
    tracker_win_rate = float(df['tracker_caught_target'].mean()) if len(df) > 0 else 0.0
    target_win_rate = float(df['target_reached_exit'].mean()) if len(df) > 0 else 0.0
    draw_rate = 1.0 - tracker_win_rate - target_win_rate

    try:
        results_path = os.path.join(config.run_dir, "results.csv")
        df.to_csv(results_path, index=False)

        stats = {
            "total_episodes": len(df),
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "tracker_win_rate": tracker_win_rate,
            "target_win_rate": target_win_rate,
            "draw_rate": draw_rate,
            "tracker_strategy": config.specific_tracker_strategy or config.tracker_type,
            "target_strategy": config.specific_target_strategy or config.target_type
        }
        stats_path = os.path.join(config.run_dir, "stats.csv")
        pd.DataFrame([stats]).to_csv(stats_path, index=False)

    except Exception as e:
        print(f"Warning: Failed to save results: {e}")

    total_time = time.time() - start_time
    
    # === 修复：输出时也显示平局率 ===
    print(f"结果: 场次={len(df)}, 平均步数={avg_steps:.1f}, "
          f"Tracker胜率={tracker_win_rate*100:.1f}%, "
          f"Target胜率={target_win_rate*100:.1f}%, "
          f"平局率={draw_rate*100:.1f}%, "
          f"用时={total_time:.1f}s")

    return df, config.run_dir


if __name__ == "__main__":
    # 强制设置启动方法为 spawn
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Get available strategies dynamically
    available_tracker_strategies = get_available_policies("tracker")
    available_target_strategies = get_available_policies("target")
    
    parser = argparse.ArgumentParser(
        description='Agent Battle Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Strategies:
  Tracker: {', '.join(available_tracker_strategies)}
  Target:  {', '.join(available_target_strategies)}

Obstacle Density Levels:
  none   - 无障碍物（仅边界墙）
  sparse - 稀疏（巷战少量建筑/路障/立柱）
  medium - 中等（默认，较复杂的巷道网络）
  dense  - 密集（巷战高复杂度、瓶颈与死角更多）

Examples:
  # Policy vs Greedy target with no obstacles
  python evaluate.py --tracker policy --target Greedy --tracker_model ./models/tracker.pkl --obstacles none

  # GVF tracker vs all target strategies with dense obstacles
  python evaluate.py --tracker CBF --target all --episodes 100 --obstacles dense
        """
    )
    
    parser.add_argument('--tracker', type=str, default="policy",
                       choices=list(TRACKER_TYPE_CHOICES),
                       help=f'Tracker type: {", ".join(TRACKER_TYPE_CHOICES)}')
    parser.add_argument('--target', type=str, default="all",
                       choices=list(TARGET_TYPE_CHOICES),
                       help=f'Target type: {", ".join(TARGET_TYPE_CHOICES)}')
    
    parser.add_argument('--tracker_model', type=str, 
                       default='./models/TrackingEnv/lstm_ppo_11-24-16-33/best_model/tracker_net_checkpoint.pkl',
                       help='Path to tracker model (required when --tracker=policy)')
    parser.add_argument('--target_model', type=str, default=None,
                       help='Path to target model (required when --target=policy)')
    
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--save_gif_freq', type=int, default=5,
                       help='Save GIF every N episodes (0 to disable)')
    parser.add_argument('--output_dir', type=str, default='./scrimp_battle',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed')
    parser.add_argument('--state_space', type=str, default='vector',
                       help='State space representation')
    
    parser.add_argument('--obstacles', type=str, 
                       default=ObstacleDensity.DENSE,
                       choices=ObstacleDensity.ALL_LEVELS,
                       help='Obstacle density level (none/sparse/medium/dense)')
    
    args = parser.parse_args()
    
    if args.tracker == 'policy' and args.tracker_model is None:
        parser.error("--tracker_model is required when tracker is 'policy'")
    if args.target == 'policy' and args.target_model is None:
        parser.error("--target_model is required when target is 'policy'")

    config = BattleConfig(
        tracker_type=args.tracker,
        target_type=args.target,
        tracker_model_path=args.tracker_model,
        target_model_path=args.target_model,
        episodes=args.episodes,
        save_gif_freq=args.save_gif_freq,
        output_dir=args.output_dir,
        seed=args.seed,
        state_space=args.state_space,
        obstacle_density=args.obstacles
    )

    if config.tracker_type == "all" or config.target_type == "all":
        run_strategy_evaluation(config)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        main_dir = os.path.join(
            config.output_dir,
            f"single_battle_{config.tracker_type}_vs_{config.target_type}_{config.obstacle_density}_{timestamp}"
        )
        os.makedirs(main_dir, exist_ok=True)
        config.output_dir = main_dir
        run_battle(config)