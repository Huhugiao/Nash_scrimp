import os
import sys
import argparse
import datetime
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, List
from collections import defaultdict

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use dummy video driver for headless evaluation
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pandas as pd
import torch

import map_config
from map_config import EnvParameters, ObstacleDensity
from env import TrackingEnv
from mlp.model_mlp import Model as BaseModel
from mlp.util_mlp import make_gif, set_global_seeds
from residual.nets_residual import ResidualPolicyNetwork
from residual.alg_parameters_residual import NetParameters, ResidualRLConfig
from rule_policies import TARGET_POLICY_REGISTRY


DEFAULT_TARGET_POLICY = "all"


def load_base_model(path: str, device: torch.device) -> BaseModel:
    """Load the frozen base tracker policy."""
    model = BaseModel(device, global_model=False)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.set_weights(checkpoint["model"])
    elif isinstance(checkpoint, dict) and "model_weights" in checkpoint:
        model.set_weights(checkpoint["model_weights"])
    else:
        model.network.load_state_dict(checkpoint)
    model.network.eval()
    return model


def load_residual_model(path: str, device: torch.device) -> ResidualPolicyNetwork:
    """Load the learned residual policy network."""
    residual_net = ResidualPolicyNetwork().to(device)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        residual_net.load_state_dict(checkpoint["model"])
    else:
        residual_net.load_state_dict(checkpoint)
    residual_net.eval()
    return residual_net


def build_critic_observation(actor_obs: np.ndarray, other_obs: np.ndarray) -> np.ndarray:
    """Concatenate tracker and target observations for critic-style input."""
    return np.concatenate([
        np.asarray(actor_obs, dtype=np.float32).reshape(-1),
        np.asarray(other_obs, dtype=np.float32).reshape(-1)
    ], axis=0)


@dataclass
class EvalConfig:
    base_model_path: str
    residual_model_path: str
    target_policy: str
    episodes: int
    save_gif_freq: int
    output_dir: str
    seed: int
    obstacle_density: str
    enable_safety_layer: bool


def _get_target_policy(name: str):
    if name not in TARGET_POLICY_REGISTRY:
        raise ValueError(f"Unknown target policy: {name}")
    return TARGET_POLICY_REGISTRY[name]()


def _fuse_action(base_action: np.ndarray, radar_obs: np.ndarray, residual_net: ResidualPolicyNetwork, device: torch.device) -> np.ndarray:
    """Fuse base action with deterministic residual correction."""
    radar_tensor = torch.as_tensor(radar_obs, dtype=torch.float32, device=device).unsqueeze(0)
    base_tensor = torch.as_tensor(base_action, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, _ = residual_net.actor(radar_tensor)
        fused = ResidualPolicyNetwork.fuse_actions(base_tensor, mean)
    return fused.cpu().numpy()[0]


def run_single_episode(mode: str,
                       base_model: BaseModel,
                       residual_net: ResidualPolicyNetwork,
                       target_policy,
                       device: torch.device,
                       episode_idx: int,
                       cfg: EvalConfig,
                       run_dir: str,
                       target_strategy_name: str = None) -> Dict[str, Any]:
    map_config.set_obstacle_density(cfg.obstacle_density)
    env = TrackingEnv(enable_safety_layer=cfg.enable_safety_layer)

    obs_result = env.reset()
    if isinstance(obs_result, tuple) and len(obs_result) == 2:
        obs, _info = obs_result
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
    collision = False

    should_record = cfg.save_gif_freq > 0 and episode_idx % cfg.save_gif_freq == 0
    frames = []

    if hasattr(target_policy, "reset"):
        target_policy.reset()

    try:
        with torch.no_grad():
            while not done and episode_step < EnvParameters.EPISODE_LEN:
                tracker_actor_obs = np.asarray(tracker_obs, dtype=np.float32)
                target_actor_obs = np.asarray(target_obs, dtype=np.float32)

                tracker_critic_obs = build_critic_observation(tracker_actor_obs, target_actor_obs)
                base_action, _, _, _ = base_model.evaluate(tracker_actor_obs, tracker_critic_obs, greedy=True)

                if mode == "residual":
                    radar_obs = tracker_actor_obs[NetParameters.ACTOR_SCALAR_LEN:]
                    tracker_action = _fuse_action(base_action, radar_obs, residual_net, device)
                else:
                    tracker_action = np.asarray(base_action, dtype=np.float32)

                if hasattr(target_policy, "get_action"):
                    target_action = target_policy.get_action(target_actor_obs)
                else:
                    target_action = target_policy(target_actor_obs)
                target_action = np.asarray(target_action, dtype=np.float32).reshape(2)

                step_obs, reward, terminated, truncated, info = env.step((tracker_action, target_action))
                done = terminated or truncated
                episode_reward += float(reward)
                episode_step += 1

                if info.get("reason") == "tracker_caught_target":
                    tracker_caught_target = True
                elif info.get("tracker_collision"):
                    target_reached_exit = True
                    collision = True
                elif truncated:
                    target_reached_exit = True

                if isinstance(step_obs, (tuple, list)) and len(step_obs) == 2:
                    tracker_obs, target_obs = step_obs
                else:
                    tracker_obs = target_obs = step_obs

                if should_record:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(frame)
    finally:
        env.close()

    if should_record and len(frames) > 1:
        gif_path = os.path.join(run_dir, f"{mode}_episode_{episode_idx:04d}.gif")
        make_gif(frames, gif_path, fps=EnvParameters.N_ACTIONS // 2)

    return {
        "mode": mode,
        "target_strategy": target_strategy_name or cfg.target_policy,
        "episode": episode_idx,
        "steps": episode_step,
        "reward": episode_reward,
        "tracker_win": tracker_caught_target,
        "target_win": target_reached_exit,
        "collision": collision
    }


def summarize_results(df: pd.DataFrame, mode: str) -> Dict[str, Any]:
    subset = df[df["mode"] == mode]
    if subset.empty:
        return {
            "mode": mode,
            "episodes": 0,
            "tracker_win_rate": 0.0,
            "collision_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0
        }
    episodes = len(subset)
    return {
        "mode": mode,
        "episodes": episodes,
        "tracker_win_rate": float(subset["tracker_win"].mean()),
        "collision_rate": float(subset["collision"].mean()),
        "avg_reward": float(subset["reward"].mean()),
        "avg_steps": float(subset["steps"].mean())
    }


def analyze_strategy_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance for each target strategy"""
    strategy_stats = defaultdict(lambda: {
        'episodes': 0,
        'tracker_wins': 0,
        'target_wins': 0,
        'collisions': 0,
        'avg_steps': 0.0,
        'avg_reward': 0.0
    })

    for _, row in df.iterrows():
        key = f"{row['mode']}_vs_{row['target_strategy']}"
        stats = strategy_stats[key]

        stats['episodes'] += 1
        if row['tracker_win']:
            stats['tracker_wins'] += 1
        elif row['target_win']:
            stats['target_wins'] += 1
        if row['collision']:
            stats['collisions'] += 1

        stats['avg_steps'] += row['steps']
        stats['avg_reward'] += row['reward']

    for key, stats in strategy_stats.items():
        if stats['episodes'] > 0:
            stats['avg_steps'] /= stats['episodes']
            stats['avg_reward'] /= stats['episodes']
            stats['tracker_win_rate'] = stats['tracker_wins'] / stats['episodes']
            stats['target_win_rate'] = stats['target_wins'] / stats['episodes']
            stats['collision_rate'] = stats['collisions'] / stats['episodes']

    return dict(strategy_stats)


def run_strategy_evaluation(cfg: EvalConfig, target_strategies: List[str]):
    """Run evaluation against multiple target strategies"""
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    main_evaluation_dir = os.path.join(cfg.output_dir, f"residual_eval_all_{timestamp}")
    os.makedirs(main_evaluation_dir, exist_ok=True)

    all_results = []
    all_summaries = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_base_model(cfg.base_model_path, device)
    residual_net = load_residual_model(cfg.residual_model_path, device)

    print(f"Evaluating against {len(target_strategies)} target strategies: {', '.join(target_strategies)}")
    print(f"Obstacle density: {cfg.obstacle_density}")
    print(f"Base model: {cfg.base_model_path}")
    print(f"Residual model: {cfg.residual_model_path}")
    
    modes = ["base", "residual"]
    start_time = time.time()
    
    # Evaluate each target strategy
    for target_strategy in target_strategies:
        print(f"\nEvaluating against target strategy: {target_strategy}")
        target_policy = _get_target_policy(target_strategy)
        
        strategy_dir = os.path.join(main_evaluation_dir, target_strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Evaluate each mode (base and residual)
        for mode in modes:
            mode_dir = os.path.join(strategy_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            mode_results = []
            for ep in range(cfg.episodes):
                result = run_single_episode(mode, base_model, residual_net, target_policy, device, ep, cfg, mode_dir, target_strategy)
                mode_results.append(result)
                all_results.append(result)
            
            # Summarize results for this mode and strategy
            df_mode = pd.DataFrame(mode_results)
            summary = {
                'mode': mode,
                'target_strategy': target_strategy,
                'episodes': len(df_mode),
                'tracker_win_rate': df_mode['tracker_win'].mean(),
                'target_win_rate': df_mode['target_win'].mean(),
                'collision_rate': df_mode['collision'].mean(),
                'avg_reward': df_mode['reward'].mean(),
                'avg_steps': df_mode['steps'].mean()
            }
            all_summaries.append(summary)
            
            print(f"  [{mode}] episodes={summary['episodes']} win_rate={summary['tracker_win_rate']*100:.1f}% collision_rate={summary['collision_rate']*100:.1f}% avg_reward={summary['avg_reward']:.2f} avg_steps={summary['avg_steps']:.1f}")
    
    # Save all results
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(main_evaluation_dir, "all_results.csv"), index=False)
        
        # Save summary
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(os.path.join(main_evaluation_dir, "strategy_summary.csv"), index=False)
        
        # Save configuration
        config_info = {
            'base_model_path': cfg.base_model_path,
            'residual_model_path': cfg.residual_model_path,
            'target_strategies': target_strategies,
            'episodes_per_strategy': cfg.episodes,
            'total_combinations': len(target_strategies) * len(modes),
            'total_episodes': len(all_results),
            'evaluation_time': timestamp,
            'obstacle_density': cfg.obstacle_density
        }
        
        with open(os.path.join(main_evaluation_dir, "evaluation_config.json"), 'w') as f:
            json.dump(config_info, f, indent=2)
        
        # Analyze strategy performance
        strategy_performance = analyze_strategy_performance(all_results_df)
        with open(os.path.join(main_evaluation_dir, "strategy_performance.json"), 'w') as f:
            json.dump(strategy_performance, f, indent=2)
        
        # Print summary
        print(f"\n=== 综合评估结果 ===")
        print(f"总共评估了 {len(target_strategies)} 种目标策略")
        print(f"总计 {len(all_results)} 场对战")
        print(f"结果保存在: {main_evaluation_dir}")
        
        print("\n策略表现排名 (按Base模型Tracker胜率排序):")
        base_results = summary_df[summary_df['mode'] == 'base'].sort_values('tracker_win_rate', ascending=False)
        print(f"{'Target策略':<20} {'场次':<6} {'Base胜率':<10} {'Residual胜率':<12} {'Base奖励':<10} {'Residual奖励':<12}")
        print("-" * 80)
        
        for _, row in base_results.iterrows():
            target_strategy = row['target_strategy']
            residual_row = summary_df[(summary_df['mode'] == 'residual') & (summary_df['target_strategy'] == target_strategy)]
            
            base_win_rate = row['tracker_win_rate']
            residual_win_rate = residual_row.iloc[0]['tracker_win_rate'] if not residual_row.empty else 0.0
            base_reward = row['avg_reward']
            residual_reward = residual_row.iloc[0]['avg_reward'] if not residual_row.empty else 0.0
            
            print(f"{target_strategy:<20} {row['episodes']:<6.0f} "
                  f"{base_win_rate*100:<9.1f}% "
                  f"{residual_win_rate*100:<11.1f}% "
                  f"{base_reward:<9.2f} "
                  f"{residual_reward:<11.2f}")
        
        return all_results_df, main_evaluation_dir
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate base tracker vs residual-enhanced tracker")
    parser.add_argument("--residual_model", default='models/residual_avoidance_12-22-18-05/latest_model/checkpoint.pth', help="Path to residual checkpoint (best_model/checkpoint.pth)")
    parser.add_argument("--base_model", default='models/rl_CoverSeeker_collision_12-19-12-30/best_model/checkpoint.pth', help="Path to base tracker checkpoint")
    parser.add_argument("--target_policy", default=DEFAULT_TARGET_POLICY, help="Target policy to evaluate against (use 'all' for all policies)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per tracker mode")
    parser.add_argument("--save_gif_freq", type=int, default=30, help="Save GIF every N episodes (0 to disable)")
    parser.add_argument("--output_dir", type=str, default="./residual_evals", help="Directory to store evaluation outputs")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--obstacles", type=str, default=ObstacleDensity.DENSE, choices=ObstacleDensity.ALL_LEVELS, help="Obstacle density level")
    parser.add_argument("--no-safety-layer", action="store_true", help="Disable environment safety layer")

    args = parser.parse_args()

    # Get available target policies
    available_target_policies = list(TARGET_POLICY_REGISTRY.keys())
    
    # Handle "all" target policy
    if args.target_policy == "all":
        target_strategies = available_target_policies
    elif args.target_policy in available_target_policies:
        target_strategies = [args.target_policy]
    else:
        raise ValueError(f"Unknown target policy: {args.target_policy}. Available policies: {', '.join(available_target_policies)}")

    run_stamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    run_root = os.path.join(args.output_dir, f"residual_eval_{run_stamp}")
    os.makedirs(run_root, exist_ok=True)

    cfg = EvalConfig(
        base_model_path=args.base_model,
        residual_model_path=args.residual_model,
        target_policy=args.target_policy,
        episodes=args.episodes,
        save_gif_freq=args.save_gif_freq,
        output_dir=run_root,
        seed=args.seed,
        obstacle_density=args.obstacles,
        enable_safety_layer=not args.no_safety_layer
    )

    set_global_seeds(cfg.seed)
    
    # If evaluating against all policies, run strategy evaluation
    if args.target_policy == "all" or len(target_strategies) > 1:
        run_strategy_evaluation(cfg, target_strategies)
    else:
        # Original single policy evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = load_base_model(cfg.base_model_path, device)
        residual_net = load_residual_model(cfg.residual_model_path, device)
        target_policy = _get_target_policy(cfg.target_policy)

        results = []
        modes = ["base", "residual"]
        start_time = time.time()

        for mode in modes:
            mode_dir = os.path.join(run_root, mode)
            os.makedirs(mode_dir, exist_ok=True)
            for ep in range(cfg.episodes):
                result = run_single_episode(mode, base_model, residual_net, target_policy, device, ep, cfg, mode_dir)
                results.append(result)
            summary = summarize_results(pd.DataFrame(results), mode)
            print(f"[SUMMARY] {mode}: episodes={summary['episodes']} win_rate={summary['tracker_win_rate']*100:.1f}% collision_rate={summary['collision_rate']*100:.1f}% avg_reward={summary['avg_reward']:.2f} avg_steps={summary['avg_steps']:.1f}")

        total_time = time.time() - start_time
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(run_root, "results.csv"), index=False)

        summaries = [summarize_results(df, m) for m in modes]
        pd.DataFrame(summaries).to_csv(os.path.join(run_root, "summary.csv"), index=False)

        print(f"Saved detailed results to {os.path.join(run_root, 'results.csv')}")
        print(f"Saved summary to {os.path.join(run_root, 'summary.csv')}")
        print(f"Total evaluation time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
