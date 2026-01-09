import os
import sys
import argparse
import datetime
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, List
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use dummy video driver for headless evaluation
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import map_config
from map_config import EnvParameters, ObstacleDensity
from env import TrackingEnv
from mlp.util_mlp import make_gif, set_global_seeds
from star_hrl.model_star import StarModel
from star_hrl.alg_parameters_star import NetParameters
from rule_policies import TARGET_POLICY_REGISTRY

DEFAULT_TARGET_POLICY = "all"

@dataclass
class EvalConfig:
    model_path: str
    target_policy: str
    episodes: int
    save_gif_freq: int
    output_dir: str
    seed: int
    obstacle_density: str
    enable_safety_layer: bool
    greedy: bool = True

def _get_target_policy(name: str):
    if name not in TARGET_POLICY_REGISTRY:
        raise ValueError(f"Unknown target policy: {name}")
    return TARGET_POLICY_REGISTRY[name]()

def load_star_model(path: str, device: torch.device) -> StarModel:
    model = StarModel(device, global_model=False)
    # Check if path is a file or directory
    if os.path.isdir(path):
        # try to find latest or best
        if os.path.exists(os.path.join(path, "best_model", "checkpoint.pth")):
             path = os.path.join(path, "best_model", "checkpoint.pth")
        elif os.path.exists(os.path.join(path, "latest_model", "checkpoint.pth")):
             path = os.path.join(path, "latest_model", "checkpoint.pth")
    
    checkpoint = torch.load(path, map_location=device)
    if 'network' in checkpoint:
        model.network.load_state_dict(checkpoint['network'])
    elif 'model' in checkpoint:
        model.set_weights(checkpoint['model'])
    else:
        # Assuming direct state dict
        model.network.load_state_dict(checkpoint)
    
    model.network.eval()
    return model

def run_single_episode(model: StarModel,
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
        tracker_obs = target_obs = obs_result

    done = False
    episode_step = 0
    episode_reward = 0.0
    tracker_caught_target = False
    target_reached_exit = False
    collision = False

    should_record = cfg.save_gif_freq > 0 and episode_idx % cfg.save_gif_freq == 0
    frames = []
    
    # Track high weights for analysis
    high_weights_history = []
    
    # 碰撞信息用于可视化
    collision_info = None

    if hasattr(target_policy, "reset"):
        target_policy.reset()

    try:
        with torch.no_grad():
            while not done and episode_step < EnvParameters.EPISODE_LEN:
                tracker_actor_obs = np.asarray(tracker_obs, dtype=np.float32)
                target_actor_obs = np.asarray(target_obs, dtype=np.float32)

                # Critic obs for evaluation is just concatenation
                tracker_critic_obs = np.concatenate([tracker_actor_obs, target_actor_obs])
                
                agent_action, _, _, _, high_weights = model.evaluate(tracker_actor_obs, tracker_critic_obs, greedy=cfg.greedy)
                high_weights_history.append(high_weights)

                if hasattr(target_policy, "get_action"):
                    target_action = target_policy.get_action(target_actor_obs)
                else:
                    target_action = target_policy(target_actor_obs)
                target_action = np.asarray(target_action, dtype=np.float32).reshape(2)

                step_obs, reward, terminated, truncated, info = env.step((agent_action, target_action))
                done = terminated or truncated
                episode_reward += float(reward)
                episode_step += 1

                if info.get("reason") == "tracker_caught_target":
                    tracker_caught_target = True
                elif info.get("tracker_collision"):
                    target_reached_exit = True
                    collision = True
                    
                    # 当发生碰撞时，检测被撞障碍物和碰撞点
                    if should_record:
                        import env_lib
                        tracker_cx = env.tracker['x'] + map_config.pixel_size * 0.5
                        tracker_cy = env.tracker['y'] + map_config.pixel_size * 0.5
                        agent_radius = float(getattr(map_config, 'agent_radius', map_config.pixel_size * 0.5))
                        collision_info = env_lib.find_colliding_obstacle(tracker_cx, tracker_cy, agent_radius)
                        
                elif truncated:
                    target_reached_exit = True

                if isinstance(step_obs, (tuple, list)) and len(step_obs) == 2:
                    tracker_obs, target_obs = step_obs
                else:
                    tracker_obs = target_obs = step_obs

                if should_record:
                    # 使用碰撞信息渲染帧
                    frame = env.render(mode="rgb_array", collision_info=collision_info)
                    if frame is not None:
                        frames.append(frame)
                    
                    # 如果发生碰撞，添加停留帧让用户看清碰撞位置
                    if collision_info and collision_info.get('collision'):
                        freeze_count = getattr(map_config, 'COLLISION_FREEZE_FRAMES', 30)
                        for _ in range(freeze_count):
                            frames.append(frame.copy())
                        
    finally:
        env.close()

    if should_record and len(frames) > 1:
        gif_path = os.path.join(run_dir, f"episode_{episode_idx:04d}.gif")
        make_gif(frames, gif_path, fps=EnvParameters.N_ACTIONS // 2)
        
    mean_weights = np.mean(high_weights_history, axis=0) if high_weights_history else [0,0]

    return {
        "target_strategy": target_strategy_name or cfg.target_policy,
        "episode": episode_idx,
        "steps": episode_step,
        "reward": episode_reward,
        "tracker_win": tracker_caught_target,
        "target_win": target_reached_exit,
        "collision": collision,
        "w_track": mean_weights[0],
        "w_safe": mean_weights[1]
    }

def run_strategy_evaluation(cfg: EvalConfig, target_strategies: List[str]):
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    main_evaluation_dir = os.path.join(cfg.output_dir, f"star_eval_{timestamp}")
    os.makedirs(main_evaluation_dir, exist_ok=True)

    all_results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_star_model(cfg.model_path, device)

    print(f"Evaluating against {len(target_strategies)} target strategies")
    print(f"Obstacle density: {cfg.obstacle_density}")
    
    start_time = time.time()
    
    for target_strategy in target_strategies:
        print(f"\nEvaluating against: {target_strategy}")
        target_policy = _get_target_policy(target_strategy)
        
        strategy_dir = os.path.join(main_evaluation_dir, target_strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        strategy_results = []
        for ep in range(cfg.episodes):
            result = run_single_episode(model, target_policy, device, ep, cfg, strategy_dir, target_strategy)
            strategy_results.append(result)
            all_results.append(result)
            
        df = pd.DataFrame(strategy_results)
        win_rate = df['tracker_win'].mean()
        coll_rate = df['collision'].mean()
        avg_rew = df['reward'].mean()
        w_track_avg = df['w_track'].mean()
        
        print(f"  Win: {win_rate*100:.1f}% | Coll: {coll_rate*100:.1f}% | Rew: {avg_rew:.2f} | W_track: {w_track_avg:.2f}")

    all_df = pd.DataFrame(all_results)
    all_df.to_csv(os.path.join(main_evaluation_dir, "results.csv"), index=False)
    
    print(f"\nResults saved to {main_evaluation_dir}")
    return all_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate Star HRL Model")
    parser.add_argument("--model", required=True, help="Path to Star HRL checkpoint")
    parser.add_argument("--target_policy", default=DEFAULT_TARGET_POLICY, help="Target policy (or 'all')")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save_gif_freq", type=int, default=30)
    parser.add_argument("--output_dir", default="./battles/star")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--obstacles", default=ObstacleDensity.DENSE, choices=ObstacleDensity.ALL_LEVELS)
    parser.add_argument("--safety-layer", action="store_true")
    
    args = parser.parse_args()

    # Get available target policies
    available_target_policies = list(TARGET_POLICY_REGISTRY.keys())
    
    if args.target_policy == "all":
        target_strategies = available_target_policies
    elif args.target_policy in available_target_policies:
        target_strategies = [args.target_policy]
    else:
        raise ValueError(f"Unknown target policy: {args.target_policy}")

    cfg = EvalConfig(
        model_path=args.model,
        target_policy=args.target_policy,
        episodes=args.episodes,
        save_gif_freq=args.save_gif_freq,
        output_dir=args.output_dir,
        seed=args.seed,
        obstacle_density=args.obstacles,
        enable_safety_layer=args.safety_layer,
    )
    
    set_global_seeds(cfg.seed)
    
    run_strategy_evaluation(cfg, target_strategies)

if __name__ == "__main__":
    main()
