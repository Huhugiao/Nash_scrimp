import os
import sys
import numpy as np
import torch
import ray
import argparse
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from targetmaker.target_alg_parameters import TargetTrainingParameters, TargetNetParameters, TargetSetupParameters, TargetRecordingParameters
from targetmaker.target_model import TargetPPO
from targetmaker.target_runner import TargetRunner
from mlp.util_mlp import set_global_seeds, make_gif

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--style", type=str, default="taunt", choices=["all", "survival", "stealth", "taunt"], help="Train a specific target style")
    args = parser.parse_args()
    
    if args.style != "all":
        TargetTrainingParameters.STYLES = [args.style]
        if not args.name:
             # Auto-name based on style if name not provided
             args.name = f"{args.style}_ppo"

    if args.name:
        TargetRecordingParameters.EXPERIMENT_NAME = args.name
        # Append TIME to ensure unique folders even with same name argument
        TargetRecordingParameters.SUMMARY_PATH = f'./target_models/{args.name}{TargetRecordingParameters.TIME}'
        TargetRecordingParameters.MODEL_PATH = f'./target_models/{args.name}{TargetRecordingParameters.TIME}'
        TargetRecordingParameters.GIFS_PATH = f'./target_models/{args.name}{TargetRecordingParameters.TIME}/gifs'

    
    if not ray.is_initialized():
        ray.init(num_gpus=TargetSetupParameters.NUM_GPU)
        
    print(f"TargetMaker PPO Multi-Agent Training Started!")
    
    set_global_seeds(TargetSetupParameters.SEED)
    global_device = torch.device('cuda') if TargetSetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    
    # Init 3 Agents
    agents = {}
    for style in TargetTrainingParameters.STYLES:
        agents[style] = TargetPPO(global_device)
        
    curr_steps = 0
    curr_steps = 0
    next_eval_threshold = TargetRecordingParameters.EVAL_INTERVAL
    next_gif_threshold = TargetRecordingParameters.GIF_INTERVAL
    os.makedirs(TargetRecordingParameters.MODEL_PATH, exist_ok=True)
    os.makedirs(TargetRecordingParameters.GIFS_PATH, exist_ok=True)
    
    # Best Reward Tracking
    best_rewards = {style: -float('inf') for style in TargetTrainingParameters.STYLES}

    
    # Init 3 Writers for clean "One Run Per Style" visualization
    writers = {}
    for style in TargetTrainingParameters.STYLES:
        writers[style] = SummaryWriter(log_dir=f"{TargetRecordingParameters.SUMMARY_PATH}/{style}")
    
    # Round-robin index for assigning the extra runner(s) if runners != styles
    rr_idx = 0
    
    # Init Runners
    runners = [TargetRunner.remote(i) for i in range(TargetTrainingParameters.N_ENVS)]
    
    try:
        while curr_steps < TargetTrainingParameters.N_MAX_STEPS:
            # 1. Distribute Tasks
            # We want to train all 3 styles. We have 4 runners.
            # Strategy: Always run 1 of each style, and the 4th runner rotates through them.
            
            style_assignments = [] 
            # First 3 runners get distinct styles
            for i in range(len(TargetTrainingParameters.STYLES)):
                style_assignments.append(TargetTrainingParameters.STYLES[i])
            
            # Remaining runners get round-robin style
            for _ in range(TargetTrainingParameters.N_ENVS - len(TargetTrainingParameters.STYLES)):
                style_assignments.append(TargetTrainingParameters.STYLES[rr_idx])
                rr_idx = (rr_idx + 1) % len(TargetTrainingParameters.STYLES)
            
            # Start Rollouts
            jobs = []
            for i, r_handle in enumerate(runners):
                 style = style_assignments[i]
                 w = agents[style].get_weights()
                 # Weights must be on CPU for Ray serialization usually, but state_dict is fine.
                 jobs.append(r_handle.run_rollout.remote(w, style))
            
            results = ray.get(jobs)
            
            # 2. Process Data & Update
            # We need to buffer data per style because multiple runners might run same style
            style_data_buffer = {s: [] for s in TargetTrainingParameters.STYLES}
            style_metrics = {s: {'r': [], 'succ': []} for s in TargetTrainingParameters.STYLES}
            
            total_steps_this_iter = 0
            
            for res, style in zip(results, style_assignments):
                 # Merge data arrays? Or just list of dicts? PPO update usually takes one big array.
                 # Let's concatenate arrays later.
                 style_data_buffer[style].append(res)
                 total_steps_this_iter += len(res['data']['rewards'])
                 
                 # Metrics
                 if res['metrics']['r']:
                     style_metrics[style]['r'].extend(res['metrics']['r'])
                     style_metrics[style]['succ'].extend(res['metrics']['succ'])

            # Log Online Metrics
            # Use separate writers so they appear as 3 curves on the "Online/Reward" chart
            scalars_r = {} # for print
            for style in TargetTrainingParameters.STYLES:
                if style_metrics[style]['r']:
                    mean_r = np.mean(style_metrics[style]['r'])
                    mean_s = np.mean(style_metrics[style]['succ'])
                    
                    writers[style].add_scalar("Online/Reward", mean_r, curr_steps)
                    writers[style].add_scalar("Online/Success", mean_s, curr_steps)
                    scalars_r[style] = mean_r
            
            curr_steps += total_steps_this_iter
            
            # update
            scalars_loss = {}
            
            for style in TargetTrainingParameters.STYLES:
                 batches = style_data_buffer[style]
                 if not batches: continue
                 
                 # Concatenate fields
                 concat_data = {}
                 keys = ['actor_obs', 'critic_obs', 'actions', 'logp']
                 for k in keys:
                      concat_data[k] = np.concatenate([b['data'][k] for b in batches])
                 
                 concat_data['returns'] = np.concatenate([b['returns'] for b in batches])
                 concat_data['adv'] = np.concatenate([b['adv'] for b in batches])
                 
                 loss_info = agents[style].train(concat_data)
                 
                 writers[style].add_scalar("Train/Loss_PG", loss_info['pg_loss'], curr_steps)
                 writers[style].add_scalar("Train/Entropy", loss_info['entropy'], curr_steps)
                 writers[style].add_scalar("Train/Loss_VF", loss_info['vf_loss'], curr_steps)
                 scalars_loss[style] = loss_info['pg_loss']
            
            print(f"Step {curr_steps} | R: {[f'{k}:{v:.1f}' for k,v in scalars_r.items()]}")

            # 3. Eval
            if curr_steps >= next_eval_threshold:
                 print(f"--- Evaluating ({curr_steps}) ---")
                 
                 # Check if we should save GIF
                 record_gif = (curr_steps >= next_gif_threshold)
                 
                 # Use first runner
                 r_handle = runners[0]
                 
                 eval_results = {}
                 for style in TargetTrainingParameters.STYLES:
                      w = agents[style].get_weights()
                      metrics = ray.get(r_handle.evaluate_policy.remote(w, 10, style=style, record_gif=record_gif))
                      
                      writers[style].add_scalar("Eval/Reward", metrics['r'], curr_steps)
                      writers[style].add_scalar("Eval/Success", metrics['succ'], curr_steps)
                      eval_results[style] = metrics['r']
                      
                      if record_gif and metrics.get('frames'):
                          gif_name = f"{style}_{curr_steps}.gif"
                          path = f"{TargetRecordingParameters.GIFS_PATH}/{gif_name}"
                          try:
                              make_gif(metrics['frames'], path)
                          except Exception as e:
                              print(f"Failed to save GIF: {e}")
                              
                      # Save Latest
                      path = f"{TargetRecordingParameters.MODEL_PATH}/{style}.pth"
                      torch.save(agents[style].get_weights(), path) 
                      # PPO wrapper has save/load? Let's check target_model.py or use torch.save(agent.get_weights()...)
                      # Actually agents[style] is TargetPPO instance. It likely has 'save' method?
                      # If not, let's look at get_weights.
                      # Usually: weights = agents[style].get_weights()...
                      # Let's trust user has saving logic or simple state_dict.
                      # "agents[style].save(path)" if exists.
                      # Let's check target_model.py previously viewed.
                      # TargetPPO has get_weights returning {k: v.cpu()}.
                      # Let's save that dictionary.
                      torch.save(agents[style].get_weights(), path)

                      # Save Best
                      if metrics['r'] > best_rewards[style]:
                          best_rewards[style] = metrics['r']
                          best_path = f"{TargetRecordingParameters.MODEL_PATH}/{style}_best.pth"
                          torch.save(agents[style].get_weights(), best_path)
                          print(f"[{style}] New Best Reward: {metrics['r']:.2f} -> Saved best model.")
                              
                 print(f"Eval R: {[f'{k}:{v:.2f}' for k,v in eval_results.items()]}")
                
                 next_eval_threshold += TargetRecordingParameters.EVAL_INTERVAL
                 if record_gif:
                      next_gif_threshold += TargetRecordingParameters.GIF_INTERVAL

    except KeyboardInterrupt:
        print("Interrupted.")

if __name__ == "__main__":
    main()
