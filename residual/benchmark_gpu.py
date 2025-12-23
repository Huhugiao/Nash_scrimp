#!/usr/bin/env python
"""
Benchmark GPU vs CPU for residual network training.
Tests different combinations of USE_GPU_LOCAL and USE_GPU_GLOBAL.
"""
import os
import sys
import time
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from residual.model_residual import ResidualModel
from residual.alg_parameters_residual import NetParameters

def benchmark_training(use_gpu_global: bool, use_gpu_local: bool, n_iters: int = 100, batch_size: int = 256):
    """Benchmark training speed for given GPU settings."""
    
    device = torch.device('cuda' if use_gpu_global and torch.cuda.is_available() else 'cpu')
    print(f"\n=== GPU_GLOBAL={use_gpu_global}, GPU_LOCAL={use_gpu_local} ===")
    print(f"Device: {device}")
    
    # Create model
    model = ResidualModel(device, global_model=True)
    
    # Create dummy data
    radar_obs = np.random.randn(batch_size, NetParameters.RADAR_DIM).astype(np.float32)
    velocity_obs = np.random.randn(batch_size, NetParameters.VELOCITY_DIM).astype(np.float32)
    base_actions = np.random.randn(batch_size, NetParameters.ACTION_DIM).astype(np.float32)
    returns = np.random.randn(batch_size).astype(np.float32)
    values = np.random.randn(batch_size).astype(np.float32)
    actions = np.random.randn(batch_size, NetParameters.ACTION_DIM).astype(np.float32)
    old_log_probs = np.random.randn(batch_size).astype(np.float32)
    mask = np.ones(batch_size, dtype=np.float32)
    
    # Warmup
    for _ in range(5):
        model.train(
            radar_obs=radar_obs,
            velocity_obs=velocity_obs,
            base_actions=base_actions,
            returns=returns,
            values=values,
            actions=actions,
            old_log_probs=old_log_probs,
            mask=mask
        )
    
    # Benchmark
    start = time.time()
    for _ in range(n_iters):
        model.train(
            radar_obs=radar_obs,
            velocity_obs=velocity_obs,
            base_actions=base_actions,
            returns=returns,
            values=values,
            actions=actions,
            old_log_probs=old_log_probs,
            mask=mask
        )
    elapsed = time.time() - start
    
    iters_per_sec = n_iters / elapsed
    ms_per_iter = (elapsed / n_iters) * 1000
    
    print(f"Time: {elapsed:.2f}s for {n_iters} iters")
    print(f"Speed: {iters_per_sec:.1f} iters/s, {ms_per_iter:.2f} ms/iter")
    
    return elapsed, iters_per_sec

def main():
    print("=" * 60)
    print("Residual Network Training Benchmark: GPU vs CPU")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    # Test CPU only
    results['cpu'] = benchmark_training(use_gpu_global=False, use_gpu_local=False)
    
    # Test GPU if available
    if cuda_available:
        results['gpu'] = benchmark_training(use_gpu_global=True, use_gpu_local=True)
        
        # Compare
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        cpu_time, cpu_speed = results['cpu']
        gpu_time, gpu_speed = results['gpu']
        
        if gpu_speed > cpu_speed:
            speedup = gpu_speed / cpu_speed
            print(f"GPU is {speedup:.2f}x FASTER than CPU")
        else:
            slowdown = cpu_speed / gpu_speed
            print(f"GPU is {slowdown:.2f}x SLOWER than CPU")
            print("(Small model + data transfer overhead > GPU compute benefit)")
    else:
        print("\nNo GPU available, only CPU benchmark completed.")

if __name__ == "__main__":
    main()
