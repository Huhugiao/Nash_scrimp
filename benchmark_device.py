#!/usr/bin/env python
"""
CPU vs GPU 性能测试脚本
测试内容：网络前向/反向传播 + 环境模拟
"""
import time
import torch
import numpy as np
from env import TrackingEnv
from mlp.model_mlp import Model
from mlp.alg_parameters_mlp import NetParameters, TrainingParameters

def benchmark_network(device, n_iters=1000):
    """测试网络前向+反向传播速度"""
    model = Model(device, global_model=True)
    
    # 模拟数据
    actor_obs = torch.randn(TrainingParameters.MINIBATCH_SIZE, NetParameters.ACTOR_RAW_LEN, device=device)
    critic_obs = torch.randn(TrainingParameters.MINIBATCH_SIZE, NetParameters.CRITIC_RAW_LEN, device=device)
    
    # 预热
    for _ in range(50):
        model.network(actor_obs, critic_obs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测试前向
    start = time.perf_counter()
    for _ in range(n_iters):
        model.network(actor_obs, critic_obs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    forward_time = time.perf_counter() - start
    
    # 测试前向+反向
    start = time.perf_counter()
    for _ in range(n_iters):
        output = model.network(actor_obs, critic_obs)
        loss = output[0].mean() + output[1].mean()  # 假设 loss
        loss.backward()
        model.net_optimizer.zero_grad()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    full_time = time.perf_counter() - start
    
    return forward_time, full_time

def benchmark_env(n_steps=2000):
    """测试环境模拟速度 (纯 CPU)"""
    env = TrackingEnv()
    env.reset()
    
    start = time.perf_counter()
    for _ in range(n_steps):
        action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step((action, action))
        if terminated or truncated:
            env.reset()
    env_time = time.perf_counter() - start
    return env_time

def main():
    print("=" * 60)
    print("CPU vs GPU 性能测试")
    print("=" * 60)
    
    # 检查 CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: 不可用")
    print(f"CPU: 可用")
    print()
    
    # 测试参数
    n_network_iters = 1000
    n_env_steps = 2000
    
    # 1. 环境测试 (纯 CPU)
    print(f"[1/3] 测试环境模拟 ({n_env_steps} steps)...")
    env_time = benchmark_env(n_env_steps)
    print(f"      环境模拟: {env_time:.3f}s ({n_env_steps/env_time:.1f} steps/s)")
    print()
    
    # 2. CPU 网络测试
    print(f"[2/3] 测试 CPU 网络 ({n_network_iters} iters, batch={TrainingParameters.MINIBATCH_SIZE})...")
    cpu_device = torch.device('cpu')
    cpu_fwd, cpu_full = benchmark_network(cpu_device, n_network_iters)
    print(f"      CPU 前向: {cpu_fwd:.3f}s ({n_network_iters/cpu_fwd:.1f} iter/s)")
    print(f"      CPU 前向+反向: {cpu_full:.3f}s ({n_network_iters/cpu_full:.1f} iter/s)")
    print()
    
    # 3. GPU 网络测试
    if cuda_available:
        print(f"[3/3] 测试 GPU 网络 ({n_network_iters} iters, batch={TrainingParameters.MINIBATCH_SIZE})...")
        gpu_device = torch.device('cuda')
        gpu_fwd, gpu_full = benchmark_network(gpu_device, n_network_iters)
        print(f"      GPU 前向: {gpu_fwd:.3f}s ({n_network_iters/gpu_fwd:.1f} iter/s)")
        print(f"      GPU 前向+反向: {gpu_full:.3f}s ({n_network_iters/gpu_full:.1f} iter/s)")
        print()
        
        # 对比
        print("=" * 60)
        print("对比结果:")
        print(f"  前向传播 - GPU 比 CPU: {cpu_fwd/gpu_fwd:.2f}x {'(GPU更快)' if gpu_fwd < cpu_fwd else '(CPU更快)'}")
        print(f"  前向+反向 - GPU 比 CPU: {cpu_full/gpu_full:.2f}x {'(GPU更快)' if gpu_full < cpu_full else '(CPU更快)'}")
        
        # 实际训练瓶颈分析
        # 假设每次训练迭代: N_ENVS * N_STEPS 环境步 + N_EPOCHS 次网络更新
        env_steps_per_iter = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
        network_updates = 10  # 假设 N_EPOCHS=10
        
        # 估算每次迭代时间
        env_ratio = env_steps_per_iter / n_env_steps  # 缩放环境时间
        net_ratio = network_updates / n_network_iters  # 缩放网络时间
        
        cpu_iter_time = env_time * env_ratio + cpu_full * net_ratio
        gpu_iter_time = env_time * env_ratio + gpu_full * net_ratio
        
        print()
        print("训练迭代估算:")
        print(f"  环境步数/迭代: {env_steps_per_iter}")
        print(f"  网络更新次数/迭代: ~{network_updates}")
        print(f"  CPU 模式估算: {cpu_iter_time:.3f}s/iter")
        print(f"  GPU 模式估算: {gpu_iter_time:.3f}s/iter")
        print(f"  预期加速比: {cpu_iter_time/gpu_iter_time:.2f}x")
    else:
        print("[3/3] 跳过 GPU 测试 (CUDA 不可用)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
