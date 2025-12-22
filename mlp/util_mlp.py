import os
import os.path as osp
import random
import numpy as np
import torch
from typing import Dict, List, Optional
import map_config

from PIL import Image as PILImage

from mlp.alg_parameters_mlp import *  # 使用 MLP 通用参数
from map_config import EnvParameters


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def _avg(vals):
    if vals is None:
        return None
    if isinstance(vals, (list, tuple)) and len(vals) > 0 and isinstance(vals[0], (list, tuple, np.ndarray)):
        return np.nanmean(vals, axis=0)
    if isinstance(vals, (list, tuple, np.ndarray)):
        return float(np.nanmean(vals)) if len(vals) > 0 else 0.0
    return vals


def write_to_tensorboard(global_summary, step: int, performance_dict: Optional[Dict] = None,
                         mb_loss: Optional[List] = None, imitation_loss: Optional[List] = None,
                         q_loss: Optional[float] = None,
                         evaluate: bool = True, greedy: bool = True):
    if global_summary is None:
        return

    # 1. Imitation Loss (Grouped under Loss/)
    if imitation_loss is not None:
        global_summary.add_scalar('Loss/Imitation', imitation_loss[0], step)
        if len(imitation_loss) > 1:
            global_summary.add_scalar('Train/Imitation_Grad_Norm', imitation_loss[1], step)

    # 2. Q Loss
    if q_loss is not None:
        global_summary.add_scalar('Loss/Q_Loss', q_loss, step)

    # 2. Performance (Grouped under Train/ or Eval/)
    if performance_dict:
        prefix = 'Eval' if evaluate else 'Train'
        # Rename keys as requested
        key_map = {
            'per_r': 'Reward',
            'per_episode_len': 'Episode_Length',
            'win': 'Win_Rate'
        }
        for k, v in performance_dict.items():
            val = _avg(v)
            if val is not None:
                name = key_map.get(k, k)
                global_summary.add_scalar(f'{prefix}/{name}', val, step)
                # Add std for Reward and Episode_Length
                if k in ['per_r', 'per_episode_len'] and len(v) > 1:
                    std_val = float(np.nanstd(v))
                    global_summary.add_scalar(f'{prefix}/{name}_Std', std_val, step)

    # 3. Losses (Grouped under Loss/)
    if mb_loss:
        loss_vals = np.nanmean(np.asarray(mb_loss, dtype=np.float32), axis=0)
        # Original names: ['total', 'policy', 'entropy', 'value', 'adv_std', 'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean']
        # User wants to remove value_clip_frac (index 6).
        # Let's map indices to names manually to be safe and clean.
        # 0: total, 1: policy, 2: entropy, 3: value, 4: adv_std, 5: approx_kl, 6: SKIP, 7: clipfrac, 8: grad_norm, 9: adv_mean
        
        mapping = {
            0: 'Total',
            1: 'Policy',
            2: 'Entropy',
            3: 'Value',
            4: 'Adv_Std',
            5: 'Approx_KL',
            # 6: value_clip_frac - SKIPPED
            7: 'Clip_Frac',
            8: 'Grad_Norm',
            9: 'Adv_Mean'
        }
        
        for idx, val in enumerate(loss_vals):
            if idx in mapping:
                name = mapping[idx]
                # Special case for Grad_Norm -> maybe put in Train? Or keep in Loss?
                # User said "don't separate so much". Loss/ is a good group.
                global_summary.add_scalar(f'Loss/{name}', float(val), step)

    global_summary.flush()


def make_gif(images, file_name, fps=20):
    """
    使用 PIL 保存 GIF。
    """
    if PILImage is None:
        return

    if isinstance(images, list):
        frames = [np.asarray(img, dtype=np.uint8) for img in images]
    else:
        frames = np.asarray(images, dtype=np.uint8)

    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        frames = [frames[i] for i in range(frames.shape[0])]

    if len(frames) == 0:
        return

    max_side = getattr(map_config, 'gif_max_side', 640)
    os.makedirs(osp.dirname(file_name), exist_ok=True)
    duration_ms = int(1000.0 / max(int(fps), 1))

    pil_frames = []
    for fr in frames:
        h, w = fr.shape[0], fr.shape[1]
        scale = 1.0
        if max(h, w) > max_side and max_side > 0:
            scale = float(max_side) / float(max(h, w))
        
        img = PILImage.fromarray(fr)
        if scale < 0.999:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), resample=PILImage.LANCZOS)
        pil_frames.append(img)

    if not pil_frames:
        return

    base_img = pil_frames[0].quantize(method=PILImage.ADAPTIVE, colors=256, dither=PILImage.NONE)
    final_frames = [base_img]
    for img in pil_frames[1:]:
        q_img = img.quantize(palette=base_img, dither=PILImage.NONE)
        final_frames.append(q_img)

    final_frames[0].save(
        file_name,
        save_all=True,
        append_images=final_frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0
    )
    print(f"GIF saved: {file_name} (frames={len(frames)})")


def update_perf(one_ep, perf):
    perf['per_r'].append(one_ep['episode_reward'])
    perf['per_episode_len'].append(one_ep['num_step'])


def build_critic_observation(actor_obs, opponent_strategy=None, policy_manager=None):
    """构建critic观测：actor_obs + opponent_id context"""
    from mlp.alg_parameters_mlp import NetParameters  # 确保导入 MLP 参数
    
    actor_vec = np.asarray(actor_obs, dtype=np.float32).reshape(-1)
    
    # 不再强制填充，使用实际观测长度
    # Tracker: 27维, Target: 24维
    
    # 注意：这里已经去掉了 CONTEXT_LEN 的依赖，因为 MLP 版本用 CTDE
    # 如果还需要 context，必须确保 NetParameters 里有定义
    # 但根据之前的修改，MLP 应该不再使用这个函数或者这个函数应该被重写
    # 这里保留原逻辑但指向正确的参数文件
    
    # 实际上 MLP 的 build_critic_observation 已经在 evaluate_mlp.py 中重写了
    # 这个 util 函数可能是给旧代码用的，但为了安全，我们指向 mlp 参数
    
    return actor_vec # 暂时直接返回，因为 MLP 不用这个
