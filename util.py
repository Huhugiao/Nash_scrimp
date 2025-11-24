import os
import os.path as osp
import random
import numpy as np
import torch
from typing import Dict, List, Optional
import map_config

# 仅使用 PIL，移除 imageio
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

try:
    import wandb
except Exception:
    wandb = None

from lstm.alg_parameters import *  # 使用通用参数
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
                         evaluate: bool = True, greedy: bool = True):
    if global_summary is None:
        return

    if imitation_loss is not None:
        global_summary.add_scalar('Loss/Imitation_loss', imitation_loss[0], step)
        if len(imitation_loss) > 1:
            global_summary.add_scalar('Grad/Imitation_grad', imitation_loss[1], step)

    # Performance
    if performance_dict:
        prefix = 'Eval' if evaluate else 'Train'
        for k, v in performance_dict.items():
            val = _avg(v)
            if val is not None:
                global_summary.add_scalar(f'{prefix}/{k}', val, step)

    # Loss
    if mb_loss:
        loss_vals = np.nanmean(np.asarray(mb_loss, dtype=np.float32), axis=0)
        names = getattr(RecordingParameters, 'LOSS_NAME', [
            'total', 'policy', 'entropy', 'value', 'value_aux', 'aux1', 'aux2', 'clipfrac', 'grad_norm', 'adv_mean'
        ])
        for i, name in enumerate(names[:len(loss_vals)]):
            tag = f'Loss/{name}' if 'grad' not in name else f'Grad/{name}'
            global_summary.add_scalar(tag, float(loss_vals[i]), step)

    global_summary.flush()


def write_to_wandb(step: int, performance_dict: Optional[Dict] = None,
                   mb_loss: Optional[List] = None, imitation_loss: Optional[List] = None,
                   evaluate: bool = True, greedy: bool = True):
    if wandb is None or not getattr(RecordingParameters, 'WANDB', False) or getattr(wandb, 'run', None) is None:
        return

    log_data = {}
    if imitation_loss is not None:
        log_data['Loss/Imitation_loss'] = imitation_loss[0]
        if len(imitation_loss) > 1:
            log_data['Grad/Imitation_grad'] = imitation_loss[1]

    if performance_dict:
        prefix = 'Eval' if evaluate else 'Train'
        for k, v in performance_dict.items():
            val = _avg(v)
            if val is not None:
                log_data[f'{prefix}/{k}'] = val

    if mb_loss:
        loss_vals = np.nanmean(np.asarray(mb_loss, dtype=np.float32), axis=0)
        names = getattr(RecordingParameters, 'LOSS_NAME', [
            'total', 'policy', 'entropy', 'value', 'value_aux', 'aux1', 'aux2', 'clipfrac', 'grad_norm', 'adv_mean'
        ])
        for i, name in enumerate(names[:len(loss_vals)]):
            tag = f'Loss/{name}' if 'grad' not in name else f'Grad/{name}'
            log_data[tag] = float(loss_vals[i])

    if log_data:
        wandb.log(log_data, step=step)


def make_gif(images, file_name, fps=20):
    """
    使用 PIL 保存 GIF。
    优化：
    1. 使用全局调色板 (Global Palette) 消除颜色闪烁。
    2. 禁用抖动 (Dither=None) 消除噪点。
    3. 启用 optimize=True 减小文件大小。
    """
    if PILImage is None:
        print("PIL (Pillow) not available, cannot save gif.")
        return

    # Prepare frames as uint8 numpy arrays
    if isinstance(images, list):
        frames = [np.asarray(img, dtype=np.uint8) for img in images]
    else:
        frames = np.asarray(images, dtype=np.uint8)

    # If frames is a single numpy array stack, convert to list of frames
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        frames = [frames[i] for i in range(frames.shape[0])]

    if len(frames) == 0:
        return

    # Determine target max side (pixels) from config
    max_side = getattr(map_config, 'gif_max_side', 640)
    
    os.makedirs(osp.dirname(file_name), exist_ok=True)
    duration_ms = int(1000.0 / max(int(fps), 1))

    try:
        pil_frames = []
        # 1. 预处理所有帧（缩放）
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

        # 2. 生成全局调色板 (Global Palette)
        # 从第一帧生成调色板，后续所有帧都强制使用这个调色板。
        # 这能彻底解决背景和半透明区域颜色闪烁的问题。
        base_img = pil_frames[0].quantize(method=PILImage.ADAPTIVE, colors=256, dither=PILImage.NONE)
        
        final_frames = [base_img]
        for img in pil_frames[1:]:
            # 使用 base_img 的调色板进行量化
            q_img = img.quantize(palette=base_img, dither=PILImage.NONE)
            final_frames.append(q_img)

        # 3. 保存
        final_frames[0].save(
            file_name,
            save_all=True,
            append_images=final_frames[1:],
            optimize=True,  # 启用 LZW 压缩和帧差异优化
            duration=duration_ms,
            loop=0
        )
        print(f"Wrote gif: {file_name} (frames={len(frames)})")
        
    except Exception as e:
        print(f"Failed to write gif {file_name}: {e}")


def update_perf(one_ep, perf):
    perf['per_r'].append(one_ep['episode_reward'])
    perf['per_episode_len'].append(one_ep['num_step'])

def get_opponent_id_one_hot(opponent_id):
    """将对手策略的ID转换为one-hot向量。"""
    num_policies = EnvParameters.NUM_TARGET_POLICIES
    one_hot = np.zeros(num_policies, dtype=np.float32)
    if 0 <= opponent_id < num_policies:
        one_hot[opponent_id] = 1.0
    return one_hot

def build_critic_observation(actor_obs, opponent_strategy=None, policy_manager=None):
    """构建critic观测：actor_obs + opponent_id context"""
    from lstm.alg_parameters import NetParameters  # 确保导入
    
    actor_vec = np.asarray(actor_obs, dtype=np.float32).reshape(-1)
    
    # 不再强制填充，使用实际观测长度
    # Tracker: 27维, Target: 24维
    
    context = np.zeros(NetParameters.CONTEXT_LEN, dtype=np.float32)
    if opponent_strategy is not None and policy_manager is not None:
        policy_id = policy_manager.get_policy_id(opponent_strategy)
        if policy_id is not None and policy_id >= 0:
            context = get_opponent_id_one_hot(policy_id)

    return np.concatenate([actor_vec, context], axis=0)