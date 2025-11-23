import os
import os.path as osp
import math
import numpy as np
import torch
import ray

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    import setproctitle
except Exception:
    setproctitle = None

from torch.utils.tensorboard import SummaryWriter

from alg_parameters import *
from env import TrackingEnv
from model_lstm import Model
from runner_lstm import Runner
from util import set_global_seeds, write_to_tensorboard, write_to_wandb, make_gif, get_opponent_id_one_hot
from rule_policies import TRACKER_POLICY_REGISTRY, TARGET_POLICY_REGISTRY
from policymanager import PolicyManager

try:
    import wandb
except Exception:
    wandb = None

IL_INITIAL_PROB = 0.8
IL_FINAL_PROB = 0.1
IL_DECAY_STEPS = 1e7

PURE_RL_SWITCH = 1
if PURE_RL_SWITCH:
    IL_INITIAL_PROB = 0
    IL_FINAL_PROB = 0
    IL_DECAY_STEPS = 1

LSTM_HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 1

if not ray.is_initialized():
    ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to SCRIMP with LSTM on Protecting Environment!\n")
print(f"Training agent: {TrainingParameters.AGENT_TO_TRAIN} with {TrainingParameters.OPPONENT_TYPE} opponent")
print(f"LSTM Configuration: Hidden Size={LSTM_HIDDEN_SIZE}, Layers={NUM_LSTM_LAYERS}")
if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"}:
    # 动态打印可用的target策略（来自注册表）
    available = sorted(list(TARGET_POLICY_REGISTRY.keys()))
    if TrainingParameters.OPPONENT_TYPE == "random":
        # 若支持专家目标策略，可选地在前面插入
        pass
    print("Available target policies:", ", ".join(available))
print(f"IL type: {getattr(TrainingParameters, 'IL_TYPE', 'expert')}")
print(f"IL probability will cosine anneal from {IL_INITIAL_PROB*100:.1f}% to {IL_FINAL_PROB*100:.1f}% over {IL_DECAY_STEPS} steps")
# Add explicit notice for APF expert when training tracker
if TrainingParameters.AGENT_TO_TRAIN == "tracker" and getattr(TrainingParameters, 'IL_TYPE', 'expert') == "expert":
    print("Imitation teacher (tracker): APF rule policy")

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
    'agent_to_train': TrainingParameters.AGENT_TO_TRAIN,
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


def main():
    model_dict = None
    wandb_id = None

    if def_attr('RETRAIN', False):
        restore_path = def_attr('RESTORE_DIR', None)
        if restore_path:
            model_path = restore_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
            if os.path.exists(model_path):
                model_dict = torch.load(model_path, map_location='cpu')

    if def_attr('WANDB', False) and wandb is not None:
        wandb_id = model_dict.get('wandb_id', None) if model_dict else None
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=getattr(RecordingParameters, 'ENTITY', None),
                   notes=getattr(RecordingParameters, 'EXPERIMENT_NOTE', ''),
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('Launching wandb...\n')

    global_summary = None
    if def_attr('TENSORBOARD', True):
        os.makedirs(SUMMARY_PATH, exist_ok=True)
        global_summary = SummaryWriter(SUMMARY_PATH)
        print('Launching tensorboard...\n')

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
        if TrainingParameters.AGENT_TO_TRAIN == "tracker":
            opp_path = SetupParameters.PRETRAINED_TARGET_PATH
        else:
            opp_path = SetupParameters.PRETRAINED_TRACKER_PATH
        if opp_path and os.path.exists(opp_path):
            opponent_dict = torch.load(opp_path, map_location='cpu')
            opponent_model.network.load_state_dict(opponent_dict['model'])
            opponent_weights = opponent_model.get_weights()

    env_mission = 0 if TrainingParameters.AGENT_TO_TRAIN == "tracker" else 1

    global_policy_manager = None
    if TrainingParameters.OPPONENT_TYPE in {"random", "random_nonexpert"} and TrainingParameters.ADAPTIVE_SAMPLING:
        global_policy_manager = PolicyManager()

    envs = [Runner.remote(i + 1, env_mission) for i in range(TrainingParameters.N_ENVS)]
    eval_env = TrackingEnv(mission=env_mission)

    curr_steps = int(model_dict.get("step", 0)) if model_dict is not None else 0
    curr_episodes = int(model_dict.get("episode", 0)) if model_dict is not None else 0
    best_perf = float(model_dict.get("reward", -1e9)) if model_dict is not None else -1e9

    last_test_t = -int(EVAL_INTERVAL) - 1
    last_model_t = -int(SAVE_INTERVAL) - 1
    last_best_t = -int(BEST_INTERVAL) - 1
    last_gif_t = -int(GIF_INTERVAL) - 1

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(GIFS_PATH, exist_ok=True)

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
                            print(f"[WARN] 非有限模仿损失，跳过该 batch（step={curr_steps}）")
                            continue
                        mb_loss.append(loss_result)
                if global_summary and mb_loss:
                    valid_il_losses = [loss for loss in mb_loss if np.all(np.isfinite(loss))]
                    if valid_il_losses:
                        avg_il_loss = np.nanmean([loss[0] for loss in valid_il_losses])
                        global_summary.add_scalar('Train/imitation_loss', avg_il_loss, curr_steps)
                curr_steps += int(TrainingParameters.N_ENVS * TrainingParameters.N_STEPS)
                curr_episodes += total_il_episodes
            else:
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
                    print(f"[WARN] 无有效段落，跳过该轮训练（step={curr_steps}）")
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
                            print(f"[WARN] 非有限损失，跳过该 batch（step={curr_steps}）")
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

            if global_summary:
                for key, value in train_stats.items():
                    global_summary.add_scalar(f'Train/{key}', value, curr_steps)

            if curr_steps - last_model_t >= SAVE_INTERVAL:
                last_model_t = curr_steps
                model_path = osp.join(MODEL_PATH, 'latest')
                os.makedirs(model_path, exist_ok=True)
                save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
                checkpoint = {"model": training_model.network.state_dict(),
                              "optimizer": training_model.net_optimizer.state_dict(),
                              "step": curr_steps, "episode": curr_episodes, "reward": avg_perf_for_best}
                torch.save(checkpoint, save_path)

            if avg_perf_for_best > best_perf and (curr_steps - last_best_t >= BEST_INTERVAL):
                best_perf = avg_perf_for_best
                last_best_t = curr_steps
                model_path = osp.join(MODEL_PATH, 'best_model')
                os.makedirs(model_path, exist_ok=True)
                save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
                checkpoint = {"model": training_model.network.state_dict(),
                              "optimizer": training_model.net_optimizer.state_dict(),
                              "step": curr_steps, "episode": curr_episodes, "reward": best_perf}
                torch.save(checkpoint, save_path)
                print(f"New best model saved at step {curr_steps} with reward {best_perf:.4f}")

            if curr_steps - last_test_t >= EVAL_INTERVAL:
                last_test_t = curr_steps
                eval_stats = evaluate_single_agent(eval_env, training_model, opponent_model, global_device)
                if global_summary:
                    for key, value in eval_stats.items():
                        global_summary.add_scalar(f'Eval/{key}', value, curr_steps)

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
        save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
        checkpoint = {"model": training_model.network.state_dict(),
                      "optimizer": training_model.net_optimizer.state_dict(),
                      "step": curr_steps, "episode": curr_episodes, "reward": best_perf}
        torch.save(checkpoint, save_path)
        print(f"Final model saved to {save_path}")
        ray.shutdown()


def get_opponent_action_for_eval(actor_obs, critic_obs, opponent_type, agent_to_train, opponent_model,
                                  policy_manager, current_policy_name, current_policy_id, opponent_hidden=None):
    if opponent_type == "policy":
        if opponent_model is None:
            raise RuntimeError("OPPONENT_TYPE=policy but opponent_model is None")
        opp_action, _, new_opponent_hidden, _, _ = opponent_model.evaluate(actor_obs, critic_obs, opponent_hidden, greedy=True)
        return opp_action, new_opponent_hidden
    elif opponent_type == "expert":
        if agent_to_train == "tracker":
            default_target = sorted(TARGET_POLICY_REGISTRY.keys())[0]
            policy_cls = TARGET_POLICY_REGISTRY.get(default_target)
            if policy_cls:
                policy_obj = policy_cls()
                opp_pair = policy_obj.get_action(actor_obs)
            else:
                raise ValueError("No target policy found")
        else:
            default_tracker = sorted(TRACKER_POLICY_REGISTRY.keys())[0]
            policy_fn = TRACKER_POLICY_REGISTRY.get(default_tracker)
            if policy_fn:
                opp_pair = policy_fn(actor_obs)
            else:
                raise ValueError("No tracker policy found")
        return opp_pair, None
    elif opponent_type in {"random", "random_nonexpert"}:
        if policy_manager and current_policy_name:
            opp_pair = policy_manager.get_action(current_policy_name, actor_obs)
        else:
            # 回退到默认策略
            if agent_to_train == "tracker":
                default_target = sorted(TARGET_POLICY_REGISTRY.keys())[0]
                policy_cls = TARGET_POLICY_REGISTRY.get(default_target)
                policy_obj = policy_cls()
                opp_pair = policy_obj.get_action(actor_obs)
            else:
                default_tracker = sorted(TRACKER_POLICY_REGISTRY.keys())[0]
                policy_fn = TRACKER_POLICY_REGISTRY.get(default_tracker)
                opp_pair = policy_fn(actor_obs)
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
            opponent_role = "target" if TrainingParameters.AGENT_TO_TRAIN == "tracker" else "tracker"
            current_policy_name, current_policy_id = eval_policy_manager.sample_policy(opponent_role)
            eval_policy_manager.reset()

        while not done and ep_len < EnvParameters.EPISODE_LEN:
            # 训练tracker：agent使用tracker_obs；对手使用target_obs
            if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                critic_obs_full = np.concatenate([tracker_obs, get_opponent_id_one_hot(current_policy_id)])
                agent_action, _, agent_hidden, _, _ = agent_model.evaluate(tracker_obs, critic_obs_full, agent_hidden, greedy=True)

                # 对手动作
                opp_actor_obs = target_obs
                opp_pair, opponent_hidden = get_opponent_action_for_eval(
                    opp_actor_obs, opp_actor_obs, TrainingParameters.OPPONENT_TYPE, TrainingParameters.AGENT_TO_TRAIN,
                    opponent_model, eval_policy_manager, current_policy_name, current_policy_id, opponent_hidden
                )

                tracker_action, target_action = agent_action, opp_pair
            else:
                # 训练target（注意：当前网络的ACTOR_VECTOR_LEN=27，目标观测24，仅保留逻辑以防将来扩展）
                critic_obs_full = np.concatenate([target_obs, get_opponent_id_one_hot(current_policy_id)])
                agent_action, _, agent_hidden, _, _ = agent_model.evaluate(target_obs, critic_obs_full, agent_hidden, greedy=True)

                opp_actor_obs = tracker_obs
                opp_pair, opponent_hidden = get_opponent_action_for_eval(
                    opp_actor_obs, opp_actor_obs, TrainingParameters.OPPONENT_TYPE, TrainingParameters.AGENT_TO_TRAIN,
                    opponent_model, eval_policy_manager, current_policy_name, current_policy_id, opponent_hidden
                )

                tracker_action, target_action = opp_pair, agent_action

            obs_result, reward, terminated, truncated, info = eval_env.step((tracker_action, target_action))
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
        opponent_role = "target" if TrainingParameters.AGENT_TO_TRAIN == "tracker" else "tracker"
        current_policy_name, current_policy_id = eval_policy_manager.sample_policy(opponent_role)
        eval_policy_manager.reset()

    while not done and ep_len < EnvParameters.EPISODE_LEN:
        frame = eval_env.render(mode='rgb_array')
        if frame is not None:
            episode_frames.append(frame)

        if TrainingParameters.AGENT_TO_TRAIN == "tracker":
            critic_obs_full = np.concatenate([tracker_obs, get_opponent_id_one_hot(current_policy_id)])
            agent_action, _, agent_hidden, _, _ = agent_model.evaluate(tracker_obs, critic_obs_full, agent_hidden, greedy=True)
            opp_actor_obs = target_obs
        else:
            critic_obs_full = np.concatenate([target_obs, get_opponent_id_one_hot(current_policy_id)])
            agent_action, _, agent_hidden, _, _ = agent_model.evaluate(target_obs, critic_obs_full, agent_hidden, greedy=True)
            opp_actor_obs = tracker_obs

        opp_pair, opponent_hidden = get_opponent_action_for_eval(
            opp_actor_obs, opp_actor_obs, TrainingParameters.OPPONENT_TYPE, TrainingParameters.AGENT_TO_TRAIN,
            opponent_model, eval_policy_manager, current_policy_name, current_policy_id, opponent_hidden
        )

        tracker_action, target_action = (agent_action, opp_pair) if TrainingParameters.AGENT_TO_TRAIN == "tracker" else (opp_pair, agent_action)
        obs_result, reward, terminated, truncated, info = eval_env.step((tracker_action, target_action))
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

# === add this entrypoint ===
if __name__ == "__main__":
    main()