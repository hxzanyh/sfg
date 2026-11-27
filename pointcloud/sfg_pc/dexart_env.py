"""DexArt environment construction, observation extraction and the BC evaluator.

Call `sfg_pc.paths.setup_dexart()` before `import_dexart()`.
"""
import numpy as np

TASKS = ('bucket', 'faucet', 'toilet')


def import_dexart():
    """Import the DexArt modules the trainer uses; returns (create_env, TRAIN_CONFIG, RANDOM_CONFIG)."""
    from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
    from dexart.env.create_env import create_env
    return create_env, TRAIN_CONFIG, RANDOM_CONFIG


def make_env(task):
    """Point-cloud DexArt env on the task's seen (training) instances, cycled on every reset."""
    create_env, TRAIN_CONFIG, RANDOM_CONFIG = import_dexart()
    return create_env(task_name=task, use_visual_obs=True, use_gui=False, is_eval=True,
                      pc_noise=True, pc_seg=True, index=TRAIN_CONFIG[task]['seen'],
                      img_type='robot', rand_pos=RANDOM_CONFIG[task]['rand_pos'],
                      rand_degree=RANDOM_CONFIG[task]['rand_degree'])


def extract_obs(obs, num_points=1024):
    """(xyz point cloud (num_points, 3), proprioceptive state) from a DexArt observation dict.

    Clouds with more than num_points points are subsampled with the global numpy RNG; DexArt's
    point-cloud processing already returns exactly num_points, so this branch does not fire there.
    """
    pc = np.asarray(obs['instance_1-point_cloud'], dtype=np.float32)
    if pc.ndim == 3:
        pc = pc[0]
    pc = pc[:, :3]
    if pc.shape[0] != num_points:
        if pc.shape[0] > num_points:
            idx = np.random.choice(pc.shape[0], num_points, replace=False)
            pc = pc[idx]
        else:
            pad = np.zeros((num_points - pc.shape[0], 3), dtype=np.float32)
            pc = np.concatenate([pc, pad], 0)
    state = np.asarray(obs['state'], dtype=np.float32)
    if state.ndim == 2:
        state = state[0]
    return pc, state


def evaluate_bc(agent, env, num_episodes=50, n_action_steps=3, normalizer=None, num_points=1024):
    """Success rate of `agent.sample_actions` (latent ~ the checkpoint's latent_dist).
    Success = env.is_eval_done at any step. Used by the BC pretraining script."""
    import torch
    successes = []
    ep_steps = []
    n_obs = agent.obs_encoder.n_obs_steps
    for ep in range(num_episodes):
        obs = env.reset()
        action_queue = []
        obs_history = []
        success = False
        nsteps = 0
        for step in range(env.horizon):
            pc, state = extract_obs(obs, num_points)
            if normalizer is not None:
                state = normalizer.normalize('agent_pos', state)
                pc_min = normalizer.stats['point_cloud']['min']
                pc_rng = normalizer.stats['point_cloud']['range']
                pc = ((pc - pc_min) / pc_rng * 2.0 - 1.0).astype(np.float32)
            obs_history.append({'pc': pc, 'state': state})
            if len(obs_history) < n_obs:
                obs_history = [obs_history[0]] * (n_obs - len(obs_history)) + obs_history

            if len(action_queue) == 0:
                recent = obs_history[-n_obs:]
                obs_dict = {
                    'point_cloud': torch.from_numpy(
                        np.stack([o['pc'] for o in recent])).unsqueeze(0).float().to(agent.device),
                    'agent_pos': torch.from_numpy(
                        np.stack([o['state'] for o in recent])).unsqueeze(0).float().to(agent.device),
                }
                actions = agent.sample_actions(obs_dict)[0].cpu().numpy()
                if normalizer is not None:
                    actions = normalizer.unnormalize('action', actions)
                actions = np.clip(actions, -1.0, 1.0)
                for a in actions[:n_action_steps]:
                    action_queue.append(a)

            action = action_queue.pop(0)
            obs, reward, done, info = env.step(action)
            nsteps += 1
            if env.is_eval_done:
                success = True
            if done:
                break
        successes.append(float(success))
        ep_steps.append(nsteps)
    return {'success_rate': float(np.mean(successes)), 'mean_steps': float(np.mean(ep_steps)),
            'n': num_episodes}
