"""Collect successful DexArt demonstrations from a released DexArt PPO expert into a zarr store.

The expert acts deterministically (no action noise); an episode is kept iff env.is_eval_done held at
some step. Per step we store the raw observation point cloud (as returned by the env, xyz + colour),
the proprioceptive state, the action and reward = float(env.is_eval_done); terminal = 1 on the last
step of each episode.

Layout: data/{point_cloud, state, action, reward, terminal}, meta/{episode_ends, episode_outcome,
episode_sigmas}. Requires $DEXART_ROOT (or $DP3_ROOT) for DexArt's examples/ and its bundled
stable_baselines3, and the DexArt assets (assets/rl_checkpoints).
"""
import argparse
import json
import os
import shutil

import numpy as np

from sfg_pc import paths


def _patch_cloudpickle():
    """Released PPO archives pickle an old gym RandomNumberGenerator; load it as None."""
    import cloudpickle
    orig = cloudpickle.loads

    def loads(data):
        try:
            return orig(data)
        except AttributeError as e:
            if 'RandomNumberGenerator' in str(e):
                return None
            raise
    cloudpickle.loads = loads


def downsample_with_fps(points, num_points=1024):
    import torch
    import pytorch3d.ops as torch3d_ops
    pts = torch.from_numpy(points).unsqueeze(0).cuda()
    _, idx = torch3d_ops.sample_farthest_points(points=pts[..., :3], K=torch.tensor([num_points]).cuda())
    return points[idx.squeeze(0).cpu().numpy()]


def collect_successes(env, policy, n_want, num_points, max_attempts):
    episodes, attempts = [], 0
    while len(episodes) < n_want and attempts < max_attempts:
        attempts += 1
        obs = env.reset()
        pcs, states, actions, rewards = [], [], [], []
        success = False
        for step in range(env.horizon):
            obs_input = {k: v[np.newaxis, :] for k, v in obs.items()} if isinstance(obs, dict) else obs[np.newaxis, :]
            action = policy.predict(observation=obs_input, deterministic=True)[0]
            pc_raw = obs['instance_1-point_cloud']
            if pc_raw.ndim == 3:
                pc_raw = pc_raw[0]
            if pc_raw.shape[0] > num_points:
                pc_raw = downsample_with_fps(pc_raw, num_points)
            state_raw = obs['state']
            if state_raw.ndim == 2:
                state_raw = state_raw[0]
            pcs.append(np.asarray(pc_raw, dtype=np.float32))
            states.append(np.asarray(state_raw, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            obs, reward, done, _ = env.step(action)
            rewards.append(float(env.is_eval_done))
            if env.is_eval_done:
                success = True
            if done:
                break
        if success:
            episodes.append({'pcs': pcs, 'states': states, 'actions': actions, 'rewards': rewards,
                             'n_steps': len(pcs)})
            print(f"  [success] {len(episodes)}/{n_want}: steps={len(pcs)} reward_sum={sum(rewards):.0f} "
                  f"(attempt {attempts})", flush=True)
    print(f"  collected {len(episodes)}/{n_want} successes from {attempts} attempts", flush=True)
    return episodes


def save_zarr(eps, save_dir):
    import zarr
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    all_pcs, all_states, all_actions, all_rewards, all_terms, ep_ends = [], [], [], [], [], []
    cumulative = 0
    pc_shape = eps[0]['pcs'][0].shape
    for ep in eps:
        for pc in ep['pcs']:
            if pc.shape == pc_shape:
                all_pcs.append(pc)
            else:
                padded = np.zeros(pc_shape, dtype=np.float32)
                n = min(pc.shape[0], pc_shape[0]); c = min(pc.shape[1], pc_shape[1])
                padded[:n, :c] = pc[:n, :c]; all_pcs.append(padded)
        all_states.extend(ep['states']); all_actions.extend(ep['actions']); all_rewards.extend(ep['rewards'])
        terms = [0.0] * ep['n_steps']; terms[-1] = 1.0
        all_terms.extend(terms)
        cumulative += ep['n_steps']; ep_ends.append(cumulative)

    comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    root = zarr.group(save_dir)
    zd = root.create_group('data'); zm = root.create_group('meta')
    pc_arr = np.stack(all_pcs).astype(np.float32)
    state_arr = np.array(all_states, dtype=np.float32)
    action_arr = np.array(all_actions, dtype=np.float32)
    zd.create_dataset('point_cloud', data=pc_arr, chunks=(100,) + pc_arr.shape[1:], dtype='float32', compressor=comp)
    zd.create_dataset('state', data=state_arr, chunks=(100, state_arr.shape[1]), dtype='float32', compressor=comp)
    zd.create_dataset('action', data=action_arr, chunks=(100, action_arr.shape[1]), dtype='float32', compressor=comp)
    zd.create_dataset('reward', data=np.array(all_rewards, dtype=np.float32), chunks=(10000,), dtype='float32',
                      compressor=comp)
    zd.create_dataset('terminal', data=np.array(all_terms, dtype=np.float32), chunks=(10000,), dtype='float32',
                      compressor=comp)
    zm.create_dataset('episode_ends', data=np.array(ep_ends, dtype=np.int64), dtype='int64', compressor=comp)
    zm.create_dataset('episode_outcome', data=np.ones(len(ep_ends), dtype=np.float32), dtype='float32',
                      compressor=comp)
    zm.create_dataset('episode_sigmas', data=np.zeros(len(ep_ends), dtype=np.float32), dtype='float32',
                      compressor=comp)
    print(f"  saved {save_dir}: {len(ep_ends)} episodes, {cumulative} steps, "
          f"rewarded steps {int(np.sum(all_rewards))}", flush=True)
    return {'n_eps': len(ep_ends), 'steps': cumulative}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', default='bucket', choices=['bucket', 'faucet', 'toilet'])
    ap.add_argument('--rl_ckpt', default=None,
                    help='PPO expert archive (default $DEXART_ROOT/assets/rl_checkpoints/<task>/<task>_seg_dam_0.zip)')
    ap.add_argument('--n_success', type=int, default=50)
    ap.add_argument('--num_points', type=int, default=1024)
    ap.add_argument('--max_attempts_ratio', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True, help='output zarr, e.g. $SFG_PC_OUT/data/dexart_bucket_succ.zarr')
    A = ap.parse_args(argv)

    paths.setup_vulkan()
    root = paths.setup_dexart(with_examples=True)
    rl_ckpt = A.rl_ckpt or os.path.join(root, 'assets', 'rl_checkpoints', A.task, f'{A.task}_seg_dam_0.zip')
    if not os.path.exists(rl_ckpt):
        raise SystemExit(f'PPO expert {rl_ckpt} not found (DexArt assets missing?)')
    _patch_cloudpickle()
    import torch
    from stable_baselines3 import PPO
    from train import get_3d_policy_kwargs          # DexArt examples/train.py
    from sfg_pc.dexart_env import make_env

    np.random.seed(A.seed); torch.manual_seed(A.seed)
    env = make_env(A.task)
    policy = PPO.load(rl_ckpt, env, 'cuda:0', policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                      check_obs_space=False, force_load=True)
    eps = collect_successes(env, policy, A.n_success, A.num_points, A.n_success * A.max_attempts_ratio)
    info = save_zarr(eps, A.out)
    json.dump({'task': A.task, 'rl_ckpt': os.path.basename(rl_ckpt), 'seed': A.seed,
               'reward': 'is_eval_done per step', **info},
              open(os.path.join(A.out, 'gen_meta.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
