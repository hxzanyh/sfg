"""Demonstration zarr loading and [-1, 1] min-max normalisation."""
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class LimitsNormalizer:
    """Per-feature min-max normaliser to [-1, 1]."""

    def __init__(self):
        self.stats = {}

    def fit(self, name, arr, eps=1e-7):
        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
        mn = flat.min(axis=0)
        mx = flat.max(axis=0)
        rng = np.where((mx - mn) < eps, 1.0, (mx - mn))
        self.stats[name] = {
            'min': mn.astype(np.float32),
            'max': mx.astype(np.float32),
            'range': rng.astype(np.float32),
        }

    def normalize(self, name, x):
        s = self.stats[name]
        return ((x - s['min']) / s['range'] * 2.0 - 1.0).astype(np.float32)

    def unnormalize(self, name, x_norm):
        s = self.stats[name]
        return ((x_norm + 1.0) / 2.0 * s['range'] + s['min']).astype(np.float32)


class DemoZarrDataset(Dataset):
    """Point-cloud demonstrations in a zarr store.

    Expected layout: data/{point_cloud (T,N,C>=3), state (T,S), action (T,A), reward (T,), terminal (T,)},
    meta/{episode_ends, episode_outcome (optional)}. Observations stack n_obs_steps frames, actions are
    horizon-length chunks; everything is normalised to [-1, 1].
    """

    def __init__(self, zarr_path, max_episodes=None, n_obs_steps=2, horizon=4, ext_normalizer=None):
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon

        z = zarr.open(zarr_path, 'r')
        ep_ends = np.array(z['meta/episode_ends'])
        n_episodes = len(ep_ends) if max_episodes is None else min(max_episodes, len(ep_ends))
        total_steps = ep_ends[n_episodes - 1]

        states_raw = np.array(z['data/state'][:total_steps], dtype=np.float32)
        actions_raw = np.array(z['data/action'][:total_steps], dtype=np.float32)
        if 'data/point_cloud' not in z:
            raise ValueError(f'{zarr_path} has no data/point_cloud')
        pc_raw = np.array(z['data/point_cloud'][:total_steps], dtype=np.float32)
        if pc_raw.ndim == 3 and pc_raw.shape[-1] > 3:
            pc_raw = pc_raw[..., :3]          # xyz only

        if 'data/reward' not in z:
            raise ValueError(f'{zarr_path} has no data/reward (per-step task-completion flag)')
        self.rewards = np.array(z['data/reward'][:total_steps], dtype=np.float32)
        if 'data/terminal' in z:
            self.terminals = np.array(z['data/terminal'][:total_steps], dtype=np.float32)
        else:
            self.terminals = np.zeros(total_steps, dtype=np.float32)
            for end in ep_ends[:n_episodes]:
                self.terminals[end - 1] = 1.0

        self.ep_ends = ep_ends[:n_episodes]
        self.ep_starts = np.concatenate([[0], self.ep_ends[:-1]])
        self.n_episodes = n_episodes
        self.action_dim = actions_raw.shape[-1]
        self.state_dim = states_raw.shape[-1]

        # per-episode outcome: meta/episode_outcome if present, else success iff any reward > 0
        if 'meta/episode_outcome' in z:
            self.episode_outcomes = np.array(
                z['meta/episode_outcome'])[:n_episodes].astype(np.float32)
        else:
            self.episode_outcomes = np.array(
                [float(self.rewards[self.ep_starts[i]:self.ep_ends[i]].sum() > 0)
                 for i in range(n_episodes)], dtype=np.float32)

        # fit on this zarr, or reuse the policy checkpoint's normaliser
        if ext_normalizer is not None:
            self.normalizer = ext_normalizer
            print("[data] using the checkpoint's normaliser")
        else:
            self.normalizer = LimitsNormalizer()
            self.normalizer.fit('agent_pos', states_raw)
            self.normalizer.fit('action', actions_raw)
            self.normalizer.fit('point_cloud', pc_raw.reshape(-1, pc_raw.shape[-1]))

        self.states = self.normalizer.normalize('agent_pos', states_raw)
        self.actions = np.clip(
            self.normalizer.normalize('action', actions_raw), -1.0 + 1e-6, 1.0 - 1e-6)
        pc_flat_min = self.normalizer.stats['point_cloud']['min']
        pc_flat_rng = self.normalizer.stats['point_cloud']['range']
        self.point_clouds = ((pc_raw - pc_flat_min) / pc_flat_rng * 2.0 - 1.0).astype(np.float32)

        # chunk start indices (+ the episode outcome of each)
        self.valid_indices = []
        self.valid_outcomes = []
        for ep_idx in range(n_episodes):
            start = self.ep_starts[ep_idx]
            end = self.ep_ends[ep_idx]
            ep_len = end - start
            for t in range(n_obs_steps - 1, ep_len - horizon):
                self.valid_indices.append(start + t)
                self.valid_outcomes.append(self.episode_outcomes[ep_idx])
        self.valid_indices = np.array(self.valid_indices)
        self.valid_outcomes = np.array(self.valid_outcomes, dtype=np.float32)

        print(f"[data] {n_episodes} episodes, {total_steps} steps, "
              f"{len(self.valid_indices)} chunks, action_dim={self.action_dim}, state_dim={self.state_dim}")
        print(f"[data] raw ranges: state=[{states_raw.min():.2f},{states_raw.max():.2f}], "
              f"pc=[{pc_raw.min():.2f},{pc_raw.max():.2f}], "
              f"action=[{actions_raw.min():.2f},{actions_raw.max():.2f}] -> [-1,1]")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        pc_stack = []
        state_stack = []
        for i in range(self.n_obs_steps):
            offset = t - (self.n_obs_steps - 1) + i
            pc_stack.append(self.point_clouds[offset])
            state_stack.append(self.states[offset])

        obs = {'agent_pos': np.stack(state_stack, axis=0),
               'point_cloud': np.stack(pc_stack, axis=0)}

        actions = self.actions[t:t + self.horizon]
        reward = self.rewards[t:t + self.horizon].sum()

        next_t = min(t + self.horizon, len(self.states) - 1)
        next_pc_stack = []
        next_state_stack = []
        for i in range(self.n_obs_steps):
            offset = next_t - (self.n_obs_steps - 1) + i
            offset = max(0, min(offset, len(self.states) - 1))
            next_pc_stack.append(self.point_clouds[offset])
            next_state_stack.append(self.states[offset])

        next_obs = {'agent_pos': np.stack(next_state_stack, axis=0),
                    'point_cloud': np.stack(next_pc_stack, axis=0)}

        mask = 1.0 - float(self.terminals[t:t + self.horizon].any())

        return {
            'observations': obs,
            'next_observations': next_obs,
            'actions': actions,
            'rewards': reward,
            'masks': mask,
            'outcome': float(self.valid_outcomes[idx]),
        }

    def sample(self, batch_size):
        """Uniform batch of chunks (global numpy RNG)."""
        indices = np.random.randint(0, len(self), size=batch_size)
        batch = [self[i] for i in indices]
        return collate_batch(batch)


def collate_batch(batch_list):
    """List of dicts -> dict of batched tensors."""
    result = {}
    for key in batch_list[0]:
        if isinstance(batch_list[0][key], dict):
            sub = {}
            for subkey in batch_list[0][key]:
                sub[subkey] = torch.from_numpy(
                    np.stack([b[key][subkey] for b in batch_list]))
            result[key] = sub
        elif isinstance(batch_list[0][key], np.ndarray):
            result[key] = torch.from_numpy(
                np.stack([b[key] for b in batch_list]))
        else:
            result[key] = torch.tensor([b[key] for b in batch_list])
    return result
