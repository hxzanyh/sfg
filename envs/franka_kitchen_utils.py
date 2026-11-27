import os

import gymnasium as gym
import numpy as np
import minari
from tqdm import tqdm

from ogpo.utils.datasets import Dataset

def flatten_obs_dict(obs_dict):
    """Flatten a dict observation into a single numpy array (live env wrapper)."""
    goal_keys = sorted(obs_dict['desired_goal'].keys())

    desired_goal_parts = [np.atleast_1d(obs_dict['desired_goal'][key]) for key in goal_keys]
    desired_goal = np.concatenate(desired_goal_parts, axis=-1)

    achieved_goal_parts = [np.atleast_1d(obs_dict['achieved_goal'][key]) for key in goal_keys]
    achieved_goal = np.concatenate(achieved_goal_parts, axis=-1)

    return np.concatenate([obs_dict['observation'], desired_goal, achieved_goal], axis=-1)


def online_reward(count):
    """FrankaKitchenWrapper.step's reward rule applied to the env reward (= number of goal tasks complete now)."""
    reward = count - 7
    return np.where(reward == -3, 0.0, reward)


def limits_norm_enabled():
    # same switch and semantics as envs/robomimic_utils.limits_norm_enabled
    return os.environ.get("SFG_LIMITS_NORM", "0").lower() not in ("", "0", "false")


def online_demos_enabled():
    """SFG_KITCHEN_ONLINE_DEMOS: make offline rows match the online env of make_env (off = OGPO's dataset).

    The online env is recovered with terminate_on_tasks_completed=True and FrankaKitchenWrapper maps its reward
    (number of goal tasks currently complete) to count - 7, with -3 -> 0. The Minari demos were recorded with
    terminate_on_tasks_completed=False and store the raw count, and the shipped loader zeroes the mask at the
    time-limit truncation. A critic trained on a mixed online/offline batch (offline_ratio > 0) would otherwise see
    the same state valued on two reward scales, with two bootstrap rules at the episode end.
    """
    return os.environ.get("SFG_KITCHEN_ONLINE_DEMOS", "0").lower() not in ("", "0", "false")


def _flatten_episode_obs(observations):
    """flatten_obs_dict over a whole Minari episode (N+1 rows), same key order."""
    goal_keys = sorted(observations['desired_goal'].keys())
    desired_goal = np.concatenate([np.asarray(observations['desired_goal'][k]).reshape(len(observations['observation']), -1)
                                   for k in goal_keys], axis=1)
    achieved_goal = np.concatenate([np.asarray(observations['achieved_goal'][k]).reshape(len(observations['observation']), -1)
                                    for k in goal_keys], axis=1)
    return np.concatenate([observations['observation'], desired_goal, achieved_goal], axis=1)


def get_limits_stats(env_name):
    """Min-max stats over the flattened demo observations and actions (limits normalizer), fit once and
    cached next to the Minari dataset (atomic write) so the subprocess eval envs load identical numbers."""
    dataset = minari.load_dataset(env_name)
    cache = os.path.join(os.path.dirname(dataset.spec.data_path), "ogpo_limits.npz")
    if not os.path.exists(cache):
        O, A = [], []
        for episode in dataset:
            if len(episode.actions) == 0 or len(episode.observations['observation']) == 0:
                continue
            n = min(len(episode.actions), len(episode.observations['observation']))
            O.append(_flatten_episode_obs(episode.observations)[:n])
            A.append(episode.actions[:n])
        O = np.concatenate(O).astype(np.float32)
        A = np.concatenate(A).astype(np.float32)

        def _fit(x):
            mn, mx = x.min(axis=0), x.max(axis=0)
            return mn.astype(np.float32), np.where((mx - mn) < 1e-7, 1.0, mx - mn).astype(np.float32)

        obs_min, obs_range = _fit(O)
        action_min, action_range = _fit(A)
        tmp = cache + f".tmp{os.getpid()}.npz"
        np.savez(tmp, obs_min=obs_min, obs_range=obs_range, action_min=action_min, action_range=action_range)
        os.replace(tmp, cache)
    z = np.load(cache)
    return {k: z[k] for k in z.files}


class FrankaKitchenWrapper(gym.Wrapper):
    """Flattens FrankaKitchen-v1 dict observations into a single vector."""
    def __init__(self, env):
        super().__init__(env)
        sample_obs, _ = env.reset()
        flattened_sample = flatten_obs_dict(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flattened_sample.shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        info['tasks_completed'] = 0
        info['success'] = 0.0
        return flatten_obs_dict(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - 7
        if reward == -3:
            reward = 0
        info['reward'] = reward
        info['tasks_completed'] = len(info.get('episode_task_completions', []))
        info['success'] = float(terminated)
        return flatten_obs_dict(obs_dict), reward, terminated, truncated, info


class LimitsNormWrapper(gym.Wrapper):
    """Agent-side obs and actions in the limits-normalized [-1, 1] space of get_limits_stats (as the robomimic
    wrapper does under SFG_LIMITS_NORM)."""

    def __init__(self, env, stats):
        super().__init__(env)
        self.obs_min, self.obs_range = stats["obs_min"], stats["obs_range"]
        self.action_min, self.action_range = stats["action_min"], stats["action_range"]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape,
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def _obs(self, obs):
        return ((obs - self.obs_min) / self.obs_range * 2.0 - 1.0).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), info

    def step(self, action):
        raw_action = (np.asarray(action) + 1.0) / 2.0 * self.action_range + self.action_min
        obs, reward, terminated, truncated, info = self.env.step(raw_action)
        return self._obs(obs), reward, terminated, truncated, info


def make_env(env_name: str, **kwargs):
    """Creates the FrankaKitchen-v1 environment and applies the flattening wrapper."""
    dataset = minari.load_dataset(env_name, download=True)
    env = dataset.recover_environment(terminate_on_tasks_completed=True, max_episode_steps=600)
    env = FrankaKitchenWrapper(env)
    if limits_norm_enabled():
        env = LimitsNormWrapper(env, get_limits_stats(env_name))
    return env


def _get_dataset_online_consistent(env_name, dataset):
    """SFG_KITCHEN_ONLINE_DEMOS=1 path of get_dataset (see online_demos_enabled)."""
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import BONUS_THRESH

    O, NO, A, R, T, M = [], [], [], [], [], []
    n_term = n_cut = 0
    for episode in tqdm(dataset, desc=f"Processing '{env_name}' (online-consistent)"):
        if len(episode.actions) == 0 or len(episode.observations['observation']) == 0:
            continue
        n = len(episode.actions)
        assert len(episode.observations['observation']) == n + 1, (len(episode.observations['observation']), n)
        flat = _flatten_episode_obs(episode.observations).astype(np.float32)
        ag, dg = episode.observations['achieved_goal'], episode.observations['desired_goal']
        tasks = sorted(dg.keys())
        # the env scores step t on the state AFTER the step (observation index t+1); the stored reward is that count
        done_now = np.stack([np.linalg.norm(np.asarray(ag[k])[1:] - np.asarray(dg[k])[1:], axis=-1) < BONUS_THRESH
                             for k in tasks], axis=1)
        count = done_now.sum(axis=1).astype(np.float32)
        assert np.array_equal(count, np.asarray(episode.rewards, dtype=np.float32)), \
            f"reward != goal-task count in an episode of {env_name}"
        # terminate_on_tasks_completed: the online episode ends at the first step at which every goal task has been
        # complete at least once (episode_task_completions accumulates step completions)
        ever = np.maximum.accumulate(done_now, axis=0).sum(axis=1)
        hit = np.nonzero(ever == len(tasks))[0]
        term = np.zeros(n, dtype=np.float32)
        if len(hit):
            n_cut += n - (hit[0] + 1)
            n = hit[0] + 1
            term[n - 1] = 1.0
            n_term += 1
        term = term[:n]
        terminals = np.zeros(n, dtype=np.float32)
        terminals[-1] = 1.0                       # every episode end (chunks never cross it)
        O.append(flat[:n]); NO.append(flat[1:n + 1]); A.append(np.asarray(episode.actions[:n], dtype=np.float32))
        R.append(online_reward(count[:n]).astype(np.float32)); T.append(terminals); M.append(1.0 - term)
    fields = dict(observations=np.concatenate(O), next_observations=np.concatenate(NO), actions=np.concatenate(A),
                  rewards=np.concatenate(R), terminals=np.concatenate(T), masks=np.concatenate(M))
    print(f"[SFG_KITCHEN_ONLINE_DEMOS] rewards -> count - 7 (-3 -> 0); {n_term} demos end at all-tasks termination "
          f"({n_cut} post-termination rows dropped); {len(O)} episodes, {len(fields['rewards'])} rows; "
          f"masks = 1 - terminations")
    return fields


def get_dataset(env_name: str):
    """Load and process the Franka Kitchen Minari dataset into a Dataset."""
    dataset = minari.load_dataset(env_name)
    total_steps = dataset.total_steps
    print(f"Loading Minari dataset '{env_name}' with {total_steps} timesteps...")

    if online_demos_enabled():
        fields = _get_dataset_online_consistent(env_name, dataset)
        if limits_norm_enabled():
            _ls = get_limits_stats(env_name)
            for _k in ("observations", "next_observations"):
                fields[_k] = ((fields[_k] - _ls["obs_min"]) / _ls["obs_range"] * 2.0 - 1.0).astype(np.float32)
            fields["actions"] = np.clip((fields["actions"] - _ls["action_min"]) / _ls["action_range"] * 2.0 - 1.0,
                                        -1.0 + 1e-6, 1.0 - 1e-6).astype(np.float32)
            print(f"[SFG_LIMITS_NORM] obs/actions -> [-1, 1]; action range per dim {np.round(_ls['action_range'], 3)}")
        return Dataset.create(**fields)

    # Find the first valid episode for shape inference (some episodes are empty).
    first_valid_episode = None
    for episode in dataset:
        if len(episode.actions) > 0 and len(episode.observations['observation']) > 0:
            first_valid_episode = episode
            break
    if first_valid_episode is None:
        raise ValueError("Dataset contains no valid episodes.")

    achieved_goal_t0 = {
        key: val[0] for key, val in first_valid_episode.observations['achieved_goal'].items()
    }
    desired_goal_t0 = {
        key: val[0] for key, val in first_valid_episode.observations['desired_goal'].items()
    }

    sample_obs_dict = {
        'observation': first_valid_episode.observations['observation'][0],
        'achieved_goal': achieved_goal_t0,
        'desired_goal': desired_goal_t0,
    }

    sample_flat_obs = flatten_obs_dict(sample_obs_dict)
    sample_action = first_valid_episode.actions[0]

    all_obs = np.zeros((total_steps, sample_flat_obs.shape[0]), dtype=np.float32)
    all_actions = np.zeros((total_steps, sample_action.shape[0]), dtype=np.float32)
    all_rewards = np.zeros(total_steps, dtype=np.float32)
    all_terminals = np.zeros(total_steps, dtype=np.float32)

    current_idx = 0
    for episode in tqdm(dataset, desc=f"Processing '{env_name}'"):
        if len(episode.actions) == 0 or len(episode.observations['observation']) == 0:
            continue

        num_steps = min(len(episode.actions), len(episode.observations['observation']))
        idx_slice = slice(current_idx, current_idx + num_steps)

        obs_part = episode.observations['observation'][:num_steps]

        goal_keys = sorted(episode.observations['desired_goal'].keys())

        desired_goal_parts = [episode.observations['desired_goal'][key][:num_steps] for key in goal_keys]
        desired_goal_flat = np.concatenate(desired_goal_parts, axis=1)

        achieved_goal_parts = [episode.observations['achieved_goal'][key][:num_steps] for key in goal_keys]
        achieved_goal_flat = np.concatenate(achieved_goal_parts, axis=1)

        episode_obs_flat = np.concatenate([obs_part, desired_goal_flat, achieved_goal_flat], axis=1)

        all_obs[idx_slice] = episode_obs_flat.astype(np.float32)
        all_actions[idx_slice] = episode.actions[:num_steps].astype(np.float32)
        all_rewards[idx_slice] = episode.rewards[:num_steps].astype(np.float32)
        dones = np.logical_or(episode.terminations, episode.truncations)
        all_terminals[idx_slice] = dones[:num_steps].astype(np.float32)
        current_idx += num_steps

    if current_idx < total_steps:
        all_obs = all_obs[:current_idx]
        all_actions = all_actions[:current_idx]
        all_rewards = all_rewards[:current_idx]
        all_terminals = all_terminals[:current_idx]

    next_observations = np.roll(all_obs, -1, axis=0)
    episode_ends = np.where(all_terminals == 1.0)[0]
    for end_idx in episode_ends:
        if end_idx < len(all_terminals) - 1:
            next_observations[end_idx] = all_obs[end_idx]

    masks = 1.0 - all_terminals

    if limits_norm_enabled():
        _ls = get_limits_stats(env_name)
        all_obs = ((all_obs - _ls["obs_min"]) / _ls["obs_range"] * 2.0 - 1.0).astype(np.float32)
        next_observations = ((next_observations - _ls["obs_min"]) / _ls["obs_range"] * 2.0 - 1.0).astype(np.float32)
        all_actions = np.clip((all_actions - _ls["action_min"]) / _ls["action_range"] * 2.0 - 1.0,
                              -1.0 + 1e-6, 1.0 - 1e-6).astype(np.float32)
        print(f"[SFG_LIMITS_NORM] obs/actions -> [-1, 1]; action range per dim {np.round(_ls['action_range'], 3)}")

    return Dataset.create(
        observations=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
        masks=masks,
        next_observations=next_observations,
    )
