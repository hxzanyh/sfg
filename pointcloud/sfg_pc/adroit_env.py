"""Point-cloud Adroit (door, pen, relocate): environment, seeding, observation extraction, BC evaluator.

The environment is VRL3's AdroitEnv (the mj_envs task, 2 simulation steps per action) wrapped by
3D-Diffusion-Policy's MujocoPointcloudWrapperAdroit (512-point cropped cloud from one camera; relocate
needs patches/dp3_relocate_pointcloud.patch). Call `sfg_pc.paths.setup_adroit()` first.

Success = the cumulative goal counter (sum of n_goal_achieved over simulation steps) reaches 10.
Episodes end at the mj_envs time limit (door, relocate: 100 actions; pen: 50) or when the pen drops.
"""
import numpy as np

TASKS = ('door', 'pen', 'relocate')
GOAL_THRESH = 10.0
MAX_STEPS = 300          # cap on actions per episode; the env's own time limit always ends it first


def make_env(task):
    from adroit import AdroitEnv
    from diffusion_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit
    env = AdroitEnv(env_name=f'{task}-v0', test_image=False, num_repeats=2, num_frames=1,
                    env_feature_type='pixels', device='cuda', reward_rescale=True)
    return MujocoPointcloudWrapperAdroit(env=env, env_name=f'adroit_{task}', use_point_crop=True)


def core(env):
    """The mj_envs MujocoEnv under the wrappers. Its np_random draws every initial state (the
    farthest-point sampling of the cloud has a fixed start point)."""
    return env.env._env._env.env.unwrapped


def seed(env, s):
    """Reseed the initial-state generator (AdroitEnv itself has no seed())."""
    import gym.utils.seeding as seeding
    c = core(env)
    c.seed(int(s))
    assert c.np_random.bit_generator.state == seeding.np_random(int(s))[0].bit_generator.state, \
        f'seeding with {s} did not reach the generator reset_model draws from'


def extract_obs(ts):
    """(xyz point cloud (512, 3), proprioceptive state) of an Adroit time step, in the env's dtype
    (the normaliser returns float32)."""
    return np.asarray(ts.observation_pointcloud)[:, :3], np.asarray(ts.observation_sensor)


def evaluate_bc(agent, env, num_episodes=20, n_action_steps=3, normalizer=None, max_steps=MAX_STEPS):
    """Success rate of `agent.sample_actions` (latent ~ the checkpoint's latent_dist). Used by the BC
    pretraining script; the environment is not reseeded."""
    import torch
    successes, goals = [], []
    n_obs = agent.obs_encoder.n_obs_steps
    for _ in range(num_episodes):
        ts = env.reset()
        goal = 0.0
        queue, hist = [], []
        for _step in range(max_steps):
            pc, st = extract_obs(ts)
            if normalizer is not None:
                st = normalizer.normalize('agent_pos', st)
                pc = normalizer.normalize('point_cloud', pc)
            hist.append((pc, st))
            if len(hist) < n_obs:
                hist = [hist[0]] * (n_obs - len(hist)) + hist
            if not queue:
                recent = hist[-n_obs:]
                obs = {'point_cloud': torch.from_numpy(np.stack([o[0] for o in recent]))[None].float().to(agent.device),
                       'agent_pos': torch.from_numpy(np.stack([o[1] for o in recent]))[None].float().to(agent.device)}
                acts = agent.sample_actions(obs)[0].cpu().numpy()
                if normalizer is not None:
                    acts = normalizer.unnormalize('action', acts)
                queue = list(np.clip(acts, -1.0, 1.0)[:n_action_steps])
            ts = env.step(queue.pop(0))
            goal += float(ts.n_goal_achieved)
            if ts.last():
                break
        successes.append(float(goal >= GOAL_THRESH))
        goals.append(goal)
    return {'success_rate': float(np.mean(successes)), 'mean_goal': float(np.mean(goals)), 'n': num_episodes}
