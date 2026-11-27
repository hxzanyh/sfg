"""CPU unit tests (no simulator, no GPU, no data): defaults, projection, losses, sampling, data, networks.

    cd pointcloud && python -m pytest tests -q      # or: python tests/test_sfg_pc.py
"""
import inspect
import math
import os
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfg_pc.meanflow.agent import MeanFlowAgent  # noqa: E402
from sfg_pc.meanflow.data import DemoZarrDataset  # noqa: E402
from sfg_pc.meanflow.networks import DP3PointNetEncoder, MeanFlowUNet  # noqa: E402
from sfg_pc.projection import project_functional_step, rms  # noqa: E402
from sfg_pc.train import SFGTrainer, build_parser, mlp  # noqa: E402


def approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)

FIG6_SFG = dict(task='bucket', episodes=2600, warmup_eps=100, critic_pretrain=10000, utd=4, bs=128,
                actor_bs=128, n_action_steps=3, gamma=0.99, tau=0.005, lr=3e-4, critic_wd=0.0, width=512,
                critic_depth=2, n_qa=10, nsub=2, lr_dec=3e-6, grad_clip=1.0, func_delta=0.01,
                func_bank_states=32, func_bank_lats=8, eval_every=5, eval_eps=20, eval_from=105,
                eval_seed_base=31337, prior_eval_eps=100, final_eval_eps=100, final_eval_seed_base=900000,
                holdout_at='300,800,1600,2000', full_snapshot_every_ep=400, seed=1)


def test_defaults_are_the_figure6_sfg_configuration():
    A = build_parser().parse_args([])
    for k, v in FIG6_SFG.items():
        assert getattr(A, k) == v, (k, getattr(A, k), v)


# ---- projection -------------------------------------------------------------------------------
def _linear_bank(theta, x):
    return torch.clamp(x @ theta, -1, 1)


def test_projection_leaves_small_steps_alone():
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    prev = torch.randn(3, 2) * 0.1
    theta = nn.Parameter(prev + 1e-4)
    f_prev = _linear_bank(prev, x)
    out = project_functional_step([theta], [prev.clone()], f_prev, lambda: _linear_bank(theta.detach(), x), 0.05)
    delta_raw, alpha, iters, resid = out
    assert delta_raw < 0.05 and alpha == 1.0 and iters == 0
    torch.testing.assert_close(theta.detach(), prev + 1e-4, rtol=0, atol=0)


def test_projection_hits_delta_on_the_segment():
    torch.manual_seed(1)
    x = torch.randn(64, 3) * 0.2
    prev = torch.zeros(3, 2)
    step = torch.randn(3, 2)
    theta = nn.Parameter(prev + step)
    f_prev = _linear_bank(prev, x)
    bank = lambda: _linear_bank(theta.detach(), x)  # noqa: E731
    delta = 0.02
    delta_raw, alpha, iters, resid = project_functional_step([theta], [prev.clone()], f_prev, bank, delta)
    assert delta_raw > delta and 0 < alpha < 1 and 1 <= iters <= 8
    assert abs(rms(bank() - f_prev) - delta) <= 0.05 * delta
    torch.testing.assert_close(theta.detach(), prev + alpha * step)


def test_projection_falls_back_to_the_best_alpha_and_never_extrapolates():
    # a decoder that jumps from -1 to 1 at alpha = 0.5: D(alpha) is 0 below, 2 above -> no alpha meets
    # delta = 0.5 within 5%; the iteration alternates 0.25 / 1 and keeps the best (0.25, resid 1)
    prev = torch.zeros(1)
    theta = nn.Parameter(torch.ones(1))
    bank = lambda: torch.clamp(1000 * (theta.detach() - 0.5), -1, 1)[None]  # noqa: E731
    f_prev = torch.full((1, 1), -1.0)
    delta_raw, alpha, iters, resid = project_functional_step([theta], [prev.clone()], f_prev, bank, 0.5)
    assert approx(delta_raw, 2.0) and iters == 8
    assert approx(alpha, 0.25) and approx(resid, 1.0)
    assert approx(float(theta), 0.25)


# ---- critic / actor objectives ----------------------------------------------------------------
def _toy_trainer(n_qa=4, nsub=2, gamma=0.9, nd=6, obs=5):
    torch.manual_seed(0)
    QA = nn.ModuleList([mlp(obs + nd, 1, 8, 2) for _ in range(n_qa)])
    QAt = nn.ModuleList([mlp(obs + nd, 1, 8, 2) for _ in range(n_qa)])
    dec = nn.Linear(obs + nd, nd)
    t = types.SimpleNamespace(A=types.SimpleNamespace(n_qa=n_qa, nsub=nsub, gamma=gamma), QA=QA, QAt=QAt,
                              dev='cpu', nd=nd)
    t.decode_fwd = lambda s, w: torch.clamp(dec(torch.cat([s, w], -1)), -1, 1)
    t.decode = lambda s, w: t.decode_fwd(s, w).detach()
    return t


def test_critic_target_is_min_over_random_target_heads():
    t = _toy_trainer()
    B = 7
    b = {'s': torch.randn(B, 5), 'a': torch.rand(B, 6) * 2 - 1, 'r': torch.rand(B), 's2': torch.randn(B, 5),
         'd': torch.zeros(B)}
    torch.manual_seed(3); np.random.seed(3)
    loss, dg = SFGTrainer.critic_loss(t, b)
    torch.manual_seed(3); np.random.seed(3)
    w2 = torch.randn(B, t.nd)
    sub = np.random.choice(4, 2, replace=False)
    a2 = t.decode(b['s2'], w2)
    q2 = torch.stack([t.QAt[i](torch.cat([b['s2'], a2], -1)).squeeze(-1) for i in sub]).min(0).values
    y = b['r'] + 0.9 * q2
    q = torch.stack([t.QA[i](torch.cat([b['s'], b['a']], -1)).squeeze(-1) for i in range(4)])
    torch.testing.assert_close(loss, ((q - y[None]) ** 2).mean(1).sum())


def test_actor_loss_is_scale_free_min_of_two_heads():
    t = _toy_trainer()
    s, w = torch.randn(9, 5), torch.randn(9, 6)
    np.random.seed(5)
    l1, q1 = SFGTrainer.actor_loss(t, s, w)
    for h in t.QA:                                   # rescaling the critic leaves the loss unchanged
        h[-1].weight.data.mul_(10.0); h[-1].bias.data.mul_(10.0)
    np.random.seed(5)
    l2, q2 = SFGTrainer.actor_loss(t, s, w)
    torch.testing.assert_close(q2, 10.0 * q1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(l1, l2, rtol=1e-5, atol=1e-6)


def test_batches_are_replay_then_demonstrations():
    t = types.SimpleNamespace(nbuf=10, n_demo=4, dev='cpu',
                              buf={'s': np.arange(20, dtype=np.float32)[:, None], 'r': np.zeros(20, np.float32)},
                              demo={'s': 100 + np.arange(4, dtype=np.float32)[:, None], 'r': np.ones(4, np.float32)})
    np.random.seed(0)
    b = SFGTrainer.sample_batch(t, 3)
    np.random.seed(0)
    io = np.random.randint(0, 10, 3); idm = np.random.randint(0, 4, 3)
    assert b['s'][:3, 0].tolist() == io.tolist()
    assert (b['s'][3:, 0] - 100).tolist() == idm.tolist()
    assert b['r'].tolist() == [0, 0, 0, 1, 1, 1]


# ---- data and networks ------------------------------------------------------------------------
def _write_zarr(path, lens=(12, 9), n_pts=16, sdim=5, adim=3):
    import zarr
    rng = np.random.RandomState(0)
    T = sum(lens)
    root = zarr.group(str(path))
    d = root.create_group('data'); m = root.create_group('meta')
    d.create_dataset('point_cloud', data=rng.randn(T, n_pts, 6).astype(np.float32))
    d.create_dataset('state', data=rng.randn(T, sdim).astype(np.float32))
    d.create_dataset('action', data=rng.uniform(-1, 1, (T, adim)).astype(np.float32))
    d.create_dataset('reward', data=(rng.rand(T) > 0.5).astype(np.float32))
    term = np.zeros(T, np.float32); term[np.cumsum(lens) - 1] = 1
    d.create_dataset('terminal', data=term)
    m.create_dataset('episode_ends', data=np.cumsum(lens).astype(np.int64))
    m.create_dataset('episode_outcome', data=np.ones(len(lens), np.float32))
    return T


def test_dataset_chunks_and_normalisation(tmp_path):
    _write_zarr(tmp_path / 'd.zarr')
    ds = DemoZarrDataset(str(tmp_path / 'd.zarr'), n_obs_steps=2, horizon=4)
    assert len(ds) == (12 - 4 - 1) + (9 - 4 - 1)
    assert ds.point_clouds.shape[-1] == 3
    assert ds.states.min() >= -1 - 1e-6 and ds.states.max() <= 1 + 1e-6
    item = ds[0]
    assert item['observations']['point_cloud'].shape == (2, 16, 3) and item['actions'].shape == (4, 3)


def test_networks_shapes():
    enc = DP3PointNetEncoder(state_dim=5, n_obs_steps=2)
    assert enc.obs_dim == 256
    f = enc({'point_cloud': torch.randn(3, 2, 16, 3), 'agent_pos': torch.randn(3, 2, 5)})
    assert f.shape == (3, 256)
    net = MeanFlowUNet(action_dim=3, horizon=4, down_dims=(8, 16), kernel_size=3, n_groups=2,
                       time_features=8, obs_dim=256)
    out = net(f, torch.randn(3, 12), torch.ones(3, 1), torch.ones(3, 1))
    assert out.shape == (3, 12)


def test_meanflow_loss_runs_on_cpu():
    cfg = dict(action_dim=3, horizon=4, state_dim=5, n_obs_steps=2, latent_dist='normal', flow_ratio=0.5,
               cfg_scale=2.0, unet_down_features=(8, 16), unet_kernel_size=3, unet_n_groups=2,
               unet_time_features=8)
    torch.manual_seed(0)
    mf = MeanFlowAgent(cfg, device='cpu')
    obs = {'point_cloud': torch.randn(6, 2, 16, 3), 'agent_pos': torch.randn(6, 2, 5)}
    loss = mf.compute_jit_mf_loss(mf.obs_encoder(obs), torch.rand(6, 12) * 2 - 1)
    loss.backward()
    assert torch.isfinite(loss)
    assert mf.sample_actions(obs).shape == (6, 4, 3)


# ---- one-shot reward ----------------------------------------------------------------------------
def test_reward_default_per_task():
    from sfg_pc.train import resolve_args
    p = os.path.abspath(__file__)
    assert resolve_args(['--task', 'door', '--ckpt', p, '--demo', p, '--out', '/tmp/x']).reward == 'oneshot'
    assert resolve_args(['--task', 'bucket', '--ckpt', p, '--demo', p, '--out', '/tmp/x']).reward == 'indicator'


def _oneshot_trainer(path, adroit):
    from sfg_pc.meanflow.data import LimitsNormalizer
    cfg = dict(action_dim=3, horizon=4, state_dim=5, n_obs_steps=2, unet_down_features=(8, 16),
               unet_kernel_size=3, unet_n_groups=2, unet_time_features=8)
    torch.manual_seed(0)
    ds0 = DemoZarrDataset(str(path), n_obs_steps=2, horizon=4)
    nm = LimitsNormalizer(); nm.stats = ds0.normalizer.stats
    t = types.SimpleNamespace(A=types.SimpleNamespace(demo=str(path)), mf=MeanFlowAgent(cfg, device='cpu'),
                              dev='cpu', NAS=3, n_obs=2, horizon=4, nm=nm, oneshot=True, adroit=adroit)
    t._demo_first_success = lambda ds: SFGTrainer._demo_first_success(t, ds)
    return t


def test_oneshot_demo_rows(tmp_path):
    import zarr
    lens = (20, 16)
    _write_zarr(tmp_path / 'd.zarr', lens=lens)
    z = zarr.open(str(tmp_path / 'd.zarr'), 'r+')
    rew = np.zeros(sum(lens), np.float32); rew[9:14] = 1; rew[20 + 7:] = 1      # indicator on for a while
    z['data/reward'][:] = rew
    goal = np.zeros(sum(lens), np.float32); goal[4:] = 2.0                      # counter reaches 10 at step 8
    z.create_dataset('data/n_goal_achieved', data=goal)
    for adroit, first in ((False, (9, 27)), (True, (8, 24))):
        t = _oneshot_trainer(tmp_path / 'd.zarr', adroit)
        demo = SFGTrainer._demo_transitions(t)
        assert demo['r'].sum() == 2 and (demo['d'] == demo['r']).all()
        # rows start at t = 1, 4, 7, ...; the success row is the one whose window [t, t+3) holds `first`
        n_rows = [len(range(1, first[0] + 1, 3)), len(range(21, first[1] + 1, 3))]
        assert len(demo['r']) == sum(n_rows)
        assert demo['r'][n_rows[0] - 1] == 1 and demo['r'][-1] == 1


class _TS:
    def __init__(self, g, last=False, tl=False):
        self.n_goal_achieved = g; self._last = last; self.time_limit_reached = tl
        self.observation_pointcloud = np.zeros((4, 6)); self.observation_sensor = np.zeros(5)

    def last(self):
        return self._last


def _macro_trainer(steps):
    from sfg_pc.meanflow.data import LimitsNormalizer
    nm = LimitsNormalizer()
    nm.stats = {k: {'min': np.zeros(d, np.float32), 'range': np.ones(d, np.float32)}
                for k, d in (('point_cloud', 3), ('agent_pos', 5), ('action', 3))}
    it = iter(steps)
    env = types.SimpleNamespace(step=lambda a: next(it))
    t = types.SimpleNamespace(NAS=3, horizon=4, adim=3, nm=nm, oneshot=True, adroit=True, _goal=0.0,
                              _terminated=False, env=lambda: env)
    t._env_macro_adroit = lambda acts: SFGTrainer._env_macro_adroit(t, acts)
    return t


def test_oneshot_adroit_steps():
    a = np.zeros(12, np.float32)
    t = _macro_trainer([_TS(4), _TS(4), _TS(0), _TS(2), _TS(2), _TS(2)])     # crosses 10 on the 4th step
    _, r, done, succ = SFGTrainer.env_macro(t, a)
    assert (r, done, succ, t._terminated) == (0.0, False, False, False)
    pairs, r, done, succ = SFGTrainer.env_macro(t, a)
    assert (r, done, succ, t._terminated, len(pairs)) == (1.0, True, True, True, 1)
    t = _macro_trainer([_TS(0), _TS(0, last=True, tl=False)])               # pen drop: terminal, no reward
    _, r, done, _ = SFGTrainer.env_macro(t, a)
    assert (r, done, t._terminated) == (0.0, True, True)
    t = _macro_trainer([_TS(0), _TS(0, last=True, tl=True)])                # time limit: truncation
    _, r, done, _ = SFGTrainer.env_macro(t, a)
    assert (r, done, t._terminated) == (0.0, True, False)


def test_oneshot_dexart_pays_once():
    flags = iter([False, True, True])
    env = types.SimpleNamespace(is_eval_done=False)

    def step(a):
        env.is_eval_done = next(flags)
        return {'instance_1-point_cloud': np.zeros((1024, 3)), 'state': np.zeros(5)}, 0.0, False, {}
    env.step = step
    t = _macro_trainer([])
    t.adroit = False; t._paid = False; t.env = lambda: env
    pairs, r, done, succ = SFGTrainer.env_macro(t, np.zeros(12, np.float32))
    assert (r, done, succ, t._terminated, len(pairs)) == (1.0, True, True, True, 2)


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith('test_') and callable(f)):
        kw = {}
        if 'tmp_path' in inspect.signature(fn).parameters:
            import pathlib
            kw['tmp_path'] = pathlib.Path(tempfile.mkdtemp())
        try:
            fn(**kw); print(f'PASS {name}')
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f'FAIL {name}: {type(e).__name__}: {e}')
    sys.exit(1 if failed else 0)
