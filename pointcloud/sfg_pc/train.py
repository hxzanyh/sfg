"""SFG online fine-tuning of a one-step MeanFlow point-cloud policy on DexArt and point-cloud Adroit.

Tasks: DexArt bucket / faucet / toilet, Adroit door / pen / relocate (sfg_pc/adroit_env.py).
Reward (--reward):
  indicator  (DexArt only) env.is_eval_done on every env step, summed over the executed steps; no
             transition is terminal (d = 0).
  oneshot    at most one reward per episode: 1 on the env step at which the episode first succeeds
             (DexArt: the first is_eval_done; Adroit: the cumulative goal counter reaching 10), 0 on
             every other step, and the episode ends there with d = 1. A pen drop also ends the episode
             with d = 1 (r = 0); time limits are truncations (d = 0). Demonstration rows: the row whose
             executed steps contain the first success gets r = 1, d = 1, later rows are dropped.

Schedule (one run, or split at the handoff with --save_handoff / --resume_handoff):
  1. warm-up: --warmup_eps episodes with theta_0, transitions stored in the replay buffer;
  2. --critic_pretrain critic-only TD updates (the handoff point);
  3. online: per decision (one executed action chunk), --utd critic updates, each followed by one
     actor update; at every episode boundary the episode's functional step is projected to at most
     --func_delta (sfg_pc/projection.py).

Critic: N = --n_qa LayerNorm MLP heads on (encoder feature, action chunk); target
  y = r + gamma (1 - d) min_{j in M} Q_tgt_j(s', f_theta(s', eps')), M = --nsub random target heads,
  loss = sum_heads mean_batch (Q_i(s, a) - y)^2; Polyak target with rate --tau.
Batches: --bs rows from the replay buffer + --bs rows from the demonstrations (RLPD, 50/50).
Actor: L = -mean( q / sg(mean |q|) ), q = min over 2 random heads of Q(s, clamp(f_theta(s, eps))),
  gradient through the one-step decoder; Adam, --lr_dec, gradient-norm clip --grad_clip; encoder frozen.
Latents (--latent): eps ~ N(0, I) as in BC ('normal'), or projected onto |eps| = sqrt(dim) ('sphere',
  the Figure 6 runs).
Evaluation: episode i of a block reseeds numpy/python and the env with seed_base + i; on DexArt it also
  uses object instance i mod n (--pin_eval_instances), so blocks on the same seeds are paired by initial
  condition (DexArt's camera/point-cloud noise is not seed-governed).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from sfg_pc import adroit_env, paths
from sfg_pc.dexart_env import TASKS as DEXART_TASKS, extract_obs, make_env
from sfg_pc.meanflow.agent import MeanFlowAgent
from sfg_pc.meanflow.data import DemoZarrDataset, LimitsNormalizer
from sfg_pc.projection import project_functional_step, rms

CAP = 400000          # replay capacity (transitions)
ENC_CHUNK = 128       # encoder batch when precomputing demonstration features
BANK_CHUNK = 64       # decoder batch on the probe bank
BURST_LOG_EVERY = 500
BANK_SEED = 12345     # probe bank (states and noise) is fixed across runs
TASKS = DEXART_TASKS + adroit_env.TASKS


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', choices=TASKS, default='bucket')
    ap.add_argument('--reward', choices=['indicator', 'oneshot'], default=None,
                    help='see the module docstring (default: oneshot on door/pen/relocate, indicator on DexArt)')
    ap.add_argument('--latent', choices=['normal', 'sphere'], default='normal',
                    help="decoder latents: N(0, I) ('normal') or projected onto |eps| = sqrt(dim) ('sphere')")
    ap.add_argument('--ckpt', default=os.environ.get('SFG_PC_THETA0'),
                    help='pretrained MeanFlow checkpoint theta_0 (default $SFG_PC_THETA0)')
    ap.add_argument('--demo', default=os.environ.get('SFG_PC_DEMO'),
                    help='demonstration zarr (default $SFG_PC_DEMO)')
    # schedule
    ap.add_argument('--episodes', type=int, default=2600)
    ap.add_argument('--warmup_eps', type=int, default=100)
    ap.add_argument('--critic_pretrain', type=int, default=10000,
                    help='critic-only updates after warm-up, before the first actor update')
    ap.add_argument('--utd', type=int, default=4, help='critic (and actor) updates per decision')
    ap.add_argument('--bs', type=int, default=128, help='rows per side; critic batch = 2 * bs')
    ap.add_argument('--actor_bs', type=int, default=128)
    ap.add_argument('--n_action_steps', type=int, default=3,
                    help='actions executed per decode (the policy predicts horizon=4)')
    ap.add_argument('--gamma', type=float, default=0.99, help='discount per decision (per action chunk)')
    # critic
    ap.add_argument('--tau', type=float, default=0.005)
    ap.add_argument('--lr', type=float, default=3e-4, help='critic learning rate')
    ap.add_argument('--critic_wd', type=float, default=0.0, help='critic AdamW weight decay')
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--critic_depth', type=int, default=2, help='hidden layers per head')
    ap.add_argument('--n_qa', type=int, default=10, help='critic heads')
    ap.add_argument('--nsub', type=int, default=2, help='random target heads in the min')
    # actor
    ap.add_argument('--lr_dec', type=float, default=3e-6, help='actor (decoder) learning rate')
    ap.add_argument('--grad_clip', type=float, default=1.0, help='actor gradient-norm clip (0 = off)')
    # functional-step bound
    ap.add_argument('--func_delta', type=float, default=0.01,
                    help='per-episode functional step bound delta; 1e9 = unbounded (the step is still '
                         'measured and logged); 0 = no probe bank and no step log')
    ap.add_argument('--func_bank_states', type=int, default=32)
    ap.add_argument('--func_bank_lats', type=int, default=8, help='noise draws per probe state')
    # evaluation
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--eval_eps', type=int, default=20)
    ap.add_argument('--eval_from', type=int, default=105)
    ap.add_argument('--eval_seed_base', type=int, default=31337)
    ap.add_argument('--prior_eval_eps', type=int, default=100,
                    help='episodes of the one evaluation of theta_0 at the start (0 = skip)')
    ap.add_argument('--prior_seed_base', type=int, default=-1,
                    help='seed base of the theta_0 evaluation (-1 = --eval_seed_base)')
    ap.add_argument('--pin_eval_instances', type=int, default=1,
                    help='DexArt: evaluation episode i uses object instance i mod n and the training instance '
                         'counter is restored afterwards (0 = instances keep cycling through evaluations)')
    ap.add_argument('--final_eval_eps', type=int, default=100)
    ap.add_argument('--final_eval_seed_base', type=int, default=900000)
    ap.add_argument('--holdout_at', default='300,800,1600,2000',
                    help='episodes with an extra --final_eval_eps holdout (the last episode always has one)')
    # checkpoints
    ap.add_argument('--full_snapshot_every_ep', type=int, default=400,
                    help='resumable snapshot resume_ep<N>.pt every N episodes, at evaluation episodes (0 = off)')
    ap.add_argument('--save_handoff', default='', help='write the post-warm-up state here and exit')
    ap.add_argument('--resume_handoff', default='', help='resume from a handoff or a resume_ep<N>.pt snapshot')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None, help='output directory (default $SFG_PC_OUT/<tag or sfg_<task>_s<seed>>)')
    ap.add_argument('--tag', default='')
    return ap


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class JsonlLog:
    """Append-only JSON-lines sink."""
    def __init__(self, path):
        self.f = open(path, 'a')

    def __call__(self, **kw):
        self.f.write(json.dumps(kw, default=float) + '\n')
        self.f.flush()


class rng_isolated:
    """Evaluation must not move the training streams: save and restore torch (CPU, CUDA), numpy and
    python RNG states around the block."""
    def __enter__(self):
        self.st = (torch.get_rng_state(), torch.cuda.get_rng_state_all(),
                   np.random.get_state(), random.getstate())
        return self

    def __exit__(self, *a):
        torch.set_rng_state(self.st[0]); torch.cuda.set_rng_state_all(self.st[1])
        np.random.set_state(self.st[2]); random.setstate(self.st[3])
        return False


def mlp(ind, out, width, depth):
    """depth hidden layers of Linear -> LayerNorm -> ReLU, then Linear."""
    layers, d = [], ind
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class SFGTrainer:
    """All state of one run. The constructor performs every RNG-consuming setup step in a fixed
    order: seeding, generator/encoder construction, critic and target construction (torch CPU
    generator), then the probe bank (its own generator). eval_only=True builds only the policy and the
    environment (for policy_eval, e.g. by select_theta0.py)."""

    def __init__(self, A, device='cuda', eval_only=False):
        self.A = A
        dev = self.dev = device
        torch.manual_seed(A.seed); np.random.seed(A.seed); random.seed(A.seed)
        os.makedirs(A.out, exist_ok=True)
        self.tag = A.tag or os.path.basename(os.path.normpath(A.out))
        self.NAS = A.n_action_steps
        self.adroit = A.task in adroit_env.TASKS
        self.oneshot = A.reward == 'oneshot'
        self._terminated = False     # the last executed step ended the episode for real (d = 1)
        self._paid = False           # DexArt oneshot: this episode's reward has been given
        self._goal = 0.0             # Adroit: cumulative goal counter of the current episode
        self.eval_records = []       # per-episode records of the last policy_eval block
        self.log_train = JsonlLog(f'{A.out}/train_metrics.jsonl')
        self.log_eval = JsonlLog(f'{A.out}/eval_episodes.jsonl')
        self.log_perf = JsonlLog(f'{A.out}/performance.jsonl')
        self.ckpt_sha = sha256(A.ckpt)

        # ---- generator and encoder (theta_0) ---------------------------------------------------
        ck = torch.load(A.ckpt, map_location='cpu', weights_only=False, mmap=True)
        cfg = self.cfg = ck['cfg']
        mf = self.mf = MeanFlowAgent(cfg, device=dev)
        mf.load_weights(ck)
        assert self.NAS <= mf.horizon, f'n_action_steps={self.NAS} exceeds the policy horizon {mf.horizon}'
        for p in mf.obs_encoder.parameters():
            p.requires_grad_(False)
        mf.unet.eval(); mf.obs_encoder.eval()
        self.nd = mf.noise_dim
        self.sphere_r = math.sqrt(self.nd)
        self.obs_dim = mf.obs_encoder.obs_dim
        self.horizon, self.adim = cfg['horizon'], cfg['action_dim']
        self.aflat = self.horizon * self.adim
        self.n_obs = cfg['n_obs_steps']
        nm = self.nm = LimitsNormalizer()
        nm.stats = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in ck['normalizer'].items()}
        del ck

        # frozen copy of theta_0, used for the evaluation of the base policy
        self.pi0 = copy.deepcopy(mf.unet).eval()
        for p in self.pi0.parameters():
            p.requires_grad_(False)
        for p in mf.unet.parameters():
            p.requires_grad_(True)
        self.dec_params = [p for p in mf.unet.parameters() if p.requires_grad]
        n_dec = sum(p.numel() for p in self.dec_params)
        print(f"[actor] trainable decoder parameters: {n_dec:,} over {len(self.dec_params)} tensors; "
              f"encoder frozen", flush=True)

        # gym < 0.26 RNGs lack _generator_ctor, which numpy >= 1.25 uses to copy/pickle them
        import gym.utils.seeding as _seed
        if not hasattr(_seed.RandomNumberGenerator, '_generator_ctor'):
            _seed.RandomNumberGenerator._generator_ctor = staticmethod(lambda *a, **k: np.random.default_rng())
        self._env = None
        self.CNT = {'actor_updates': 0, 'critic_updates_total': 0, 'critic_updates_online': 0,
                    'env_steps': 0, 'episodes': 0}
        if eval_only:
            return

        # ---- critic ----------------------------------------------------------------------------
        cin = self.obs_dim + self.aflat
        self.QA = nn.ModuleList([mlp(cin, 1, A.width, A.critic_depth) for _ in range(A.n_qa)]).to(dev)
        self.QAt = nn.ModuleList([mlp(cin, 1, A.width, A.critic_depth) for _ in range(A.n_qa)]).to(dev)
        self.QAt.load_state_dict(self.QA.state_dict())
        for p in self.QAt.parameters():
            p.requires_grad_(False)
        self.oqa = torch.optim.AdamW(self.QA.parameters(), A.lr, weight_decay=A.critic_wd)
        self.opt_dec = torch.optim.Adam(self.dec_params, A.lr_dec)

        # ---- demonstration transitions (s_t, a_{t:t+H}, sum r_{t:t+NAS}, s_{t+NAS}, d=0) ----------
        self.demo = self._demo_transitions()
        self.n_demo = len(self.demo['s'])

        # ---- probe bank for the functional step --------------------------------------------------
        self.func_on = A.func_delta > 0
        if self.func_on:
            g = torch.Generator(device=dev).manual_seed(BANK_SEED)
            si = torch.randint(0, self.n_demo, (A.func_bank_states,), device=dev, generator=g)
            self.fb_s = torch.from_numpy(self.demo['s']).to(dev)[si].repeat_interleave(A.func_bank_lats, 0)
            self.fb_w = torch.randn(len(self.fb_s), self.nd, device=dev, generator=g)
            self.log_steps = JsonlLog(f'{A.out}/sfg_steps.jsonl')
            self.path_length = 0.0
            self.f_fork = None          # bank actions at the first online episode start

        # ---- replay ----------------------------------------------------------------------------
        self.buf = {k: np.zeros((CAP,) + sh, dtype=np.float32) for k, sh in
                    [('s', (self.obs_dim,)), ('a', (self.aflat,)), ('r', ()), ('s2', (self.obs_dim,)), ('d', ())]}
        self.nbuf = 0; self.ptr = 0

        # ---- counters, curve -------------------------------------------------------------------
        self.online = False
        self.n_paid = 0                 # oneshot: training episodes that received their reward
        self.prior_sr = float('nan')
        self.ep_rews = []; self.curve = []
        self.holdout_at = set(int(x) for x in A.holdout_at.split(',') if x)
        for h in sorted(self.holdout_at):
            assert A.eval_from <= h <= A.episodes and h % A.eval_every == 0, \
                f'--holdout_at {h} is not an evaluation episode ({A.eval_from} <= e <= {A.episodes}, e % {A.eval_every} == 0)'
            assert h != A.episodes, f'--holdout_at {h} == --episodes: the final holdout already runs there'

        resolved = {**vars(A), 'obs_dim': int(self.obs_dim), 'aflat': self.aflat, 'horizon': self.horizon,
                    'adim': self.adim, 'n_obs_steps': self.n_obs, 'noise_dim': int(self.nd),
                    'N_demo': self.n_demo, 'ckpt_sha256': self.ckpt_sha,
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0) if dev.startswith('cuda') else dev, 'argv': sys.argv}
        json.dump(resolved, open(f'{A.out}/resolved_config.json', 'w'), indent=2, default=str)
        print(f"[cfg] {self.tag} task={A.task} reward={A.reward} latent={A.latent} lr_critic={A.lr} "
              f"lr_actor={A.lr_dec} utd={A.utd} func_delta={A.func_delta} seed={A.seed}", flush=True)

    # =========================================================================================
    # decoding
    # =========================================================================================
    def sphere(self, w):
        if self.A.latent == 'normal':
            return w
        return w / w.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.sphere_r

    def raw_fwd(self, net, s_feat, w):
        tt = torch.ones(len(s_feat), 1, device=self.dev)
        return net(s_feat, self.sphere(w), tt, tt)

    def decode_fwd(self, s_feat, w):
        """Differentiable one-step decode with the online generator, clamped to [-1, 1]."""
        return self.raw_fwd(self.mf.unet, s_feat, w).clamp(-1, 1)

    @torch.no_grad()
    def decode(self, s_feat, w):
        return self.decode_fwd(s_feat, w)

    @torch.no_grad()
    def decode_pi0(self, s_feat, w):
        return self.raw_fwd(self.pi0, s_feat, w).clamp(-1, 1)

    @torch.no_grad()
    def decode_bank(self):
        return torch.cat([self.decode_fwd(self.fb_s[i:i + BANK_CHUNK], self.fb_w[i:i + BANK_CHUNK])
                          for i in range(0, len(self.fb_s), BANK_CHUNK)])

    # =========================================================================================
    # data
    # =========================================================================================
    def _demo_transitions(self):
        A, mf, dev, NAS, n_obs, horizon = self.A, self.mf, self.dev, self.NAS, self.n_obs, self.horizon
        print("precomputing demonstration transitions ...", flush=True)
        ds = DemoZarrDataset(A.demo, max_episodes=None, n_obs_steps=n_obs, horizon=horizon,
                             ext_normalizer=self.nm)
        demo = {'s': [], 'a': [], 'r': [], 's2': [], 'd': []}
        first = self._demo_first_success(ds) if self.oneshot else None
        hits = dropped = 0
        with torch.no_grad():
            for e in range(ds.n_episodes):
                s0, s1 = ds.ep_starts[e], ds.ep_ends[e]
                ts = list(range(s0 + n_obs - 1, s1 - horizon - NAS, NAS))
                if not ts:
                    continue

                def frames(t):
                    idx = [max(s0, t - n_obs + 1 + i) for i in range(n_obs)]
                    return np.stack([ds.point_clouds[j] for j in idx]), np.stack([ds.states[j] for j in idx])

                for lo in range(0, len(ts), ENC_CHUNK):
                    chunk = ts[lo:lo + ENC_CHUNK]
                    cur = [frames(t) for t in chunk]
                    nxt = [frames(t + NAS) for t in chunk]
                    obs = {'agent_pos': torch.from_numpy(np.stack([x[1] for x in cur])).to(dev),
                           'point_cloud': torch.from_numpy(np.stack([x[0] for x in cur])).to(dev)}
                    obs2 = {'agent_pos': torch.from_numpy(np.stack([x[1] for x in nxt])).to(dev),
                            'point_cloud': torch.from_numpy(np.stack([x[0] for x in nxt])).to(dev)}
                    f1 = mf.obs_encoder(obs); f2 = mf.obs_encoder(obs2)
                    for kk, t in enumerate(chunk):
                        a = ds.actions[t:t + horizon].reshape(-1)
                        if first is None:
                            r = float(ds.rewards[t:t + NAS].sum()); d = 0.0
                        else:
                            f = first[e]
                            if f is not None and t > f:
                                dropped += 1          # after the success: such states never occur online
                                continue
                            hit = f is not None and t <= f < t + NAS
                            r = d = float(hit); hits += int(hit)
                        demo['s'].append(f1[kk].cpu().numpy()); demo['a'].append(a.astype(np.float32))
                        demo['r'].append(r); demo['s2'].append(f2[kk].cpu().numpy())
                        demo['d'].append(d)
        demo = {k: np.array(v, dtype=np.float32) for k, v in demo.items()}
        if first is not None:
            n_succ = sum(f is not None for f in first.values())
            print(f"[oneshot] demonstrations: {ds.n_episodes} episodes, {n_succ} successful; success rows "
                  f"(r=1, d=1) {hits}, later rows dropped {dropped}", flush=True)
            assert hits == n_succ, f'{n_succ - hits} episodes have their first success outside every row window'
        print(f"demonstration transitions: {len(demo['s'])} (reward>0 frac {float((demo['r'] > 0).mean()):.3f})",
              flush=True)
        return demo

    def _demo_first_success(self, ds):
        """Global zarr index of each demonstration episode's first success step (None if it has none)."""
        if self.adroit:
            import zarr
            z = zarr.open(self.A.demo, 'r')
            assert 'data/n_goal_achieved' in z, f'{self.A.demo} has no data/n_goal_achieved (see patches/)'
            T = int(ds.ep_ends[-1])
            goal = np.asarray(z['data/n_goal_achieved'][:T], dtype=np.float32)
            assert len(goal) == T, f'data/n_goal_achieved has {len(goal)} steps, the episodes {T}'
        first = {}
        for e in range(ds.n_episodes):
            s0, s1 = int(ds.ep_starts[e]), int(ds.ep_ends[e])
            if self.adroit:
                nz = np.flatnonzero(np.cumsum(goal[s0:s1]) >= adroit_env.GOAL_THRESH)
            else:
                nz = np.flatnonzero(np.asarray(ds.rewards[s0:s1]) > 0)
            first[e] = s0 + int(nz[0]) if len(nz) else None
        return first

    def push(self, s, a, r, s2, d):
        b, i = self.buf, self.ptr
        b['s'][i] = s; b['a'][i] = a; b['r'][i] = r; b['s2'][i] = s2; b['d'][i] = d
        self.ptr = (self.ptr + 1) % CAP; self.nbuf = min(self.nbuf + 1, CAP)

    def sample_batch(self, bs):
        """bs replay rows followed by bs demonstration rows (RLPD symmetric sampling)."""
        io = np.random.randint(0, self.nbuf, bs)
        idm = np.random.randint(0, self.n_demo, bs)
        return {k: torch.from_numpy(np.concatenate([self.buf[k][io], self.demo[k][idm]])).to(self.dev)
                for k in self.buf}

    # =========================================================================================
    # environment
    # =========================================================================================
    def env(self):
        """Built on first use. DexArt: its point-cloud/camera-noise generator is seeded by DexArt from OS
        entropy at construction and re-seeded by every evaluation episode (env.seed). Adroit: the
        initial-state generator of training episodes is seeded with --seed here; evaluation episodes
        reseed it and policy_eval puts the training generator back afterwards."""
        if self._env is None:
            if self.adroit:
                self._env = adroit_env.make_env(self.A.task)
                adroit_env.seed(self._env, self.A.seed)
            else:
                self._env = make_env(self.A.task)
        return self._env

    def max_steps(self):
        return adroit_env.MAX_STEPS if self.adroit else self.env().horizon

    def obs_to_feat(self, obs_history):
        recent = obs_history[-self.n_obs:]
        while len(recent) < self.n_obs:
            recent = [recent[0]] + recent
        st = torch.from_numpy(np.stack([r[1] for r in recent])[None]).float().to(self.dev)
        pc = torch.from_numpy(np.stack([r[0] for r in recent])[None]).float().to(self.dev)
        with torch.no_grad():
            return self.mf.obs_encoder({'point_cloud': pc, 'agent_pos': st})[0]

    def env_reset(self):
        if self.adroit:
            ts = self.env().reset(); self._goal = 0.0
            pc, st = adroit_env.extract_obs(ts)
            return [(self.nm.normalize('point_cloud', pc), self.nm.normalize('agent_pos', st))]
        obs = self.env().reset(); self._paid = False
        pc, st = extract_obs(obs, 1024)
        st = self.nm.normalize('agent_pos', st); pc = self.nm.normalize('point_cloud', pc)
        return [(pc, st)]

    def env_macro(self, a_flat):
        """Execute the first NAS actions of a decoded chunk; returns (observations, reward, done, success)
        and sets self._terminated. DexArt success = is_eval_done at any executed step."""
        acts = self.nm.unnormalize('action', a_flat.reshape(self.horizon, self.adim))
        if self.adroit:
            return self._env_macro_adroit(acts)
        r = 0.0; done = False; succ = False; pairs = []
        self._terminated = False
        env = self.env()
        for i in range(self.NAS):
            obs, _, done, _info = env.step(acts[i])
            if self.oneshot:
                hit = bool(env.is_eval_done) and not self._paid
                r += float(hit); succ = succ or bool(env.is_eval_done)
                self._terminated = hit          # other env-side ends (horizon) are truncations
                if hit:
                    self._paid = True; done = True
            else:
                r += float(env.is_eval_done); succ = succ or bool(env.is_eval_done)
            pc, st = extract_obs(obs, 1024)
            pairs.append((self.nm.normalize('point_cloud', pc), self.nm.normalize('agent_pos', st)))
            if done:
                break
        return pairs, r, done, succ

    def _env_macro_adroit(self, acts):
        """Adroit: actions clipped to [-1, 1]; one-shot reward at the step where the cumulative goal counter
        first reaches GOAL_THRESH. A genuine end (pen drop: ts.last() before the time limit) is terminal."""
        r = 0.0; done = False; pairs = []
        self._terminated = False
        for i in range(self.NAS):
            ts = self.env().step(np.clip(acts[i], -1.0, 1.0))
            g0 = self._goal
            self._goal += float(ts.n_goal_achieved)
            hit = g0 < adroit_env.GOAL_THRESH <= self._goal       # at most once: the counter is monotone
            r += float(hit)
            done = bool(ts.last())
            if done:
                self._terminated = not bool(getattr(ts, 'time_limit_reached', False))
            if hit:
                done = True; self._terminated = True
            pc, st = adroit_env.extract_obs(ts)
            pairs.append((self.nm.normalize('point_cloud', pc), self.nm.normalize('agent_pos', st)))
            if done:
                break
        return pairs, r, done, self._goal >= adroit_env.GOAL_THRESH

    # =========================================================================================
    # updates
    # =========================================================================================
    def critic_loss(self, b):
        A, QA, QAt, dev = self.A, self.QA, self.QAt, self.dev
        with torch.no_grad():
            w2 = torch.randn(len(b['s2']), self.nd, device=dev)
            a2 = self.decode(b['s2'], w2)
            sub = np.random.choice(A.n_qa, A.nsub, replace=False)
            x2 = torch.cat([b['s2'], a2], -1)
            q2h = torch.stack([QAt[i](x2).squeeze(-1) for i in range(A.n_qa)])       # (H, 2B)
            q2 = q2h[sub].min(0).values
            y = b['r'] + A.gamma * (1 - b['d']) * q2
        x = torch.cat([b['s'], b['a']], -1)
        qh = torch.stack([QA[i](x).squeeze(-1) for i in range(A.n_qa)])                # (H, 2B)
        loss = ((qh - y[None]) ** 2).mean(dim=1).sum()
        dg = {'y_mean': float(y.mean()), 'q_mean': float(qh.mean().detach()),
              'q_std': float(qh.std().detach()), 'td_mse': float(loss.detach() / A.n_qa)}
        return loss, dg

    def actor_loss(self, s, w):
        A, QA = self.A, self.QA
        a = self.decode_fwd(s, w)
        sub = np.random.choice(A.n_qa, 2, replace=False)
        q = torch.stack([QA[i](torch.cat([s, a], -1)).squeeze(-1) for i in sub]).min(0).values
        qn = q / (q.detach().abs().mean() + 1e-6)
        return -qn.mean(), q

    def train_step(self):
        """One critic update, then (online) one actor update on a subset of the same batch."""
        A, CNT = self.A, self.CNT
        CNT['critic_updates_total'] += 1
        if self.online:
            CNT['critic_updates_online'] += 1
        b = self.sample_batch(A.bs)
        qa_loss, dg = self.critic_loss(b)
        self.oqa.zero_grad(); qa_loss.backward(); self.oqa.step()
        with torch.no_grad():
            for tp, p in zip(self.QAt.parameters(), self.QA.parameters()):
                tp.data.mul_(1 - A.tau).add_(p.data, alpha=A.tau)
        if not self.online:
            return float(qa_loss.detach()), float('nan'), dg

        nb = len(b['s'])
        idx = torch.randint(0, nb, (min(A.actor_bs, nb),), device=self.dev)
        s_act = b['s'][idx]
        w = torch.randn(len(s_act), self.nd, device=self.dev)
        a_loss, q = self.actor_loss(s_act, w)
        self.opt_dec.zero_grad(); a_loss.backward()
        if A.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.dec_params, A.grad_clip)
        self.opt_dec.step()
        CNT['actor_updates'] += 1
        return float(qa_loss.detach()), float(a_loss.detach()), dg

    def critic_pretrain_burst(self):
        A = self.A
        if A.critic_pretrain <= 0:
            return
        t = time.time()
        print(f"[critic warm-up] {A.critic_pretrain} critic-only updates (replay {self.nbuf}, demo {self.n_demo})",
              flush=True)
        for i in range(A.critic_pretrain):
            ql, _, dg = self.train_step()
            if (i + 1) % BURST_LOG_EVERY == 0:
                print(f"[critic warm-up {i + 1}] td {dg['td_mse']:.3f} q_std {dg['q_std']:.2f}", flush=True)
                self.log_train(phase='critic_warmup', i=i + 1, **dg, **self.CNT)
        print(f"[critic warm-up] done in {time.time() - t:.0f}s", flush=True)

    def func_step(self, ep, f_pre, th_pre):
        """Project this episode's step and log the functional path."""
        A = self.A
        delta_raw, alpha, iters, resid = project_functional_step(
            self.dec_params, th_pre, f_pre, self.decode_bank, A.func_delta)
        f_now = self.decode_bank()
        step = rms(f_now - f_pre)
        self.path_length += step
        if self.f_fork is None:
            self.f_fork = f_pre.clone()
        net = rms(f_now - self.f_fork)
        self.log_steps(ep=ep, actor_updates=self.CNT['actor_updates'], delta_raw=delta_raw,
                       step_committed=step, alpha=alpha, iters=iters, resid=resid,
                       projected=bool(delta_raw > A.func_delta), path_length=self.path_length,
                       net_distance=net, tortuosity=self.path_length / max(net, 1e-12))

    # =========================================================================================
    # evaluation
    # =========================================================================================
    @torch.no_grad()
    def policy_eval(self, nep, seed_base, ep, kind='policy'):
        """nep episodes; episode i reseeds numpy/python and the env with seed_base + i and draws its
        latents from a generator seeded with 0x5EED * (seed_base + i) + 1. kind='prior' decodes with
        theta_0. DexArt's object instance is a reset counter, not seed-governed: with
        --pin_eval_instances episode i uses instance i mod n and the training counter is restored."""
        if nep <= 0:
            return float('nan')
        succ = 0
        self.eval_records = []
        with rng_isolated():
            if self.adroit:          # env.seed replaces the generator: park the training one
                train_rng = adroit_env.core(self.env()).np_random
                train_st = copy.deepcopy(train_rng.bit_generator.state)
            dx = None if self.adroit else self.env()
            pin = bool(self.A.pin_eval_instances) and dx is not None and hasattr(dx, 'instance_list') \
                and hasattr(dx, 'i')
            if pin:
                i0 = dx.i
            for i in range(nep):
                es = seed_base + i
                np.random.seed(es); random.seed(es)
                if self.adroit:
                    adroit_env.seed(self.env(), es)
                else:
                    try:
                        self.env().seed(es)
                    except Exception:
                        pass
                g = torch.Generator(device=self.dev); g.manual_seed(0x5EED * es + 1)
                if pin:
                    dx.i = (i - 1) % len(dx.instance_list)          # reset_env() advances it to i mod n
                hist = self.env_reset(); done = False; steps = 0; got = False; ret = 0.0
                extra = {'instance': int(dx.index)} if pin else {}
                while not done and steps < self.max_steps():
                    s = self.obs_to_feat(hist)
                    w = torch.randn(1, self.nd, device=self.dev, generator=g)
                    a = (self.decode_pi0 if kind == 'prior' else self.decode)(s[None], w)[0]
                    pairs, r, done, sc = self.env_macro(a.cpu().numpy())
                    hist += pairs; steps += self.NAS; got = got or sc; ret += r
                succ += int(got)
                if self.adroit:
                    extra.update(goal_total=self._goal, terminated=bool(self._terminated))
                rec = dict(kind=kind, ep=ep, actor_u=self.CNT['actor_updates'], i=i, env_seed=es,
                           policy_seed=0x5EED * es + 1, success=int(got), ret=ret, steps=steps, **extra)
                self.eval_records.append(rec)
                self.log_eval(**rec)
            if self.adroit:
                assert train_rng.bit_generator.state == train_st, 'evaluation drew from the training env RNG'
                adroit_env.core(self.env()).np_random = train_rng
            if pin:
                dx.i = i0
        return succ / nep

    # =========================================================================================
    # checkpoints
    # =========================================================================================
    def save_handoff(self, path, kind='handoff', allow_actor_state=False):
        """Full restorable state (networks, optimizers, replay, counters, RNG). The simulator state is
        not saved: a resumed run starts at an episode boundary with a freshly built env."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'kind': kind, 'allow_actor_state': allow_actor_state,
            'unet': self.mf.unet.state_dict(), 'QA': self.QA.state_dict(), 'QAt': self.QAt.state_dict(),
            'oqa': self.oqa.state_dict(), 'opt_dec': self.opt_dec.state_dict(),
            'buf': {k: v[:self.nbuf].copy() for k, v in self.buf.items()}, 'nbuf': self.nbuf, 'ptr': self.ptr,
            'CNT': dict(self.CNT), 'EP_REWS': self.ep_rews, 'curve': self.curve, 'PRIOR0': self.prior_sr,
            'ep': self.CNT['episodes'],
            'FUNC_L': (self.path_length if self.func_on else None),
            'FUNC_BB': (self.f_fork.cpu() if self.func_on and self.f_fork is not None else None),
            'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
            'rng_np': np.random.get_state(), 'rng_py': random.getstate(),
            'adroit_env_rng': (copy.deepcopy(adroit_env.core(self._env).np_random.bit_generator.state)
                               if self.adroit and self._env is not None else None),
            'args': vars(self.A), 'ckpt_sha256': self.ckpt_sha,
        }, path)
        print(f"[{kind}] wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)

    def load_handoff(self, path):
        d = torch.load(path, map_location=self.dev, weights_only=False)
        da = d.get('args', {})
        rw = da.get('reward') or da.get('dexart_reward', 'indicator')
        lt = da.get('latent') or {'ckpt': 'normal', 'sphere': 'sphere'}.get(da.get('latent_conv', 'sphere'))
        assert (rw, lt) == (self.A.reward, self.A.latent), \
            f'{path} was written with reward={rw} latent={lt}, this run has {self.A.reward} / {self.A.latent}'
        self.mf.unet.load_state_dict(d['unet']); self.QA.load_state_dict(d['QA']); self.QAt.load_state_dict(d['QAt'])
        self.oqa.load_state_dict(d['oqa'])
        if not d.get('allow_actor_state', False):
            assert not d['opt_dec']['state'], 'handoff carries actor optimizer state; it must be empty'
        else:
            self.opt_dec.load_state_dict(d['opt_dec'])
        n = d['nbuf']
        for k in self.buf:
            self.buf[k][:n] = d['buf'][k]
        self.nbuf = n; self.ptr = d['ptr']
        self.CNT.update(d['CNT']); self.ep_rews = list(d['EP_REWS']); self.curve = list(d['curve'])
        self.prior_sr = d.get('PRIOR0', float('nan'))
        torch.set_rng_state(d['rng_torch'].cpu() if hasattr(d['rng_torch'], 'cpu') else d['rng_torch'])
        torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, 'cpu') else s for s in d['rng_cuda']])
        np.random.set_state(d['rng_np']); random.setstate(d['rng_py'])
        if self.adroit and d.get('adroit_env_rng') is not None:
            with rng_isolated():                 # building the env must not move the restored streams
                g = adroit_env.core(self.env()).np_random
            g.bit_generator.state = d['adroit_env_rng']
            print('[resume] Adroit training initial-state RNG restored', flush=True)
        if self.func_on and d.get('FUNC_L') is not None:
            self.path_length = float(d['FUNC_L'])
            self.f_fork = d['FUNC_BB'].to(self.dev) if d.get('FUNC_BB') is not None else None
            print(f"[resume] functional path restored: path_length={self.path_length:.3f}", flush=True)
        print(f"[resume] loaded {path} | replay {self.nbuf} | {self.CNT}", flush=True)
        return {'kind': d.get('kind'), 'ep': d.get('ep')}

    # =========================================================================================
    # main loop
    # =========================================================================================
    def run(self):
        A, CNT = self.A, self.CNT
        t0 = time.time()
        start_ep = 0
        prior_base = A.eval_seed_base if A.prior_seed_base < 0 else A.prior_seed_base
        if A.resume_handoff:
            hd = self.load_handoff(A.resume_handoff)
            if np.isnan(self.prior_sr) and A.prior_eval_eps > 0:
                self.prior_sr = self.policy_eval(A.prior_eval_eps, prior_base, 0, kind='prior')
            print(f"[base] theta_0 SR = {self.prior_sr:.3f} (measured once)", flush=True)
            if hd['kind'] == 'periodic' and hd['ep'] is not None:
                start_ep = int(hd['ep'])
                assert start_ep >= A.warmup_eps, 'periodic snapshot predates the end of warm-up'
                self.curve[:] = [x for x in self.curve if x.get('ep', 0) <= start_ep]
                print(f"[resume] periodic snapshot: continuing after ep{start_ep} "
                      f"(actor updates {CNT['actor_updates']})", flush=True)
            else:
                start_ep = A.warmup_eps
            self.online = True                      # the critic warm-up happened before the handoff
        else:
            if A.prior_eval_eps > 0:
                self.prior_sr = self.policy_eval(A.prior_eval_eps, prior_base, 0, kind='prior')
            sr0 = self.prior_sr
            self.curve.append({'ep': 0, 'sr': sr0, 'prior': sr0, 'actor_u': 0, 'prior_eval_eps': A.prior_eval_eps})
            print(f"RESULT_BASE priorSR={sr0:.3f} over {A.prior_eval_eps} episodes", flush=True)

        for ep in range(start_ep + 1, A.episodes + 1):
            CNT['episodes'] = ep
            if ep > A.warmup_eps and not self.online:
                self.critic_pretrain_burst()
                self.online = True
                if A.save_handoff:
                    self.online = False             # a handoff precedes the first actor update
                    self.save_handoff(A.save_handoff)
                    print('[handoff] stopping before the first actor update', flush=True)
                    return
            if self.func_on and self.online:
                func_pre = self.decode_bank(); theta_pre = [p.detach().clone() for p in self.dec_params]
            hist = self.env_reset(); done = False; steps = 0; ep_r = 0.0; ep_d = []
            last_qa = last_al = float('nan'); last_dg = {}
            while not done and steps < self.max_steps():
                s = self.obs_to_feat(hist)
                w = torch.randn(self.nd, device=self.dev)
                a = self.decode(s[None], w[None])[0]
                pairs, r, done, sc = self.env_macro(a.cpu().numpy())
                hist += pairs; steps += self.NAS; ep_r += r; CNT['env_steps'] += self.NAS
                s2 = self.obs_to_feat(hist)
                d = float(self._terminated) if self.oneshot else 0.0
                ep_d.append((r, d))
                self.push(s.cpu().numpy(), a.cpu().numpy(), r, s2.cpu().numpy(), d)
                if self.online:
                    for _ in range(A.utd):
                        last_qa, last_al, last_dg = self.train_step()
            if self.oneshot:
                n_r = sum(r > 0 for r, _ in ep_d); n_d = sum(d > 0 for _, d in ep_d)
                assert n_r <= 1 and n_d <= 1 and (n_r == 0 or ep_d[-1] == (1.0, 1.0)) \
                    and (self.adroit or n_d == n_r), f'ep {ep}: one-shot reward/terminal invariant broken'
                self.n_paid += n_r
            if self.func_on and self.online:
                self.func_step(ep, func_pre, theta_pre)
            self.ep_rews.append(ep_r)
            if last_dg:
                self.log_train(phase='online', ep=ep, ep_r=ep_r, qa_loss=last_qa, a_loss=last_al, **last_dg, **CNT)
            if ep % 10 == 0:
                print(f"[ep {ep}] ep_r {ep_r:.1f} replay {self.nbuf} qa {last_qa:.3f} al {last_al:.3f} "
                      f"u={CNT['actor_updates']} cu={CNT['critic_updates_total']} "
                      f"elapsed {time.time() - t0:.0f}s"
                      + (f" | rewarded episodes {self.n_paid}/{ep - start_ep}" if self.oneshot else ''), flush=True)
            if ep >= A.eval_from and ep % A.eval_every == 0:
                te = time.time()
                sr = self.policy_eval(A.eval_eps, A.eval_seed_base, ep, kind='policy')
                srp = self.prior_sr
                self.curve.append({'ep': ep, 'sr': sr, 'prior': srp, 'sr_minus_prior': sr - srp,
                                   'actor_u': CNT['actor_updates'], 'critic_u': CNT['critic_updates_total']})
                print(f"RESULT_ONLINE ep={ep} SR={sr:.3f} priorSR={srp:.3f} d={sr - srp:+.3f} "
                      f"u={CNT['actor_updates']}", flush=True)
                if ep in self.holdout_at:
                    sm = self.policy_eval(A.final_eval_eps, A.final_eval_seed_base, ep, kind='holdout')
                    self.curve.append({'ep': ep, 'holdout_sr': sm, 'actor_u': CNT['actor_updates']})
                    print(f"RESULT_HOLDOUT ep={ep} holdout_SR={sm:.3f} over {A.final_eval_eps} episodes", flush=True)
                json.dump(self.curve, open(f'{A.out}/curve.json', 'w'))
                if A.full_snapshot_every_ep and ep % A.full_snapshot_every_ep == 0:
                    self.save_handoff(f'{A.out}/resume_ep{ep}.pt', kind='periodic', allow_actor_state=True)
                self.log_perf(ep=ep, eval_seconds=time.time() - te, elapsed=time.time() - t0,
                              gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                              env_steps=CNT['env_steps'], actor_u=CNT['actor_updates'])

        # final 100-episode holdout on a disjoint seed range
        srF = self.policy_eval(A.final_eval_eps, A.final_eval_seed_base, A.episodes, kind='final')
        print(f"RESULT_FINAL ep={A.episodes} holdout_SR={srF:.3f} over {A.final_eval_eps} episodes", flush=True)
        self.curve.append({'ep': A.episodes, 'final_holdout_sr': srF, 'actor_u': CNT['actor_updates']})
        json.dump(self.curve, open(f'{A.out}/curve.json', 'w'))
        torch.save({'unet': self.mf.unet.state_dict(), 'QA': self.QA.state_dict(), 'QAt': self.QAt.state_dict(),
                    'oqa': self.oqa.state_dict(), 'opt_dec': self.opt_dec.state_dict(), 'CNT': dict(CNT),
                    'args': vars(A)}, f'{A.out}/final_ep{A.episodes}.pt')
        print(f"DONE {self.tag} u={CNT['actor_updates']} cu={CNT['critic_updates_total']} "
              f"env_steps={CNT['env_steps']} elapsed={time.time() - t0:.0f}s", flush=True)


def resolve_args(argv=None):
    A = build_parser().parse_args(argv)
    if A.reward is None:
        A.reward = 'oneshot' if A.task in adroit_env.TASKS else 'indicator'
    if A.task in adroit_env.TASKS and A.reward != 'oneshot':
        raise SystemExit(f'--task {A.task} supports --reward oneshot only')
    A.ckpt = paths.require(A.ckpt, '--ckpt', 'SFG_PC_THETA0')
    A.demo = paths.require(A.demo, '--demo', 'SFG_PC_DEMO')
    if A.out is None:
        A.out = paths.default_out(A.tag or f'sfg_{A.task}_s{A.seed}')
        if A.out is None:
            raise SystemExit('--out not given and $SFG_PC_OUT is not set')
    if A.resume_handoff and not os.path.exists(A.resume_handoff):
        raise SystemExit(f'--resume_handoff {A.resume_handoff} does not exist')
    return A


def main(argv=None):
    A = resolve_args(argv)
    setup_simulator(A.task)
    SFGTrainer(A).run()


def setup_simulator(task):
    import warnings
    if task in adroit_env.TASKS:
        paths.setup_adroit()         # the Adroit modules are imported when the env is first built
        warnings.filterwarnings('ignore')
        return
    paths.setup_vulkan()
    paths.setup_dexart()
    warnings.filterwarnings('ignore')
    from sfg_pc.dexart_env import import_dexart
    import_dexart()                  # import the simulator before seeding, as its import order is fixed


if __name__ == '__main__':
    main()
