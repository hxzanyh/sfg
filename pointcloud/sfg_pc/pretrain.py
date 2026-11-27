"""MeanFlow behaviour cloning of the point-cloud generator theta_0 (DexArt or point-cloud Adroit).

Common: batch 128, AdamW (lr 1e-4) on generator + encoder, gradient-norm clip 1, MeanFlow loss with
guidance scale 2 and flow ratio 0.5, Gaussian latents, 3 of 4 predicted actions executed per decode.
Task defaults (each can be overridden):
  DexArt (bucket theta_0): 5,000 steps on the chunks of successful episodes (meta/episode_outcome == 1),
    weight decay 0.01, one 100-episode evaluation after the last step.
  Adroit (door / pen / relocate theta_0): 12,000 steps on all chunks of the 90 expert episodes, weight
    decay 0, a 20-episode evaluation and a weights-only checkpoint every 1,000 steps; theta_0 is then
    picked among the checkpoints with select_theta0.py.
DexArt evaluation draws from the global numpy stream (DexArt's reset noise), so --eval_interval > 0
changes the sequence of training batches that follows it.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from sfg_pc import adroit_env, dexart_env, paths
from sfg_pc.meanflow.agent import MeanFlowAgent
from sfg_pc.meanflow.data import DemoZarrDataset


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', choices=dexart_env.TASKS + adroit_env.TASKS, default='bucket')
    ap.add_argument('--demo', default=os.environ.get('SFG_PC_DEMO'),
                    help='demonstration zarr (default $SFG_PC_DEMO)')
    ap.add_argument('--max_episodes', type=int, default=None)
    ap.add_argument('--steps', type=int, default=None, help='default 5000 (DexArt) / 12000 (Adroit)')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=None, help='AdamW weight decay (default 0.01 DexArt / 0 Adroit)')
    ap.add_argument('--success_only', type=int, default=None,
                    help='train on the chunks of successful episodes only (default 1 DexArt / 0 Adroit)')
    ap.add_argument('--n_obs_steps', type=int, default=2)
    ap.add_argument('--horizon', type=int, default=4)
    ap.add_argument('--n_action_steps', type=int, default=3, help='actions executed per decode at evaluation')
    ap.add_argument('--cfg_scale', type=float, default=2.0)
    ap.add_argument('--flow_ratio', type=float, default=0.5)
    ap.add_argument('--latent_dist', default='normal', choices=['normal', 'sphere'])
    ap.add_argument('--eval_interval', type=int, default=None,
                    help='evaluate every N steps (0 = only once, after the last step; default 0 DexArt / 1000 Adroit)')
    ap.add_argument('--eval_episodes', type=int, default=None, help='default 100 (DexArt) / 20 (Adroit)')
    ap.add_argument('--save_every', type=int, default=None,
                    help='also save meanflow_step<N>.pt (weights only) every N steps and at step 0 '
                         '(default 0 DexArt / 1000 Adroit)')
    ap.add_argument('--save_dir', default=None, help='default $SFG_PC_OUT/bc_<task>_s<seed>')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    return ap


def main(argv=None):
    A = build_parser().parse_args(argv)
    adroit = A.task in adroit_env.TASKS
    for k, dex, adr in (('steps', 5000, 12000), ('weight_decay', 0.01, 0.0), ('success_only', 1, 0),
                        ('eval_interval', 0, 1000), ('eval_episodes', 100, 20), ('save_every', 0, 1000)):
        if getattr(A, k) is None:
            setattr(A, k, adr if adroit else dex)
    A.demo = paths.require(A.demo, '--demo', 'SFG_PC_DEMO')
    if A.save_dir is None:
        A.save_dir = paths.default_out(f'bc_{A.task}_s{A.seed}')
        if A.save_dir is None:
            raise SystemExit('--save_dir not given and $SFG_PC_OUT is not set')
    if adroit:
        paths.setup_adroit()
    else:
        paths.setup_vulkan()
        paths.setup_dexart()
    import warnings; warnings.filterwarnings('ignore')
    dev = A.device

    torch.manual_seed(A.seed); np.random.seed(A.seed)
    ds = DemoZarrDataset(A.demo, max_episodes=A.max_episodes, n_obs_steps=A.n_obs_steps, horizon=A.horizon)
    print(f"[data] {A.demo} | success fraction of chunks = {float(np.mean(ds.valid_outcomes)):.3f}", flush=True)

    cfg = dict(action_dim=ds.action_dim, horizon=A.horizon, state_dim=ds.state_dim,
               n_obs_steps=A.n_obs_steps, latent_dist=A.latent_dist, flow_ratio=A.flow_ratio,
               cfg_scale=A.cfg_scale, mf_method='jit_mf', lr=A.lr, grad_clip=1.0,
               unet_down_features=(512, 1024, 2048), unet_kernel_size=5, unet_n_groups=8,
               unet_time_features=128, pn_out_channels=64, pn_block_channels=(64, 128, 256),
               state_hidden_dims=(64,), state_out_dim=64, ema_decay=0.0)
    mf = MeanFlowAgent(cfg, device=dev)
    params = mf.parameters()
    opt = mf.optimizer = torch.optim.AdamW(params, lr=A.lr, weight_decay=A.weight_decay)
    print(f"=== MeanFlow BC | task={A.task} seed={A.seed} steps={A.steps} obs_dim={mf.obs_encoder.obs_dim} "
          f"params={sum(p.numel() for p in params):,} ===", flush=True)

    os.makedirs(A.save_dir, exist_ok=True)
    json.dump(vars(A), open(os.path.join(A.save_dir, 'args.json'), 'w'), indent=2)

    def todev(b):
        return {k: (v.to(dev) if torch.is_tensor(v) else
                    ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) else v))
                for k, v in b.items()}

    env = None

    def do_eval(step):
        nonlocal env
        if A.eval_episodes <= 0:
            return float('nan')
        mod = adroit_env if adroit else dexart_env
        if env is None:
            env = mod.make_env(A.task)
        ev = mod.evaluate_bc(mf, env, A.eval_episodes, n_action_steps=A.n_action_steps, normalizer=ds.normalizer)
        print(f"[eval] step={step} SR={ev['success_rate']:.4f} (nep={A.eval_episodes})", flush=True)
        return ev['success_rate']

    def save_weights(path):
        torch.save({'unet': mf.unet.state_dict(), 'obs_encoder': mf.obs_encoder.state_dict(),
                    'ema_unet': None, 'ema_encoder': None, 'optimizer': None, 'step': mf.step_count,
                    'cfg': mf.cfg, 'normalizer': ds.normalizer.stats}, path)
        print(f"saved {path}", flush=True)

    if A.save_every > 0:
        save_weights(os.path.join(A.save_dir, 'meanflow_step0.pt'))
    last_sr = None
    for step in range(1, A.steps + 1):
        b = todev(ds.sample(A.batch_size))
        obs = b['observations']
        actions_flat = b['actions'].reshape(b['actions'].shape[0], -1).float()
        mf.unet.train(); mf.obs_encoder.train()
        obs_enc = mf.obs_encoder(obs)
        if A.success_only:
            succ = b['outcome'].float() > 0.5
            bc_loss = torch.zeros((), device=dev)
            if succ.any():
                bc_loss = mf.compute_jit_mf_loss(obs_enc[succ], actions_flat[succ])
        else:
            succ = torch.ones(len(actions_flat), dtype=torch.bool, device=dev)
            bc_loss = mf.compute_jit_mf_loss(obs_enc, actions_flat)
        opt.zero_grad(); bc_loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); mf.step_count += 1

        if A.save_every > 0 and step % A.save_every == 0:
            save_weights(os.path.join(A.save_dir, f'meanflow_step{step}.pt'))
        if step % 500 == 0 or step == 1:
            print(f"  step {step}: bc={bc_loss.item():.4f} n_succ={int(succ.sum().item())}/{A.batch_size}", flush=True)
        if A.eval_interval > 0 and step % A.eval_interval == 0:
            last_sr = do_eval(step)

    if last_sr is None:
        last_sr = do_eval(A.steps)
    mf.save(os.path.join(A.save_dir, 'meanflow_final.pt'), normalizer=ds.normalizer)
    print(f"RESULT task={A.task} seed={A.seed} final_SR={last_sr:.4f} -> "
          f"{os.path.join(A.save_dir, 'meanflow_final.pt')}", flush=True)


if __name__ == '__main__':
    main()
