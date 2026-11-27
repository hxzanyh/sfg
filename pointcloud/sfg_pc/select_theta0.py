"""Pick theta_0 among the BC checkpoints of pretrain.py (Adroit).

Every candidate is evaluated with the online trainer's own evaluation (sfg_pc.train.SFGTrainer.policy_eval,
kind 'prior': free N(0, I) latents, episode i seeded with --seed_base + i) on a seed block disjoint from
the 900000 block of the online runs' prior/holdout evaluations. The argmax under --rule is copied to
--dest; ties go to the later checkpoint.

  --rule vrl3  success = episode goal total > 20 (pen) / > 25 (door, relocate), the mj_envs/VRL3 rule
  --rule ge10  success = goal total >= 10, the criterion of the online trainer
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np

from sfg_pc import adroit_env
from sfg_pc.train import resolve_args, setup_simulator, SFGTrainer

VRL3_THRESH = {'door': 25, 'pen': 20, 'relocate': 25}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', choices=adroit_env.TASKS, required=True)
    ap.add_argument('--bc_dir', required=True, help='pretrain.py --save_dir (holds meanflow_step<N>.pt)')
    ap.add_argument('--seed_base', type=int, default=700000)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--rule', choices=['vrl3', 'ge10'], default='vrl3')
    ap.add_argument('--dest', default=None, help='default <bc_dir>/theta0.pt')
    a = ap.parse_args(argv)
    cks = sorted((int(re.search(r'meanflow_step(\d+)\.pt$', p).group(1)), p)
                 for p in glob.glob(os.path.join(a.bc_dir, 'meanflow_step*.pt')))
    cks = [(s, p) for s, p in cks if s > 0]
    if not cks:
        raise SystemExit(f'no meanflow_step<N>.pt in {a.bc_dir} (pretrain.py --save_every)')
    setup_simulator(a.task)
    rows = []
    for step, p in cks:
        out = os.path.join(a.bc_dir, 'select', f'step{step}')
        A = resolve_args(['--task', a.task, '--ckpt', p, '--demo', p, '--out', out, '--func_delta', '0'])
        T = SFGTrainer(A, eval_only=True)
        T.policy_eval(a.episodes, a.seed_base, 0, kind='prior')
        g = np.array([r['goal_total'] for r in T.eval_records])
        sr = float((g > VRL3_THRESH[a.task]).mean() if a.rule == 'vrl3' else (g >= adroit_env.GOAL_THRESH).mean())
        rows.append({'step': step, 'ckpt': p, 'sr': sr, 'sr_ge10': float((g >= adroit_env.GOAL_THRESH).mean())})
        print(f"[select] step {step}: SR({a.rule}) = {sr:.3f}  SR(ge10) = {rows[-1]['sr_ge10']:.3f}", flush=True)
        del T
    best = max(rows, key=lambda r: (r['sr'], r['step']))
    dest = a.dest or os.path.join(a.bc_dir, 'theta0.pt')
    shutil.copyfile(best['ckpt'], dest)
    json.dump({'rule': a.rule, 'seed_base': a.seed_base, 'episodes': a.episodes, 'rows': rows,
               'selected': best}, open(os.path.join(a.bc_dir, 'select_theta0.json'), 'w'), indent=1)
    print(f"SELECTED step {best['step']} SR({a.rule}) = {best['sr']:.3f} -> {dest}", flush=True)


if __name__ == '__main__':
    main()
