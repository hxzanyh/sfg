"""Summarise SFG runs the way the paper's tables do.

    uv run python scripts/sfg/summarize.py exp/sfg_tool_hang_s1 [exp/sfg_tool_hang_s2 ...]

Per run: the score over the first 0.3M online steps and over the last 200k, the final score, and the first
online step at which the score reaches 0.8. The score is the success rate, except on Franka-Kitchen where
it is tasks_completed / 4 (success there requires all four subtasks). Online steps are evaluation labels
minus the warm-up steps W (the labels count the warm-up). Also summarises sfg_steps.jsonl: the median
committed functional step and the fraction of episodes whose step was projected.
"""
import argparse
import glob
import json
import os

import numpy as np


def read_eval(path):
    """eval.csv -> dict column -> list of floats. Kitchen rows hold unquoted lists in their first columns,
    so values are taken from the right, where every column is a scalar."""
    with open(path) as f:
        lines = f.read().splitlines()
    header = lines[0].split(',')
    cols = {name: [] for name in header}
    for line in lines[1:]:
        parts = line.split(',')
        for j, name in enumerate(reversed(header)):
            try:
                cols[name].append(float(parts[-1 - j]))
            except (ValueError, IndexError):
                cols[name].append(float('nan'))
    return {k: np.asarray(v) for k, v in cols.items()}


def summarize(run_dir, early=300_000, late=200_000, target=0.8):
    evals = glob.glob(os.path.join(run_dir, '**', 'eval.csv'), recursive=True)
    flags = glob.glob(os.path.join(run_dir, '**', 'flags.json'), recursive=True)
    if not evals or not flags:
        return None
    cfg = json.load(open(flags[0]))
    warmup = int(cfg['training'].get('q_warmup_steps', 0))
    e = read_eval(evals[0])
    kitchen = 'tasks_completed' in e
    score = e['tasks_completed'] / 4.0 if kitchen else e['success']
    step = e['step'] - warmup
    end = step.max()
    hit = step[score >= target]
    out = {
        'run': os.path.basename(os.path.normpath(run_dir)),
        'metric': 'tasks/4' if kitchen else 'success',
        'first_eval': float(score[np.argmin(step)]),     # after the first eval interval, not theta_0
        f'first_{early // 1000}k': float(score[step <= early].mean()),
        f'last_{late // 1000}k': float(score[step > end - late].mean()),
        'final': float(score[np.argmax(step)]),
        f'steps_to_{target}': int(hit.min()) if len(hit) else None,
        'online_steps': int(end),
    }
    logs = glob.glob(os.path.join(run_dir, '**', 'sfg_steps.jsonl'), recursive=True)
    if logs:
        rows = [json.loads(l) for l in open(logs[0]) if l.strip()]
        if rows:
            out['median_step'] = float(np.median([r['step_committed'] for r in rows]))
            out['projected_frac'] = float(np.mean([bool(r['projected']) for r in rows]))
            out['net_distance'] = float(rows[-1]['net_distance'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='run directories (exp/<run>)')
    args = ap.parse_args()
    rows = [r for r in (summarize(d) for d in args.runs) if r is not None]
    if not rows:
        raise SystemExit('no eval.csv / flags.json found')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    print('\t'.join(keys))
    for r in rows:
        print('\t'.join(f'{r[k]:.3f}' if isinstance(r.get(k), float) else str(r.get(k)) for k in keys))


if __name__ == '__main__':
    main()
