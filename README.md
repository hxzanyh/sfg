# SFG: Simple Flow Gradient — reproduction

Built on [OGPO](https://github.com/simchowitzlabpublic/OGPO_public) (MIT). The point-cloud experiments
(DexArt Bucket, Figure 6; one-shot sparse reward on Adroit door / pen / relocate and Bucket) are in
[`pointcloud/`](pointcloud/README.md).

## Install

```bash
uv sync --extra cuda12 --extra robomimic     # or --extra cuda13
```

`uv.lock` pins the versions of the paper's runs (JAX 0.10.1, MuJoCo 3.9.0, robosuite 1.5.1,
gymnasium-robotics 1.4.2, Minari 0.5.3). The launchers use `uv run`; `PYTHON=/path/to/python` uses an
existing environment instead.

## Data

- Robomimic low-dim (`square-mh`, `tool_hang-ph`, `transport-mh`): `~/.robomimic/<task>/<type>/*.hdf5`, or
  set `$ROBOMIMIC_DATASET_ROOT`.
- Franka-Kitchen (`D4RL/kitchen/{partial,mixed,complete}-v2`): downloaded by Minari on first use.

## Pretrained generator θ₀

Place the checkpoint at `pretrained/<task>/params.pkl` (`<task>_flow` for the multi-step version), or train
it (one-step MeanFlow, Appendix E):

```bash
bash scripts/sfg/pretrain.sh <task> --gpu=0
# pick theta_0 by rollout success among exp/meanflow_bc_<task>_s1/.../params_<step>.pkl, then
mkdir -p pretrained/<task> && cp <that params_<step>.pkl> pretrained/<task>/params.pkl
```

Tasks: `square`, `tool_hang`, `transport`, `kitchen-partial`, `kitchen-mixed`, `kitchen-complete`.

## Main results (Figure 3, Table 1)

```bash
for s in 1 2 3 4; do bash scripts/sfg/train.sh <task> --seed=$s --gpu=0; done
uv run python scripts/sfg/summarize.py exp/sfg_<task>_s*     # first-0.3M / last-200k / steps to 0.8
```

Kitchen is scored as `tasks_completed / 4`.

## Multi-step version (Appendix G and D)

```bash
bash scripts/sfg/train_multistep.sh tool_hang --K=10 --m=5          # truncated backprop through the last m steps
bash scripts/sfg/train_multistep.sh kitchen-partial --K=10 --endpoint # endpoint prediction
bash scripts/sfg/train_multistep.sh transport --K=3 --m=3
```

## Ablations (Table 2, Appendix C)

Append to `bash scripts/sfg/train.sh <task>`:

| ablation | arguments |
|---|---|
| no bound | `--agent.sfg_func_delta=0` |
| bound size | `--agent.sfg_func_delta=0.005` / `0.05` |
| learning rate | `--agent.sfg_actor_lr=<lr>` |
| without demonstration replay | `--buffer.offline_ratio=0` |
| from a random initialisation | `--agent.sfg_allow_scratch=true --checkpoint.restore_actor_path=null` |
| straight-through clip | `--agent.sfg_act_ste=true` |
| fixed BC anchor (Section 3) | `--agent.sfg_func_delta=0 --agent.sfg_anchor_beta=<beta>` |
| penalty instead of projection | `--agent.sfg_func_delta=0 --agent.sfg_ref_pen=<lambda>` |
| parameter-space trust region | `--agent.sfg_func_delta=0 --agent.sfg_param_delta=<eps>` |
| SGD actor | `--agent.sfg_actor_opt=sgd --agent.sfg_actor_lr=<lr>` |
| longer critic warm-up | `--training.q_warmup_steps=50000` or `--buffer.utd_warmup=20` |

Section 3 runs: `--agent.sfg_actor_lr=4.5e-5 --agent.tau=0.005`, δ = 0.05 where bounded.

Overrides become part of the run name; a rerun of the same command resumes from `checkpoints/<run>/`.
Outputs per run: `flags.json`, `eval.csv`, `online_agent.csv`, `sfg_steps.jsonl` (per-episode functional
step, blend factor, path length, net distance), `params_<step>.pkl`.
