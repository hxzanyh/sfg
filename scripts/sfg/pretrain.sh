#!/bin/bash
# Behaviour-cloning pretraining of the one-step MeanFlow generator theta_0 (paper Appendix E).
#
#   bash scripts/sfg/pretrain.sh <task> [--seed=N] [--gpu=N] [--section.key=value ...]
#
# Recipe: MeanFlow objective with r = t on half of the batch, guidance weight 2 folded into the target,
# adaptive loss weight (p = 0.5, c = 1e-3); Adam lr 1e-4, no weight decay, batch 128, gradient-norm clip
# 1.0, EMA 0.999 (the EMA weights are evaluated and saved); 200k gradient steps; observations and actions
# limits-normalised to [-1, 1]. A checkpoint (params_<step>.pkl) is saved and evaluated on 64 episodes
# every 10k steps; pick theta_0 by rollout success, never by the training loss. Then:
#   mkdir -p pretrained/<task> && cp exp/<run>/.../params_<step>.pkl pretrained/<task>/params.pkl
set -euo pipefail

TASK=${1:?usage: pretrain.sh <task> [--seed=N] [--gpu=N] [overrides]}; shift
SEED=1; GPU=${GPU:-0}; EXTRA=()
for a in "$@"; do
  case $a in
    --seed=*) SEED=${a#*=} ;;
    --gpu=*)  GPU=${a#*=} ;;
    *)        EXTRA+=("$a") ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case $TASK in
# generator architecture must match scripts/sfg/train.sh: a plain MeanFlow MLP on square, an MLP with its
# own state encoder (octo_mlp) elsewhere
  square)           ENV=square-mh-low_dim;       H=4; GAMMA=0.99;  DIMS="[512,512,512,512]";     ARCH=mlp ;;
  tool_hang)        ENV=tool_hang-ph-low_dim;    H=8; GAMMA=0.999; DIMS="[512,512,512,512,512]"; ARCH=octo_mlp ;;
  transport)        ENV=transport-mh-low_dim;    H=8; GAMMA=0.999; DIMS="[512,512,512,512,512]"; ARCH=octo_mlp ;;
  kitchen-partial)  ENV=D4RL/kitchen/partial-v2;  H=4; GAMMA=0.99; DIMS="[512,512,512,512]";     ARCH=octo_mlp ;;
  kitchen-mixed)    ENV=D4RL/kitchen/mixed-v2;    H=4; GAMMA=0.99; DIMS="[512,512,512,512]";     ARCH=octo_mlp ;;
  kitchen-complete) ENV=D4RL/kitchen/complete-v2; H=4; GAMMA=0.99; DIMS="[512,512,512,512]";     ARCH=octo_mlp ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

# run name: task, seed and every override, so a variant never resumes another run's checkpoint
_tag() { local t=""; for a in "$@"; do a=${a#--}; a=${a#*.}; a=${a#sfg_}; a=${a//=/}; a=${a//[^A-Za-z0-9._-]/}; t+="_$a"; done; echo "$t"; }
RUN=${RUN_NAME:-meanflow_bc_${TASK}$(_tag "${EXTRA[@]}")_s${SEED}}
echo "[sfg] run: $RUN"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# PYTHON=/path/to/python uses an existing environment instead of `uv run`
exec ${PYTHON:-uv run python} ogpo/main.py --algo=sfg \
  --experiment.seed="$SEED" \
  --experiment.run_name="$RUN" --experiment.wandb_name="$RUN" --experiment.run_group="$TASK" \
  --experiment.save_dir="$ROOT/exp/$RUN" \
  --env.name="$ENV" \
  --agent.horizon_length=$H --agent.discount=$GAMMA \
  --agent.actor_hidden_dims="$DIMS" --agent.value_hidden_dims="$DIMS" \
  --agent.sfg=false --agent.meanflow_arch=$ARCH --agent.meanflow_bc_ema=0.999 \
  --agent.meanflow_flow_ratio=0.5 --agent.meanflow_cfg_scale=2.0 \
  --agent.batch_size=128 --agent.actor_lr=1e-4 --agent.actor_weight_decay=0.0 --agent.clip_grad_norm=1.0 \
  --agent.use_constant_scheduler_for_bc=true \
  --training.bc_pi_steps=200000 --training.online_steps=0 --training.bc_only=true --training.q_warmup_steps=0 \
  --bc.clip_bc=false \
  --intervals.eval_interval_bc=10000 --intervals.save_interval_bc=10000 --intervals.log_interval=1000 \
  "${EXTRA[@]}"
