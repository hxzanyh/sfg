#!/bin/bash
# SFG fine-tuning of a pretrained one-step MeanFlow generator (paper Section 5).
#
#   bash scripts/sfg/train.sh <task> [--seed=N] [--gpu=N] [--section.key=value ...]
#
#   tasks: square | tool_hang | transport | kitchen-partial | kitchen-mixed | kitchen-complete
#
# Shared settings live in ogpo/configs/algos/sfg.yaml; this script adds the per-task values below.
# Any further argument is passed to ogpo/main.py unchanged, e.g. --agent.sfg_func_delta=0.005.
# The pretrained generator is read from $THETA0_DIR/<task>/params.pkl (default ./pretrained),
# or pass --checkpoint.restore_actor_path=<params_*.pkl>.
set -euo pipefail

TASK=${1:?usage: train.sh <task> [--seed=N] [--gpu=N] [overrides]}; shift
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

# per task: env | chunk H | discount | MLP widths (actor = critic) | actor lr | generator arch | freeze encoder | eval interval
UTD=1; STEPS=1400000
case $TASK in
  square)           ENV=square-mh-low_dim;       H=4; GAMMA=0.99;  DIMS="[512,512,512,512]";     LR=3e-6; ARCH=mlp;      FREEZE=false; EVAL=10000 ;;
  tool_hang)        ENV=tool_hang-ph-low_dim;    H=8; GAMMA=0.999; DIMS="[512,512,512,512,512]"; LR=3e-7; ARCH=octo_mlp; FREEZE=true;  EVAL=20000 ;;
  transport)        ENV=transport-mh-low_dim;    H=8; GAMMA=0.999; DIMS="[512,512,512,512,512]"; LR=4e-5; ARCH=octo_mlp; FREEZE=true;  EVAL=50000
                    UTD=4; STEPS=3000000 ;;
  kitchen-partial)  ENV=D4RL/kitchen/partial-v2;  H=4; GAMMA=0.99; DIMS="[512,512,512,512]";     LR=3e-6; ARCH=octo_mlp; FREEZE=true;  EVAL=20000 ;;
  kitchen-mixed)    ENV=D4RL/kitchen/mixed-v2;    H=4; GAMMA=0.99; DIMS="[512,512,512,512]";     LR=3e-6; ARCH=octo_mlp; FREEZE=true;  EVAL=20000 ;;
  kitchen-complete) ENV=D4RL/kitchen/complete-v2; H=4; GAMMA=0.99; DIMS="[512,512,512,512]";     LR=3e-6; ARCH=octo_mlp; FREEZE=true;  EVAL=20000 ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

THETA0="${THETA0_DIR:-$ROOT/pretrained}/$TASK/params.pkl"
# run name: task, seed and every override, so a variant never resumes another run's checkpoint
_tag() { local t=""; for a in "$@"; do a=${a#--}; a=${a#*.}; a=${a#sfg_}; a=${a//=/}; a=${a//[^A-Za-z0-9._-]/}; t+="_$a"; done; echo "$t"; }
RUN=${RUN_NAME:-sfg_${TASK}$(_tag "${EXTRA[@]}")_s${SEED}}
echo "[sfg] run: $RUN"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# rolling resume checkpoints (agent + replay buffer), one directory per run
export OGPO_CHECKPOINT_ROOT="${OGPO_CHECKPOINT_ROOT:-$ROOT/checkpoints/$RUN}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# PYTHON=/path/to/python uses an existing environment instead of `uv run`
exec ${PYTHON:-uv run python} ogpo/main.py --algo=sfg \
  --experiment.seed="$SEED" \
  --experiment.run_name="$RUN" --experiment.wandb_name="$RUN" --experiment.run_group="$TASK" \
  --experiment.save_dir="$ROOT/exp/$RUN" \
  --env.name="$ENV" \
  --agent.horizon_length=$H --agent.discount=$GAMMA \
  --agent.actor_hidden_dims="$DIMS" --agent.value_hidden_dims="$DIMS" \
  --agent.meanflow_arch=$ARCH --agent.sfg_freeze_obs_encoder=$FREEZE \
  --agent.sfg_actor_lr=$LR \
  --buffer.utd_online=$UTD --buffer.utd_q=$UTD --buffer.utd_pi=$UTD \
  --training.online_steps=$STEPS --intervals.eval_interval=$EVAL \
  --checkpoint.restore_actor_path="$THETA0" \
  "${EXTRA[@]}"
