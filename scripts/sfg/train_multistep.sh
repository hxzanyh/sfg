#!/bin/bash
# SFG on a K-step flow-matching policy (paper Appendix G and D).
#
#   bash scripts/sfg/train_multistep.sh <task> [--K=10] [--m=M | --endpoint[=k0]] [--seed=N] [--gpu=N] [overrides]
#
#   tasks: square | tool_hang | transport | kitchen-partial | kitchen-mixed | kitchen-complete
#   --K=K          Euler steps of the decode (paper: 10; transport 3 or 4 in Appendix D)
#   --m=M          truncated backpropagation through the last M steps (Algorithm 2; default M = K)
#   --endpoint=k0  endpoint-prediction rule from step k0 on (Algorithm 3; default k0 = K // 2)
#
# The pretrained policy is a flow-matching policy trained by OGPO's behaviour cloning (see README),
# in OGPO's own observation/action normalisation (env.limits_norm=false).
# Read from $THETA0_DIR/<task>_flow/params.pkl (default ./pretrained) or pass --checkpoint.restore_actor_path.
set -euo pipefail

TASK=${1:?usage: train_multistep.sh <task> [--K=10] [--m=M | --endpoint[=k0]] [overrides]}; shift
SEED=1; GPU=${GPU:-0}; K=10; M=0; MODE=bptt; K0=-1; EXTRA=()
for a in "$@"; do
  case $a in
    --seed=*)     SEED=${a#*=} ;;
    --gpu=*)      GPU=${a#*=} ;;
    --K=*)        K=${a#*=} ;;
    --m=*)        M=${a#*=} ;;
    --endpoint)   MODE=endpoint ;;
    --endpoint=*) MODE=endpoint; K0=${a#*=} ;;
    *)            EXTRA+=("$a") ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case $TASK in
  square)           ENV=square-mh-low_dim;       H=4; GAMMA=0.99;  DIMS="[512,512,512,512]" ;;
  tool_hang)        ENV=tool_hang-ph-low_dim;    H=8; GAMMA=0.999; DIMS="[512,512,512,512,512]" ;;
  transport)        ENV=transport-mh-low_dim;    H=8; GAMMA=0.999; DIMS="[512,512,512,512,512]" ;;
  kitchen-partial)  ENV=D4RL/kitchen/partial-v2;  H=4; GAMMA=0.99; DIMS="[512,512,512,512]" ;;
  kitchen-mixed)    ENV=D4RL/kitchen/mixed-v2;    H=4; GAMMA=0.99; DIMS="[512,512,512,512]" ;;
  kitchen-complete) ENV=D4RL/kitchen/complete-v2; H=4; GAMMA=0.99; DIMS="[512,512,512,512]" ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

THETA0="${THETA0_DIR:-$ROOT/pretrained}/${TASK}_flow/params.pkl"
if [ "$MODE" = endpoint ]; then TAG="K${K}_endpoint"; else TAG="K${K}_m${M}"; fi
# run name: task, seed and every override, so a variant never resumes another run's checkpoint
_tag() { local t=""; for a in "$@"; do a=${a#--}; a=${a#*.}; a=${a#sfg_}; a=${a//=/}; a=${a//[^A-Za-z0-9._-]/}; t+="_$a"; done; echo "$t"; }
RUN=${RUN_NAME:-sfg_${TASK}_${TAG}$(_tag "${EXTRA[@]}")_s${SEED}}
echo "[sfg] run: $RUN"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OGPO_CHECKPOINT_ROOT="${OGPO_CHECKPOINT_ROOT:-$ROOT/checkpoints/$RUN}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Appendix G: delta 0.01, tau 0.05, actor lr 3e-7, 50k warm-up steps then 400k critic updates,
# 4 critic + 4 actor updates per env step, 1.4M online steps, evaluation every 40k steps.
# PYTHON=/path/to/python uses an existing environment instead of `uv run`
exec ${PYTHON:-uv run python} ogpo/main.py --algo=sfg \
  --experiment.seed="$SEED" \
  --experiment.run_name="$RUN" --experiment.wandb_name="$RUN" --experiment.run_group="$TASK" \
  --experiment.save_dir="$ROOT/exp/$RUN" \
  --env.name="$ENV" --env.limits_norm=false \
  --agent.horizon_length=$H --agent.discount=$GAMMA \
  --agent.actor_hidden_dims="$DIMS" --agent.value_hidden_dims="$DIMS" \
  --agent.use_meanflow=false --agent.sfg_flow_decode=true --agent.flow_steps=$K \
  --agent.time_embedding=sinusoidal --agent.sfg_freeze_obs_encoder=false \
  --agent.sfg_grad_mode=$MODE --agent.sfg_grad_steps=$M --agent.sfg_endpoint_from=$K0 \
  --agent.sfg_proj_bisect=true --agent.sfg_actor_lr=3e-7 \
  --training.q_warmup_steps=50000 --buffer.utd_warmup=8 \
  --buffer.utd_online=4 --buffer.utd_q=4 --buffer.utd_pi=4 \
  --intervals.eval_interval=40000 \
  --checkpoint.restore_actor_path="$THETA0" \
  "${EXTRA[@]}"
