#!/bin/bash
# SFG with the one-shot sparse reward (at most one reward per episode, the episode ends at the first
# success) on point-cloud Adroit door / pen / relocate and DexArt Bucket. One process per run:
# 100 warm-up episodes with theta_0, 10,000 critic-only updates, then the online phase.
#
#   bash run_oneshot.sh <door|pen|relocate|bucket> [--gpu=N] [--seed=N] [--func_delta=D] [extra args]
#
# theta_0 and demonstrations: $SFG_PC_THETA0 / $SFG_PC_DEMO if set, else
#   $SFG_PC_OUT/bc_<task>_s0/theta0.pt and $SFG_PC_OUT/data/adroit_<task>_expert.zarr (Adroit),
#   $SFG_PC_OUT/bc_bucket_s0/meanflow_final.pt and $SFG_PC_OUT/data/dexart_bucket_succ.zarr (bucket).
# Adroit needs $DP3_ROOT (patched, see README), bucket DexArt (pip-installed, or $DEXART_ROOT / $DP3_ROOT).
# Output: $SFG_PC_OUT/<task>_oneshot_fd<D>_s<seed>. PYTHON=... selects the interpreter.
set -euo pipefail

TASK=${1:?usage: run_oneshot.sh door|pen|relocate|bucket [--gpu=N] [--seed=N] [--func_delta=D] [extra args]}; shift
GPU=${GPU:-0}; SEED=1; FD=0.01; EXTRA=()
for a in "$@"; do
  case $a in
    --gpu=*)        GPU=${a#*=} ;;
    --seed=*)       SEED=${a#*=} ;;
    --func_delta=*) FD=${a#*=} ;;
    *)              EXTRA+=("$a") ;;
  esac
done
: "${SFG_PC_OUT:?set SFG_PC_OUT to the output root}"
case $TASK in
  door|pen|relocate)
    THETA0=${SFG_PC_THETA0:-$SFG_PC_OUT/bc_${TASK}_s0/theta0.pt}
    DEMO=${SFG_PC_DEMO:-$SFG_PC_OUT/data/adroit_${TASK}_expert.zarr}
    EPISODES=2300; HOLDOUT=300,800,1300,1800 ;;
  bucket)
    THETA0=${SFG_PC_THETA0:-$SFG_PC_OUT/bc_bucket_s0/meanflow_final.pt}
    DEMO=${SFG_PC_DEMO:-$SFG_PC_OUT/data/dexart_bucket_succ.zarr}
    EPISODES=2600; HOLDOUT=300,800,1300,1800,2300 ;;
  *) echo "task must be door|pen|relocate|bucket" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$GPU
PY=${PYTHON:-python}
NAME=${TASK}_oneshot_fd$(echo "$FD" | tr -d '.')_s$SEED
OUT=$SFG_PC_OUT/$NAME
mkdir -p "$OUT"
RESUME=()
latest=$(ls -1 "$OUT"/resume_ep*.pt 2>/dev/null | sed 's/.*resume_ep//;s/\.pt$//' | sort -n | tail -1 || true)
if [ -n "$latest" ]; then
  RESUME=(--resume_handoff "$OUT/resume_ep$latest.pt")
  echo "[run_oneshot] $NAME: resuming from $OUT/resume_ep$latest.pt"
fi
# theta_0 is evaluated once on the holdout seeds (prior_seed_base = final_eval_seed_base), so every
# holdout is paired with it episode by episode.
$PY -u -m sfg_pc.train --task "$TASK" --reward oneshot --latent normal --ckpt "$THETA0" --demo "$DEMO" \
  --func_delta "$FD" --seed "$SEED" --episodes $EPISODES --holdout_at $HOLDOUT \
  --warmup_eps 100 --critic_pretrain 10000 --eval_from 125 --eval_every 25 --eval_eps 20 \
  --prior_eval_eps 100 --final_eval_eps 100 --eval_seed_base 31337 --prior_seed_base 900000 \
  --final_eval_seed_base 900000 --full_snapshot_every_ep 200 \
  --out "$OUT" --tag "$NAME" ${RESUME[@]+"${RESUME[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$OUT.log"
