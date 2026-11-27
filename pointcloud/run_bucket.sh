#!/bin/bash
# The three DexArt Bucket runs of Figure 6.
#
#   bash run_bucket.sh handoff  [--gpu=N] [--seed=N]   # warm-up with theta_0 + critic warm-up, saves the handoff
#   bash run_bucket.sh sfg      [--gpu=N] [--seed=N]   # SFG, delta = 0.01, from the handoff
#   bash run_bucket.sh nobound  [--gpu=N] [--seed=N]   # same without the bound (delta = 1e9), from the handoff
#   bash run_bucket.sh all      [--gpu=N] [--seed=N]   # the three, one after the other
#
# Needs $SFG_PC_THETA0 (theta_0 checkpoint), $SFG_PC_DEMO (demonstration zarr), $SFG_PC_OUT (output
# root) and DexArt importable (pip-installed, or $DEXART_ROOT / $DP3_ROOT). PYTHON=... selects the
# interpreter. Extra arguments are passed to sfg_pc.train. A run that has a resume_ep<N>.pt snapshot
# continues from the latest one.
set -euo pipefail

MODE=${1:?usage: run_bucket.sh handoff|sfg|nobound|all [--gpu=N] [--seed=N] [extra args]}; shift
GPU=${GPU:-0}; SEED=1; EXTRA=()
for a in "$@"; do
  case $a in
    --gpu=*)  GPU=${a#*=} ;;
    --seed=*) SEED=${a#*=} ;;
    *)        EXTRA+=("$a") ;;
  esac
done

: "${SFG_PC_THETA0:?set SFG_PC_THETA0 to the theta_0 checkpoint}"
: "${SFG_PC_DEMO:?set SFG_PC_DEMO to the demonstration zarr}"
: "${SFG_PC_OUT:?set SFG_PC_OUT to the output root}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$GPU
PY=${PYTHON:-python}
HANDOFF_DIR=$SFG_PC_OUT/bucket_handoff_s$SEED
HANDOFF=$HANDOFF_DIR/state.pt
# the Figure 6 runs: per-step indicator reward, latents on the sphere, DexArt instances cycling through evaluations
FIG6="--reward indicator --latent sphere --pin_eval_instances 0"

handoff() {
  # 100 warm-up episodes with theta_0 (after a 100-episode evaluation of theta_0), then 10,000
  # critic-only updates; stops before the first actor update.
  $PY -u -m sfg_pc.train --task bucket --seed "$SEED" --func_delta 0 $FIG6 \
    --save_handoff "$HANDOFF" --out "$HANDOFF_DIR" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$HANDOFF_DIR.log"
}

online() {  # online <name> <args...>
  local name=$1; shift
  local out=$SFG_PC_OUT/$name
  local from=$HANDOFF
  local latest
  latest=$(ls -1 "$out"/resume_ep*.pt 2>/dev/null | sed 's/.*resume_ep//;s/\.pt$//' | sort -n | tail -1 || true)
  if [ -n "$latest" ]; then
    from=$out/resume_ep$latest.pt
    echo "[run_bucket] $name: resuming from $from"
  fi
  [ -f "$from" ] || { echo "[run_bucket] no handoff at $from; run 'bash run_bucket.sh handoff' first" >&2; exit 1; }
  mkdir -p "$out"
  # 2,600 episodes in total (2,500 online), evaluation every 5 episodes from episode 105
  $PY -u -m sfg_pc.train --task bucket --seed "$SEED" --episodes 2600 $FIG6 \
    --resume_handoff "$from" --out "$out" --tag "$name" "$@" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$out.log"
}

mkdir -p "$SFG_PC_OUT" "$HANDOFF_DIR"
case $MODE in
  handoff) handoff ;;
  sfg)     online bucket_sfg_delta0.01_s$SEED --func_delta 0.01 --holdout_at 300,800,1600,2000 ;;
  # the unbounded run of the paper had its holdouts at episodes 300 and 1600 (plus the final one)
  nobound) online bucket_nobound_s$SEED --func_delta 1e9 --holdout_at 300,1600 ;;
  all)     handoff
           online bucket_sfg_delta0.01_s$SEED --func_delta 0.01 --holdout_at 300,800,1600,2000
           online bucket_nobound_s$SEED --func_delta 1e9 --holdout_at 300,1600 ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
