# SFG on point clouds: DexArt Bucket (Figure 6) and the one-shot sparse reward on Adroit / Bucket — reproduction

PyTorch, independent of the JAX code in the repository root.

## Install

The paper's runs used Python 3.11, PyTorch 2.5.1 (CUDA 12.1), NumPy 1.26.4, zarr 2.12.0, numcodecs 0.11.0,
SAPIEN 2.2.1, gym 0.22.0 and transforms3d 0.4.2, on one 24 GB GPU (peak allocation below 15 GB; about
19 hours per 2,500-episode run).

1. Install [3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) following its
   `INSTALL.md` up to and including DexArt: SAPIEN, `third_party/dexart-release` (`pip install -e .`) and the
   DexArt assets (unpacked into `third_party/dexart-release/assets`, which also holds DexArt's released PPO
   experts). Only DexArt is used; DP3's policy code is not imported. `pytorch3d_simplified` is needed only
   by `gen_demos.py`, and only for point clouds of more than 1024 points.
2. In the same environment: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`
   and `pip install numpy==1.26.4 zarr==2.12.0 numcodecs==0.11.0 transforms3d termcolor tqdm cloudpickle`
   (gym comes with the DP3 installation; the runs used gym 0.22.0).
3. Run from this directory (`cd pointcloud`), or add it to `PYTHONPATH`.

For Adroit (door, pen, relocate), also follow DP3's `INSTALL.md` for MuJoCo 2.1 / mujoco-py and
`third_party/rrl-dependencies` (`mj_envs`, `mjrl`), download the VRL3 expert checkpoints into
`third_party/VRL3/ckpts/`, and apply the two patches from the 3D-Diffusion-Policy root:

```bash
cd $DP3_ROOT && patch -p1 < <this repo>/pointcloud/patches/dp3_relocate_pointcloud.patch \
             && patch -p1 < <this repo>/pointcloud/patches/vrl3_demo_rewards.patch
```

The first adds a point-cloud configuration for relocate (DP3 has none); the second makes the VRL3 demo
generator store the per-step reward, goal counter (`n_goal_achieved`) and episode-end flag. MuJoCo renders
headless with `MUJOCO_GL=egl` (set by default) and needs `~/.mujoco/mujoco210/bin` on `LD_LIBRARY_PATH`.

Headless SAPIEN needs the NVIDIA Vulkan ICD; if it is not found automatically, set
`VK_ICD_FILENAMES` (e.g. to `/usr/share/vulkan/icd.d/nvidia_icd.json`).

Paths are read from environment variables only:

| variable | meaning |
|---|---|
| `SFG_PC_DEMO` | demonstration zarr (default of `--demo`) |
| `SFG_PC_THETA0` | pretrained θ₀ checkpoint (default of `--ckpt`) |
| `SFG_PC_OUT` | output root (default parent of `--out` / `--save_dir`) |
| `DEXART_ROOT` | dexart-release checkout; optional if `dexart` is pip-installed, required by `gen_demos.py` |
| `DP3_ROOT` | 3D-Diffusion-Policy checkout; `DEXART_ROOT` defaults to `$DP3_ROOT/third_party/dexart-release`; required for Adroit |

## Data

`dexart_bucket_succ.zarr` holds 50 successful Bucket episodes (4,894 environment steps) of a DexArt-released
PPO expert acting deterministically; only episodes that reach `is_eval_done` are kept. Per step it stores
the environment's point cloud (1024 × 6, xyz + colour; only xyz is used), the 33-d state, the 22-d action,
reward = `is_eval_done` (1,652 rewarded steps; on Bucket the flag stays on for about a third of each
episode) and a terminal flag on the last step; `meta/` has the episode ends and outcomes. To generate it:

```bash
python -m sfg_pc.gen_demos --task bucket --n_success 50 --out $SFG_PC_OUT/data/dexart_bucket_succ.zarr
export SFG_PC_DEMO=$SFG_PC_OUT/data/dexart_bucket_succ.zarr
```

We could not determine which of DexArt's released Bucket PPO checkpoints and which seed produced the file
used in the paper; `gen_demos.py` defaults to `bucket_seg_dam_0.zip`, and the file's layout and statistics
are those this generator produces. DexArt seeds its environment noise from OS entropy, so a regenerated
file is a new sample of the same distribution, not a copy.

## Pretraining θ₀

```bash
python -m sfg_pc.pretrain --task bucket --seed 0 --save_dir $SFG_PC_OUT/bc_bucket_s0
export SFG_PC_THETA0=$SFG_PC_OUT/bc_bucket_s0/meanflow_final.pt
```

The defaults are the recipe of the paper's θ₀: MeanFlow behaviour cloning (guidance scale 2, flow ratio 0.5,
adaptive L2, Gaussian latents) on the success chunks, 5,000 steps of batch 128, AdamW (lr 1e-4, weight decay
0.01) on generator and encoder, gradient-norm clip 1, followed by one 100-episode evaluation. The paper's θ₀
scored 0.45 in that evaluation and 0.47 in the online trainer's own 100-episode evaluation of θ₀ (normalised
latents, fixed seeds).

## Figure 6

```bash
export SFG_PC_DEMO=... SFG_PC_THETA0=... SFG_PC_OUT=...
bash run_bucket.sh handoff  --gpu=0
bash run_bucket.sh sfg      --gpu=0      # the two online runs can run in parallel on two GPUs
bash run_bucket.sh nobound  --gpu=1
```

Equivalently, with `python -m sfg_pc.train` (`F6="--latent sphere --pin_eval_instances 0"`, all other
settings are the defaults):

```bash
python -m sfg_pc.train $F6 --func_delta 0 --save_handoff $SFG_PC_OUT/bucket_handoff_s1/state.pt --out $SFG_PC_OUT/bucket_handoff_s1
python -m sfg_pc.train $F6 --func_delta 0.01 --resume_handoff $SFG_PC_OUT/bucket_handoff_s1/state.pt --out $SFG_PC_OUT/bucket_sfg_delta0.01_s1
python -m sfg_pc.train $F6 --func_delta 1e9 --holdout_at 300,1600 --resume_handoff $SFG_PC_OUT/bucket_handoff_s1/state.pt --out $SFG_PC_OUT/bucket_nobound_s1
```

The reward of these runs is the per-step indicator `is_eval_done` (`--reward indicator`, the DexArt default).

`--resume_handoff` also accepts the periodic snapshots `resume_ep<N>.pt`; `run_bucket.sh` continues from
the latest one automatically. Without `--save_handoff`/`--resume_handoff` a single process runs warm-up,
critic warm-up and the online phase. `--task faucet|toilet` runs the same recipe on DexArt Faucet / Toilet
(not part of Figure 6).

## One-shot sparse reward: door, pen, relocate, Bucket

`--reward oneshot`: each episode carries at most one reward, 1 on the step at which it first succeeds
(Adroit: the cumulative goal counter reaches 10; Bucket: the first `is_eval_done`), and ends there (d = 1).
A pen drop also ends the episode with d = 1 and r = 0; time limits are truncations. In the demonstrations
the row that contains the first success gets r = 1, d = 1, and later rows are dropped. Latents are
N(0, I) (`--latent normal`).

Adroit demonstrations (90 successful VRL3-expert episodes per task, with the patched generator):

```bash
cd $DP3_ROOT/third_party/VRL3/src
for t in door pen relocate; do
  python gen_demonstration_expert.py --env_name $t --num_episodes 90 --root_dir $SFG_PC_OUT/data \
    --expert_ckpt_path ../ckpts/vrl3_$t.pt --img_size 84 --not_use_multi_view --use_point_crop --min_episode_length 0
done                     # -> $SFG_PC_OUT/data/adroit_<task>_expert.zarr
```

Adroit θ₀ (MeanFlow BC, 12,000 steps on all chunks, AdamW lr 1e-4 without weight decay, a checkpoint every
1,000 steps), then the checkpoint with the highest success on 100 episodes of seed block 700000 (disjoint
from the 900000 block of the online evaluations; VRL3 rule: goal total > 20 for pen, > 25 otherwise):

```bash
cd pointcloud
python -m sfg_pc.pretrain --task pen --seed 0 --demo $SFG_PC_OUT/data/adroit_pen_expert.zarr --save_dir $SFG_PC_OUT/bc_pen_s0
python -m sfg_pc.select_theta0 --task pen --bc_dir $SFG_PC_OUT/bc_pen_s0     # -> bc_pen_s0/theta0.pt
```

Our runs used steps 6,000 (door), 8,000 (pen) and 12,000 (relocate). Bucket uses the Figure 6 θ₀.

Online runs (one process each, about 13.5 GB of GPU memory; 2,300 episodes on Adroit, 2,600 on Bucket):

```bash
bash run_oneshot.sh door     --gpu=0                     # δ = 0.01
bash run_oneshot.sh door     --gpu=1 --func_delta=0.05
bash run_oneshot.sh pen      --gpu=2
bash run_oneshot.sh relocate --gpu=3
bash run_oneshot.sh bucket   --gpu=0
```

θ₀ is evaluated once on the 100 holdout seeds (`--prior_seed_base 900000`), and the holdouts at episodes
300, 800, 1300, 1800 (Bucket also 2300) and the final one use the same seeds, so each holdout is paired
with θ₀ episode by episode (McNemar). On Adroit the evaluation is reproducible per episode; on DexArt
the pairing fixes the object instance and the seeds, but the point-cloud noise is not seed-governed.
Results are in `curve.json` (`holdout_sr`, `final_holdout_sr`, `prior`) and per episode in
`eval_episodes.jsonl` (kinds `prior`, `holdout`, `final`); `sfg_steps.jsonl` logs the per-episode
functional step and path length.
