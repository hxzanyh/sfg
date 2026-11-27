import os
import tempfile
from datetime import datetime

import numpy as np
import wandb
from PIL import Image, ImageEnhance


# Set by main.py when the run resumed from a rolling checkpoint, so CsvLogger appends to the existing CSVs
# (keeping the rows logged before the checkpoint) instead of truncating them.
RESUMED = False
RESUMED_STEP = None   # log step of that checkpoint: rows after it were logged by the interrupted run


def _continue_csv(path, first_step):
    """Resume: keep the rows logged before `first_step` (rows at/after it were logged after the checkpoint the run
    resumed from, and are about to be logged again), atomically rewrite, return the existing header (or None)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        lines = f.read().splitlines()
    if not lines or lines[0].split(',')[-1] != 'step':
        return None
    kept = [lines[0]]
    for line in lines[1:]:
        try:
            # 'step' is the last column; values may contain commas (kitchen task lists), so split from the right
            if float(line.rsplit(',', 1)[-1]) < first_step:
                kept.append(line)
        except ValueError:
            continue
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write('\n'.join(kept) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"[CsvLogger] resumed: kept {len(kept) - 1}/{len(lines) - 1} rows of {os.path.basename(path)} (step < {first_step})")
    return lines[0].split(',')


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            _first = step if RESUMED_STEP is None else min(step, int(RESUMED_STEP) + 1)
            _prev = _continue_csv(self.path, _first) if RESUMED else None
            if _prev is not None and self.header is None:
                self.header = _prev
                self.file = open(self.path, 'a')
            else:
                if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                    # a fresh (non-resumed) run in an existing dir: set the old file aside instead of erasing it
                    os.replace(self.path, self.path + datetime.now().strftime('.bak_%Y%m%d_%H%M%S'))
                self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict(config=None):
    """Return the config as a plain dict for logging (empty dict if None)."""
    if config is None:
        return {}

    def to_dict(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        else:
            return obj

    return to_dict(config)


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    config=None,
    id=None,
    resume=None,
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(config),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    if id is not None:
        init_kwargs['id'] = id
    if resume is not None:
        init_kwargs['resume'] = resume

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Reshape a list of (t, h, w, c) videos into a single tiled wandb.Video."""
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Dim the padded frames so they are visually distinct.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
