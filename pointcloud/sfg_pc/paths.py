"""External paths, read from environment variables only.

  DEXART_ROOT    dexart-release checkout (optional if `dexart` is pip-installed; required by gen_demos)
  DP3_ROOT       3D-Diffusion-Policy checkout; DEXART_ROOT defaults to $DP3_ROOT/third_party/dexart-release
  SFG_PC_DEMO    demonstration zarr (default of --demo)
  SFG_PC_THETA0  pretrained MeanFlow checkpoint theta_0 (default of --ckpt)
  SFG_PC_OUT     output root (default parent of --out)
"""
import os
import sys

ENV_HELP = {
    'DEXART_ROOT': 'the dexart-release checkout (contains dexart/ and examples/)',
    'DP3_ROOT': 'the 3D-Diffusion-Policy checkout (its third_party/dexart-release is used)',
    'SFG_PC_DEMO': 'the demonstration zarr, e.g. dexart_bucket_succ.zarr',
    'SFG_PC_THETA0': 'the pretrained MeanFlow checkpoint (theta_0)',
    'SFG_PC_OUT': 'the output root directory',
}


def env_path(name, must_exist=True):
    """Value of $name, or None if unset. Raises if set but missing on disk."""
    v = os.environ.get(name)
    if v and must_exist and not os.path.exists(v):
        raise SystemExit(f'${name}={v} does not exist ({ENV_HELP.get(name, "")})')
    return v or None


def require(value, flag, name):
    """`value` from a CLI flag (whose default is $name); fail with an actionable message if absent."""
    if not value:
        raise SystemExit(f'{flag} not given and ${name} is not set: {ENV_HELP.get(name, name)}')
    if not os.path.exists(value):
        raise SystemExit(f'{flag} {value} does not exist ({ENV_HELP.get(name, name)})')
    return value


def dexart_root(required=False):
    root = env_path('DEXART_ROOT')
    if root is None and env_path('DP3_ROOT'):
        root = os.path.join(os.environ['DP3_ROOT'], 'third_party', 'dexart-release')
        if not os.path.isdir(root):
            raise SystemExit(f'$DP3_ROOT is set but {root} does not exist; set $DEXART_ROOT')
    if root is None and required:
        raise SystemExit('set $DEXART_ROOT (dexart-release checkout) or $DP3_ROOT (3D-Diffusion-Policy '
                         'checkout with third_party/dexart-release)')
    return root


def setup_dexart(with_examples=False):
    """Make `import dexart` work: prepend DEXART_ROOT (and its examples/) to sys.path if given."""
    root = dexart_root(required=with_examples)
    if root is not None:
        paths = [root] + ([os.path.join(root, 'examples')] if with_examples else [])
        for p in paths:
            if p not in sys.path:
                sys.path.insert(0, p)
    try:
        import dexart  # noqa: F401
    except ImportError as e:
        raise SystemExit(f'cannot import dexart ({e}); install dexart-release (pip install -e) or set '
                         f'$DEXART_ROOT / $DP3_ROOT') from e
    return root


def setup_adroit():
    """Make `import adroit` (VRL3's AdroitEnv) and `import diffusion_policy_3d` (DP3's point-cloud
    wrapper) work from $DP3_ROOT; mj_envs / mjrl / mujoco-py must be installed (DP3's INSTALL.md)."""
    root = env_path('DP3_ROOT')
    if root is None:
        raise SystemExit('set $DP3_ROOT to the 3D-Diffusion-Policy checkout (patched with '
                         'patches/dp3_relocate_pointcloud.patch)')
    for p in (os.path.join(root, 'third_party', 'VRL3', 'src'), os.path.join(root, '3D-Diffusion-Policy')):
        if not os.path.isdir(p):
            raise SystemExit(f'{p} does not exist; $DP3_ROOT must be the 3D-Diffusion-Policy checkout')
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault('MUJOCO_GL', 'egl')


def setup_vulkan():
    """SAPIEN needs a Vulkan ICD; use the NVIDIA one if VK_ICD_FILENAMES is unset."""
    if 'VK_ICD_FILENAMES' not in os.environ:
        for c in ('/usr/share/vulkan/icd.d/nvidia_icd.json', '/etc/vulkan/icd.d/nvidia_icd.json'):
            if os.path.exists(c):
                os.environ['VK_ICD_FILENAMES'] = c
                break


def default_out(name):
    root = env_path('SFG_PC_OUT', must_exist=False)
    return os.path.join(root, name) if root else None
