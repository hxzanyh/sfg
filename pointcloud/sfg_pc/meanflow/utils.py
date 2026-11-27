"""MeanFlow training utilities: (t, r) sampling, latent noise, adaptive L2 loss."""
import math
import torch


def sample_t_r(batch_size, flow_ratio=0.25, device='cuda'):
    """t, r ~ sigmoid(N(0,1) - 0.4), ordered so that t >= r; for the first (1 - flow_ratio) of the
    batch r = t (instantaneous-velocity samples)."""
    t = torch.sigmoid(torch.randn(batch_size, 1, device=device) - 0.4)
    r = torch.sigmoid(torch.randn(batch_size, 1, device=device) - 0.4)

    t_new = torch.maximum(t, r)
    r_new = torch.minimum(t, r)
    t, r = t_new, r_new

    data_size = int(batch_size * (1 - flow_ratio))
    mask = torch.arange(batch_size, device=device).unsqueeze(1) < data_size
    r = torch.where(mask, t, r)

    return t, r


def sample_sphere(batch_size, dim, device='cuda'):
    """Uniform on the sphere of radius sqrt(dim)."""
    e = torch.randn(batch_size, dim, device=device)
    norm = torch.linalg.norm(e, dim=-1, keepdim=True).clamp(min=1e-6)
    return e / norm * math.sqrt(dim)


def sample_latent_dist(batch_size, dim, latent_dist='normal', device='cuda'):
    if latent_dist == 'normal':
        return torch.randn(batch_size, dim, device=device)
    if latent_dist == 'sphere':
        return sample_sphere(batch_size, dim, device=device)
    raise ValueError(f'unknown latent_dist {latent_dist!r}')


def adaptive_l2_loss(error, p=0.5, c=1e-3):
    """Per-sample squared error reweighted by sg(1 / (err^2 + c)^p)."""
    squared_error = (error ** 2).mean(dim=-1)  # (B,)
    w = 1.0 / (squared_error.detach() + c) ** p
    return (w * squared_error).mean()
