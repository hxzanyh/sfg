"""PyTorch port of Octo's ConditionalUnet1D.

Continuous time input (B, 1), conditioning = concat(obs, time embedding), FiLM (scale, bias) in every
residual block, actions in (B, T, D) layout.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mish(x):
    return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x * emb  # (B, 1) * (D/2,) -> (B, D/2)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv1d(features, features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class Upsample1d(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv = nn.ConvTranspose1d(features, features, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish on (B, T, D)."""
    def __init__(self, in_features, out_features, kernel_size, n_groups):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_features, out_features, kernel_size,
                              stride=1, padding=padding)
        self.norm = nn.GroupNorm(n_groups, out_features)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = mish(x)
        return x.permute(0, 2, 1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_features, out_features, cond_dim, kernel_size=3,
                 n_groups=8, residual_proj=False):
        super().__init__()
        self.conv1 = Conv1dBlock(in_features, out_features, kernel_size, n_groups)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_features)
        self.conv2 = Conv1dBlock(out_features, out_features, kernel_size, n_groups)
        self.residual_proj = None
        if residual_proj or in_features != out_features:
            self.residual_proj = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x, cond):
        """x: (B, T, D), cond: (B, cond_dim)."""
        residual = x
        x = self.conv1(x)

        # FiLM conditioning
        cond_out = self.cond_proj(mish(cond))  # (B, 2*D)
        scale, bias = cond_out.chunk(2, dim=-1)
        x = x * scale.unsqueeze(1) + bias.unsqueeze(1)

        x = self.conv2(x)

        if self.residual_proj is not None:
            residual = self.residual_proj(
                residual.permute(0, 2, 1)).permute(0, 2, 1)

        return x + residual


class OctoConditionalUnet1D(nn.Module):
    """Octo's ConditionalUnet1D.

    Args:
        obs_dim: conditioning dimension
        input_dim: per-step action dimension
        down_features: channel sizes of the down/up path
        kernel_size, n_groups: conv parameters
        time_features: time embedding dimension
    """
    def __init__(self, obs_dim, input_dim, down_features=(256, 512, 1024),
                 mid_layers=2, kernel_size=3, n_groups=8, time_features=256):
        super().__init__()
        self.time_features = time_features

        # time embedding: SinusoidalPosEmb -> 2-layer MLP
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_features),
            nn.Linear(time_features, 4 * time_features),
        )
        self.time_mlp2 = nn.Linear(4 * time_features, time_features)

        cond_dim = obs_dim + time_features

        # down path
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_feat = input_dim
        for i, feat in enumerate(down_features):
            self.down_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    in_feat, feat, cond_dim,
                    kernel_size, n_groups, residual_proj=True),
                ConditionalResidualBlock1D(
                    feat, feat, cond_dim, kernel_size, n_groups),
            ]))
            if i != len(down_features) - 1:
                self.downsamplers.append(Downsample1d(feat))
            else:
                self.downsamplers.append(None)
            in_feat = feat

        # mid
        self.mid_blocks = nn.ModuleList([
            ConditionalResidualBlock1D(
                down_features[-1], down_features[-1], cond_dim,
                kernel_size, n_groups)
            for _ in range(mid_layers)
        ])

        # up path with skip connections
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        prev_feat = down_features[-1]
        for feat, skip_feat in zip(
                reversed(down_features[:-1]),
                reversed(down_features[1:])):
            self.up_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    prev_feat + skip_feat, feat, cond_dim,
                    kernel_size, n_groups, residual_proj=True),
                ConditionalResidualBlock1D(
                    feat, feat, cond_dim, kernel_size, n_groups),
            ]))
            self.upsamplers.append(Upsample1d(feat))
            prev_feat = feat

        self.final_conv = Conv1dBlock(
            down_features[0], down_features[0], kernel_size, n_groups)

    def forward(self, obs, action, time):
        """
        Args:
            obs: (B, obs_dim) conditioning
            action: (B, T, per_step_dim)
            time: (B, 1) continuous time in [0, 1]
        Returns:
            (B, T, down_features[0]); the caller projects to per_step_dim
        """
        time_emb = self.time_emb(time)
        time_emb = mish(time_emb)
        time_emb = self.time_mlp2(time_emb)  # (B, time_features)

        cond = torch.cat([obs, time_emb], dim=-1)

        x = action
        hidden_reps = []
        for i, (blocks, ds) in enumerate(zip(self.down_blocks, self.downsamplers)):
            res1, res2 = blocks
            x = res1(x, cond)
            x = res2(x, cond)
            if i != 0:
                hidden_reps.append(x)
            if ds is not None:
                x = ds(x)

        for mid in self.mid_blocks:
            x = mid(x, cond)

        for (res1, res2), us, h in zip(
                self.up_blocks, self.upsamplers,
                reversed(hidden_reps)):
            x = torch.cat([x, h], dim=-1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = us(x)

        x = self.final_conv(x)
        return x
