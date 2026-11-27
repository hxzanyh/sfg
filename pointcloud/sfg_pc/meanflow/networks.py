"""One-step MeanFlow generator (1-D UNet) and the point-cloud + proprioception encoder."""
import torch
import torch.nn as nn

from sfg_pc.meanflow.octo_unet1d import OctoConditionalUnet1D


class MeanFlowUNet(nn.Module):
    """MeanFlow field u(obs, z, t, u) on an Octo 1-D UNet.

    The scalar u (= t - r) is appended to the observation feature, (obs, z, t) go through the UNet,
    and a 1x1 conv projects the output to the per-step action dimension.
    """
    def __init__(self, action_dim, horizon,
                 down_dims=(512, 1024, 2048), kernel_size=5, n_groups=8,
                 time_features=128, obs_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.obs_dim = obs_dim

        self.unet = OctoConditionalUnet1D(
            obs_dim=obs_dim + 1,  # +1 for the appended u scalar
            input_dim=action_dim,
            down_features=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            time_features=time_features,
        )

        # final 1x1 conv: down_dims[0] -> action_dim
        self.output_proj = nn.Conv1d(down_dims[0], action_dim, kernel_size=1)

    def forward(self, obs_enc, z, t, u):
        """
        Args:
            obs_enc: (B, obs_dim) encoded observation
            z: (B, horizon * action_dim) latent / noisy action
            t: (B, 1) or (B,) time
            u: (B, 1) or (B,) = t - r
        Returns:
            (B, horizon * action_dim)
        """
        B = z.shape[0]

        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if u.ndim == 1:
            u = u.unsqueeze(-1)

        obs_with_u = torch.cat([obs_enc, u], dim=-1)  # (B, obs_dim + 1)
        action = z.reshape(B, self.horizon, self.action_dim)
        out = self.unet(obs_with_u, action, t)  # (B, T, down_dims[0])
        out = self.output_proj(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out.reshape(B, -1)


class DP3PointNetEncoder(nn.Module):
    """DP3-style PointNet (per-point 1x1 convs + max-pool) plus a state MLP, per observation frame;
    the per-frame features of the n_obs_steps frames are concatenated."""
    def __init__(self, pn_out_channels=64, pn_block_channels=(64, 128, 256),
                 state_dim=24, state_hidden_dims=(64,), state_out_dim=64,
                 n_obs_steps=2, use_color=False):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        in_channels = 6 if use_color else 3

        pn_layers = []
        in_ch = in_channels
        for out_ch in pn_block_channels:
            pn_layers.append(nn.Conv1d(in_ch, out_ch, 1))
            pn_layers.append(nn.GroupNorm(1, out_ch))
            pn_layers.append(nn.ReLU())
            in_ch = out_ch
        self.pointnet = nn.Sequential(*pn_layers)
        self.pn_proj = nn.Linear(pn_block_channels[-1], pn_out_channels)

        state_layers = []
        in_dim = state_dim
        for h_dim in state_hidden_dims:
            state_layers.append(nn.Linear(in_dim, h_dim))
            state_layers.append(nn.ReLU())
            in_dim = h_dim
        state_layers.append(nn.Linear(in_dim, state_out_dim))
        self.state_mlp = nn.Sequential(*state_layers)

        self.per_step_dim = pn_out_channels + state_out_dim
        self.obs_dim = self.per_step_dim * n_obs_steps

    def _encode_single_step(self, point_cloud, state):
        pc = point_cloud.permute(0, 2, 1)
        feat = self.pointnet(pc)
        feat = feat.max(dim=-1).values
        pn_out = self.pn_proj(feat)
        state_out = self.state_mlp(state)
        return torch.cat([pn_out, state_out], dim=-1)

    def forward(self, observations):
        pc = observations['point_cloud']
        state = observations['agent_pos']

        if pc.ndim == 4:
            encodings = []
            for t_idx in range(pc.shape[1]):
                enc = self._encode_single_step(pc[:, t_idx], state[:, t_idx])
                encodings.append(enc)
            return torch.cat(encodings, dim=-1)
        else:
            return self._encode_single_step(pc, state)
