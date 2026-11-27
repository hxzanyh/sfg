"""One-step MeanFlow policy: point-cloud encoder + 1-D UNet generator, with the MeanFlow BC loss."""
import torch

from sfg_pc.meanflow.utils import sample_t_r, sample_latent_dist, adaptive_l2_loss
from sfg_pc.meanflow.networks import MeanFlowUNet, DP3PointNetEncoder


class MeanFlowAgent:
    """Holds `obs_encoder` (observations -> feature) and `unet` (the generator f(feature, z, t, u)).

    Construction order (encoder, then UNet) fixes the parameter-initialisation draws from the torch
    CPU generator; callers that seed before constructing rely on it.
    """

    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.action_dim = cfg['action_dim']
        self.horizon = cfg['horizon']
        self.noise_dim = self.action_dim * self.horizon
        self.latent_dist = cfg.get('latent_dist', 'normal')
        self.flow_ratio = cfg.get('flow_ratio', 0.25)
        self.cfg_scale = cfg.get('cfg_scale', 1.0)
        assert cfg.get('mf_method', 'jit_mf') == 'jit_mf', cfg.get('mf_method')
        assert not cfg.get('ema_decay', 0.0), 'EMA weights are not supported'

        self.obs_encoder = DP3PointNetEncoder(
            pn_out_channels=cfg.get('pn_out_channels', 64),
            pn_block_channels=cfg.get('pn_block_channels', (64, 128, 256)),
            state_dim=cfg.get('state_dim', 24),
            state_hidden_dims=cfg.get('state_hidden_dims', (64,)),
            state_out_dim=cfg.get('state_out_dim', 64),
            n_obs_steps=cfg.get('n_obs_steps', 2),
        ).to(device)

        obs_dim = self.obs_encoder.obs_dim

        self.unet = MeanFlowUNet(
            action_dim=self.action_dim,
            horizon=self.horizon,
            down_dims=cfg.get('unet_down_features', (512, 1024, 2048)),
            kernel_size=cfg.get('unet_kernel_size', 5),
            n_groups=cfg.get('unet_n_groups', 8),
            time_features=cfg.get('unet_time_features', 128),
            obs_dim=obs_dim,
        ).to(device)

        self.optimizer = None
        self.step_count = 0

    def parameters(self):
        return list(self.unet.parameters()) + list(self.obs_encoder.parameters())

    def compute_jit_mf_loss(self, obs_enc, actions_flat):
        """MeanFlow loss with a finite-difference d/dt.

          z     = (1 - t) x + t e
          u     = z - f(obs, z, t, t - r)
          u_tgt = v_g - clip(t - r, 0, 1) * du/dt,   v_g = w v + (1 - w) (z - f(obs, z, t, 0)), v = e - x
          loss  = adaptive_l2(u - sg(u_tgt))
        """
        B = actions_flat.shape[0]
        x = actions_flat
        e = sample_latent_dist(B, self.noise_dim, self.latent_dist, self.device)
        t, r = sample_t_r(B, self.flow_ratio, self.device)

        z = (1 - t) * x + t * e
        v = e - x

        # guided velocity (training only)
        if self.cfg_scale > 1.0:
            with torch.no_grad():
                x_pred_t = self.unet(obs_enc, z, t, torch.zeros_like(t))
                u_t = z - x_pred_t
            v_guided = self.cfg_scale * v + (1 - self.cfg_scale) * u_t
        else:
            v_guided = v

        x_pred = self.unet(obs_enc, z, t, t - r)
        u = z - x_pred

        # du/dt ~ (u(z + eps v_g, t + eps) - u(z, t)) / eps
        eps = self.cfg.get('jvp_eps', 1e-3)
        with torch.no_grad():
            z_eps = z + eps * v_guided
            t_eps = (t + eps).clamp(max=1.0)
            x_pred_eps = self.unet(obs_enc, z_eps, t_eps, t_eps - r)
            u_eps = z_eps - x_pred_eps
            dudt = (u_eps - u.detach()) / eps

        u_tgt = (v_guided - (t - r).clamp(0, 1) * dudt).detach()

        err = u - u_tgt
        return adaptive_l2_loss(err)

    @torch.no_grad()
    def decode(self, noise, obs_enc):
        """One-step decode at t = 1, u = 1, clipped to [-1, 1]."""
        self.unet.eval()
        B = noise.shape[0]
        t = torch.ones(B, 1, device=self.device)
        u = torch.ones(B, 1, device=self.device)
        pred = self.unet(obs_enc, noise, t, u)
        return pred.clamp(-1, 1)

    @torch.no_grad()
    def sample_actions(self, observations):
        """obs -> encode -> latent ~ latent_dist -> decode; returns (B, horizon, action_dim)."""
        self.unet.eval()
        self.obs_encoder.eval()
        obs_enc = self.obs_encoder(observations)
        B = obs_enc.shape[0]
        noise = sample_latent_dist(B, self.noise_dim, self.latent_dist, self.device)
        actions_flat = self.decode(noise, obs_enc)
        return actions_flat.reshape(B, self.horizon, self.action_dim)

    def save(self, path, normalizer=None):
        torch.save({
            'unet': self.unet.state_dict(),
            'obs_encoder': self.obs_encoder.state_dict(),
            'ema_unet': None,
            'ema_encoder': None,
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'step': self.step_count,
            'cfg': self.cfg,
            'normalizer': normalizer.stats if normalizer else None,
        }, path)

    def load_weights(self, ckpt):
        """Load generator and encoder weights from a checkpoint dict written by `save`."""
        assert ckpt.get('ema_unet') is None and ckpt.get('ema_encoder') is None, \
            'checkpoint carries EMA weights, which are not supported'
        self.unet.load_state_dict(ckpt['unet'])
        self.obs_encoder.load_state_dict(ckpt['obs_encoder'])
        self.step_count = ckpt.get('step', 0)
