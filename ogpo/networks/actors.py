"""Actor (policy) network architectures: Gaussian, flow-matching, transformer, and MeanFlow policies."""

from typing import Any, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules import (
    MLP,
    MLPCond,
    MLPWithFiLM,
    SimBaMLP,
    TanhMultivariateNormalDiag,
    AdaLNLayer,
    AdaLNFinalLayer,
    CrossAttnLayer,
    CrossAttnFinalLayer,
    SinusoidalTimeEmbedding,
    TwoTierObsEncoder,
)


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class Actor(nn.Module):
    """Standard Gaussian policy with MLP backbone."""
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = True
    const_std: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None
    # Two-tier obs encoder (frozen-encoder image runs): see TwoTierObsEncoder.
    obs_two_tier: bool = False
    two_tier_img_dim: int = 0
    two_tier_proprio_dim: int = 0
    two_tier_fused_dim: int = 0  # if 0, defaults to hidden_dims[0]

    def setup(self):
        if self.obs_two_tier:
            fused = self.two_tier_fused_dim if self.two_tier_fused_dim > 0 else self.hidden_dims[0]
            self.two_tier_encoder = TwoTierObsEncoder(
                img_feat_dim=self.two_tier_img_dim,
                proprio_dim=self.two_tier_proprio_dim,
                fused_dim=fused,
            )
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0,):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        if self.obs_two_tier:
            inputs = self.two_tier_encoder(inputs)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TanhMultivariateNormalDiag(distribution, low=self.low, high=self.high)

        return distribution


class ActorVectorField(nn.Module):
    """Flow-matching policy (velocity field) with optional FiLM conditioning."""
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_film: bool = False
    use_denoiser: bool = False
    time_embedding: Optional[nn.Module] = None
    # Two-tier obs encoder (frozen-encoder image runs): L2-normalize the
    # image-feature slice and project image / proprio through separate
    # Dense -> LN -> GELU towers, concat into a fused representation.
    obs_two_tier: bool = False
    two_tier_img_dim: int = 0
    two_tier_proprio_dim: int = 0
    two_tier_fused_dim: int = 0  # if 0, defaults to hidden_dims[0]

    def setup(self) -> None:
        if self.obs_two_tier:
            fused = self.two_tier_fused_dim if self.two_tier_fused_dim > 0 else self.hidden_dims[0]
            self.two_tier_encoder = TwoTierObsEncoder(
                img_feat_dim=self.two_tier_img_dim,
                proprio_dim=self.two_tier_proprio_dim,
                fused_dim=fused,
            )

        if self.use_film:
            self.mlp = MLPWithFiLM(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
        else:
            self.mlp = MLP(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )

        if self.use_denoiser:
            if self.use_film:
                self.denoiser_mlp = MLPWithFiLM(
                    (*self.hidden_dims, self.action_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm
                )
            else:
                self.denoiser_mlp = MLP(
                    (*self.hidden_dims, self.action_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm
                )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False, return_denoiser=False, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Runs regardless of `is_encoded`: normalizes already-encoded image features.
        if self.obs_two_tier:
            observations = self.two_tier_encoder(observations)

        if times is not None and self.time_embedding is not None:
            times = self.time_embedding(times)

        if self.use_film:
            cond = observations
            if times is not None:
                cond = jnp.concatenate([cond, times], axis=-1)
            if dt is not None:
                cond = jnp.concatenate([cond, dt], axis=-1)
            v = self.mlp(actions, cond)
        else:
            if times is None:
                inputs = jnp.concatenate([observations, actions], axis=-1)
            else:
                inputs = jnp.concatenate([observations, actions, times], axis=-1)
            if dt is not None:
                inputs = jnp.concatenate([inputs, dt], axis=-1)
            v = self.mlp(inputs)

        if return_denoiser:
            if not self.use_denoiser:
                raise ValueError("Actor was initialized with use_denoiser=False")

            if self.use_film:
                z, _ = self.denoiser_mlp(actions, cond)  # MLPWithFiLM returns tuple
            else:
                z = self.denoiser_mlp(inputs)
            return v, z

        return v


class ActorVectorFieldTF(nn.Module):
    """Transformer flow-matching policy with AdaLN or cross-attention conditioning.

    Bidirectional self-attention over action tokens, no positional embeddings.
    """
    hidden_dim: int
    action_dim: int
    action_chunk_size: int = 4  # T
    layer_norm: bool = False
    encoder: nn.Module = None
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.0
    use_denoiser: bool = False
    conditioning_type: str = 'adaln'  # 'adaln' or 'cross_attn'
    time_embedding: Optional[nn.Module] = None

    def setup(self) -> None:
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.conditioning_type in ('adaln', 'cross_attn'), \
            f"conditioning_type must be 'adaln' or 'cross_attn', got '{self.conditioning_type}'"

        self.action_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.cond_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())

        if self.conditioning_type == 'adaln':
            self.transformer_layers = [
                AdaLNLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                           self.dropout_rate)
                for _ in range(self.num_layers)
            ]
            self.final_layer = AdaLNFinalLayer(self.hidden_dim, self.action_dim)
            if self.use_denoiser:
                self.denoiser_final_layer = AdaLNFinalLayer(self.hidden_dim, self.action_dim)
        else:  # cross_attn
            self.transformer_layers = [
                CrossAttnLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                               self.dropout_rate)
                for _ in range(self.num_layers)
            ]
            self.final_layer = CrossAttnFinalLayer(self.hidden_dim, self.action_dim)
            if self.use_denoiser:
                self.denoiser_final_layer = CrossAttnFinalLayer(self.hidden_dim, self.action_dim)

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False,
                 return_denoiser=False, deterministic=True, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Reshape flat actions -> (B, T, a_dim); handle unbatched (2D) input
        if actions.ndim == 2:
            actions = actions.reshape(-1, self.action_chunk_size, self.action_dim)
        else:
            actions = actions.reshape(self.action_chunk_size, self.action_dim)

        x = self.action_proj(actions)  # (B, T, hidden_dim)
        cond = self.cond_proj(observations)  # (B, hidden_dim)

        if times is not None:
            t_feat = self.time_embedding(times) if self.time_embedding is not None else times
            time_emb = nn.Dense(self.hidden_dim, kernel_init=default_init())(t_feat)
            cond = cond + time_emb

        if dt is not None:
            dt_emb = nn.Dense(self.hidden_dim, kernel_init=default_init())(dt)
            cond = cond + dt_emb

        if self.conditioning_type == 'adaln':
            for layer in self.transformer_layers:
                x = layer(x, cond, deterministic=deterministic)
            v = self.final_layer(x, cond)  # (B, T, action_dim)
            if self.use_denoiser:
                z = self.denoiser_final_layer(x, cond)
        else:
            # cond holds state + time summed into (B, D); expand to a single context token.
            context = cond[:, None, :]  # (B, 1, D)
            for layer in self.transformer_layers:
                x = layer(x, context, deterministic=deterministic)
            v = self.final_layer(x)  # (B, T, action_dim)
            if self.use_denoiser:
                z = self.denoiser_final_layer(x)

        if self.use_denoiser:
            z = z.reshape(-1, self.action_chunk_size * self.action_dim)
            if return_denoiser:
                v = v.reshape(-1, self.action_chunk_size * self.action_dim)
                return v, z

        v = v.reshape(-1, self.action_chunk_size * self.action_dim)  # (B, T*a_dim)
        return v


class ActorVectorFieldSimBa(nn.Module):
    """Flow-matching policy using the SimBa architecture (residual blocks + running-stat norm)."""
    hidden_dims: Sequence[int]
    action_dim: int
    encoder: Optional[nn.Module] = None
    rs_norm_momentum: float = 0.99
    rs_norm_epsilon: float = 1e-8
    time_embedding: Optional[nn.Module] = None

    def setup(self):
        self.simba_mlp = SimBaMLP(
            hidden_dims=(*self.hidden_dims, self.action_dim),
            rs_norm_momentum=self.rs_norm_momentum,
            rs_norm_epsilon=self.rs_norm_epsilon
        )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        if times is not None and self.time_embedding is not None:
            times = self.time_embedding(times)

        inputs = [observations, actions]
        if times is not None:
            inputs.append(times)
        if dt is not None:
            inputs.append(dt)

        x = jnp.concatenate(inputs, axis=-1)
        return self.simba_mlp(x)


__all__ = [
    'Actor',
    'ActorVectorField',
    'ActorVectorFieldTF',
    'ActorVectorFieldSimBa',
]


# ---------------------------------------------------------------------------------------------------------
# MeanFlowMLPOcto: one-step MeanFlow decoder with its own state encoder (used by SFG's pretrained
# generators on tool hang and the kitchen tasks; built with --agent.meanflow_arch=octo_mlp).
#   * state encoder: Linear -> LayerNorm -> ReLU per hidden layer, then Linear to 128 (the `enc_*`
#     submodules, frozen during SFG fine-tuning by --agent.sfg_freeze_obs_encoder);
#   * t and u = t - r each pass through a 32-d sinusoidal embedding (frequencies 1 .. 1e-4) and one
#     shared Linear(128) -> SiLU -> Linear(32);
#   * trunk Linear -> LayerNorm -> SiLU, zero-initialised output layer;
#   * PyTorch nn.Linear default initialisation, LayerNorm eps 1e-5.
# ---------------------------------------------------------------------------------------------------------
import math as _math


def _torch_kernel_init():
    # torch.nn.Linear: kaiming_uniform_(a=sqrt(5)) == U(-1/sqrt(fan_in), 1/sqrt(fan_in)); flax's
    # variance_scaling(1/3, fan_in, uniform) has bound sqrt(3 * (1/3) / fan_in) -- the same law.
    return nn.initializers.variance_scaling(1.0 / 3.0, 'fan_in', 'uniform')


class _TorchDense(nn.Module):
    features: int
    zero_init: bool = False

    @nn.compact
    def __call__(self, x):
        if self.zero_init:
            return nn.Dense(self.features, kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.zeros)(x)
        bound = 1.0 / _math.sqrt(x.shape[-1])

        def _bias_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype, -bound, bound)

        return nn.Dense(self.features, kernel_init=_torch_kernel_init(), bias_init=_bias_init)(x)


def _octo_sinusoidal(x, dim):
    """Sinusoidal embedding x (..., 1) -> (..., dim), Octo/Diffusion-Policy convention."""
    half = dim // 2
    freq = jnp.exp(jnp.arange(half, dtype=x.dtype) * -(_math.log(10000) / (half - 1)))
    e = x * freq
    return jnp.concatenate([jnp.sin(e), jnp.cos(e)], axis=-1)


class MeanFlowMLPOcto(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int                               # flat chunk dim (horizon * per-step action dim)
    obs_hidden_dims: Sequence[int] = (256, 256)
    obs_out_dim: int = 128
    time_features: int = 32
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False,
                 return_denoiser=False, images=None):
        if return_denoiser:
            raise ValueError("MeanFlowMLPOcto has no denoiser head")
        # `is_encoded` refers to OGPO's external encoder, which state runs do not have; this state
        # encoder is part of the actor and always runs.
        h = observations
        for i, w in enumerate(self.obs_hidden_dims):
            h = _TorchDense(w, name=f'enc_dense_{i}')(h)
            h = nn.LayerNorm(epsilon=1e-5, name=f'enc_ln_{i}')(h)
            h = nn.relu(h)
        obs_enc = _TorchDense(self.obs_out_dim, name='enc_out')(h)

        t_fc1 = _TorchDense(4 * self.time_features, name='t_mlp_0')
        t_fc2 = _TorchDense(self.time_features, name='t_mlp_2')

        def _temb(x):
            if x.ndim == observations.ndim - 1:
                x = x[..., None]
            return t_fc2(nn.swish(t_fc1(_octo_sinusoidal(x, self.time_features))))

        if times is None:
            times = jnp.ones((*observations.shape[:-1], 1), dtype=observations.dtype)
        u = dt if dt is not None else jnp.zeros_like(times)
        h = jnp.concatenate([obs_enc, actions, _temb(times), _temb(u)], axis=-1)
        for i, w in enumerate(self.hidden_dims):
            h = _TorchDense(w, name=f'dense_{i}')(h)
            if self.layer_norm:
                h = nn.LayerNorm(epsilon=1e-5, name=f'ln_{i}')(h)
            h = nn.swish(h)                                   # torch nn.SiLU
        return _TorchDense(self.action_dim, zero_init=True, name='out')(h)
