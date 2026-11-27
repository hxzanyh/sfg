"""MeanFlow behaviour-cloning objective (Geng et al., 2025) for a one-step flow generator.

The network u_theta(z_t, t, u | s) is called with the interval width u = t - r in the actor's `dt`
slot, so the OGPO actor signature (obs, actions, times, dt) is reused unchanged. One-step decoding
evaluates the network at t = 1, u = 1 (see `sfg_helper.decode_meanflow`).

Paper, Appendix E: r = t on a fraction (1 - flow_ratio) of every batch (plain flow matching there),
classifier-free guidance folded into the target with weight `cfg_scale`, and an adaptive loss weight
1 / (||err||^2 + c)^p with p = 0.5, c = 1e-3. Nothing in this file runs unless
`--agent.use_meanflow=true`, so every OGPO configuration is unchanged.
"""
import jax
import jax.numpy as jnp


def adaptive_l2_loss(error, p=0.5, c=1e-3):
    """Mean of w * ||err||^2 with w = 1 / (||err||^2 + c)^p computed without gradient."""
    squared_error = jnp.mean(jnp.square(error), axis=-1)
    w = jax.lax.stop_gradient(1.0 / (squared_error + c) ** p)
    return (w * squared_error).mean()


def sample_t_r(batch_size, rng, flow_ratio=0.25):
    """t, r ~ sigmoid(N(-0.4, 1)) with t >= r; r is set to t on the first (1 - flow_ratio) rows."""
    rng, t_rng, r_rng = jax.random.split(rng, 3)
    t = jax.nn.sigmoid(jax.random.normal(t_rng, [batch_size, 1]) - 0.4)
    r = jax.nn.sigmoid(jax.random.normal(r_rng, [batch_size, 1]) - 0.4)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)
    data_size = int(batch_size * (1 - flow_ratio))
    zero_mask = (jnp.arange(batch_size) < data_size).reshape(batch_size, 1)
    r = jnp.where(zero_mask, t, r)
    return t, r


def sample_latent_dist(x_rng, sample_shape, latent_dist='normal'):
    """Base latent of the generator. Only N(0, I) is used by SFG."""
    if latent_dist == 'normal':
        return jax.random.normal(x_rng, sample_shape)
    raise NotImplementedError(f'latent_dist={latent_dist}')


def meanflow_bc_loss(actor_call, observations, actions, rng, flow_ratio=0.25,
                     latent_dist='normal', cfg_scale=1.0, valid_mask=None):
    """MeanFlow BC loss.

    actor_call(obs, z, t, u) -> predicted clean action, with u = t - r.
    valid_mask (B, horizon * action_dim) zeroes chunk entries that run past an episode boundary.
    Returns (loss, info).
    """
    batch_size, action_dim = actions.shape
    rng, x_rng, t_rng = jax.random.split(rng, 3)

    x = actions
    t, r = sample_t_r(batch_size, t_rng, flow_ratio)
    e = sample_latent_dist(x_rng, (batch_size, action_dim), latent_dist)

    z = (1 - t) * x + t * e
    v = e - x

    def mean_flow_forward(z_, t_, r_):
        return actor_call(observations, z_, t_, t_ - r_)

    if cfg_scale > 1.0:
        x_pred_t = jax.lax.stop_gradient(mean_flow_forward(z, t, t))
        u_t = z - x_pred_t
        v_guided = cfg_scale * v + (1 - cfg_scale) * u_t
    else:
        v_guided = v

    # Exact total derivative along (v_guided, 1, 0), i.e. d/dt u(z_t, r, t) with r held fixed.
    x_pred, dxdt = jax.jvp(
        mean_flow_forward, (z, t, r),
        (v_guided, jnp.ones_like(t), jnp.zeros_like(r)))
    u, dudt = z - x_pred, v_guided - dxdt
    u_tgt = jax.lax.stop_gradient(v_guided - jnp.clip(t - r, 0.0, 1.0) * dudt)

    err = u - u_tgt
    if valid_mask is not None:
        loss = adaptive_l2_loss(err * valid_mask)
    else:
        loss = adaptive_l2_loss(err)
    return loss, {
        'meanflow_loss': loss,
        'meanflow_err_sq': jnp.mean(jnp.square(err)),
        'meanflow_frac_r_eq_t': jnp.mean((t - r) < 1e-8),
        'meanflow_t_mean': jnp.mean(t),
    }
