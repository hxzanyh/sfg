"""Simple Flow Gradient (SFG): direct Q-gradient fine-tuning of a pretrained flow policy with a
per-episode bound on the policy's functional step.

Pieces, in the order Algorithm 1 of the paper uses them:

  decode     f_theta(s, z): one-step MeanFlow decode (`decode_meanflow`), or a K-step Euler decode of
             a flow-matching velocity field (`decode_flow`, `decode_flow_endpoint`). The output is
             clipped to the action box once; execution, the TD target, the actor loss and the probe
             bank all use the same clipped action (`sfg_decode` dispatches from the config).
  actor      loss = -mean( q / sg(mean|q|) ),  q = min over M random heads of Q(s, f_theta(s, z)),
             on the states of the critic's mixed batch (`sfg_actor_loss`).
  critic     y = r + gamma^H * mask * min over M random heads of Q_target(s', f_theta(s', z')),
             a' from the current actor with a fresh latent (see OGPOAgent.critic_loss).
  bound      at every episode boundary the functional step d(theta, theta~) on a fixed probe bank of
             demonstration states x latent draws is projected back to `sfg_func_delta` by blending
             theta + alpha * (theta~ - theta) (`project_functional_step`).

Ablations are kept behind flags that default to off:
`sfg_anchor_beta` (fixed BC anchor on demo rows, Section 3), `sfg_ref_pen` (penalty on the action
distance to the episode-start actor instead of the projection), `sfg_param_delta` (trust region on
the parameter step instead of the functional step), `sfg_act_ste` (straight-through clip).
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.modules.meanflow_helper import sample_latent_dist


# ---------------------------------------------------------------------------------------------------
# decoders
# ---------------------------------------------------------------------------------------------------

def decode_meanflow(actor_apply, observations, z, act_min, act_max, params=None, ste=False,
                    return_raw=False):
    """One-step MeanFlow decode at t = 1, u = t - r = 1, then clip.

    ste=True keeps the clipped forward action but passes the gradient through the clip unchanged
    (straight-through). Off by default: the paper uses the exact gradient of the clip.
    """
    one = jnp.ones((*observations.shape[:-1], 1))
    if params is None:
        a_raw = actor_apply(observations, z, one, one, is_encoded=True)
    else:
        a_raw = actor_apply(observations, z, one, one, params=params, is_encoded=True)
    a = jnp.clip(a_raw, act_min, act_max)
    if ste:
        a = a_raw + jax.lax.stop_gradient(a - a_raw)
    return (a, a_raw) if return_raw else a


def decode_flow(actor_apply, observations, z, act_min, act_max, params=None, flow_steps=10,
                grad_steps=None):
    """K-step Euler decode of a flow-matching velocity field v(s, a, t), t: 0 -> 1, then clip.

    Same forward recursion as `pg_helper.sample_flow_actions_ode` (no intermediate clipping), so SFG
    and the OGPO evaluator decode a checkpoint identically. `grad_steps = m` backpropagates through
    the last m steps only (Algorithm 2, truncated backpropagation); None or m = K is the full chain.
    The forward pass is the full K-step chain for every m.
    """
    K = int(flow_steps)
    M = K if grad_steps is None else max(1, min(int(grad_steps), K))
    dt = 1.0 / K
    a = z
    for i in range(K):
        t = jnp.full((*observations.shape[:-1], 1), i / K)
        if params is None:
            v = actor_apply(observations, a, t, is_encoded=True)
        else:
            v = actor_apply(observations, a, t, params=params, is_encoded=True)
        a = a + v * dt
        if i < K - M:
            a = jax.lax.stop_gradient(a)
    return jnp.clip(a, act_min, act_max)


def decode_flow_endpoint(actor_apply, observations, z, act_min, act_max, params=None, flow_steps=10,
                         endpoint_from=None):
    """K-step Euler forward pass with endpoint-prediction probes (Algorithm 3).

    From every intermediate state z_k (detached) at t_k, k >= k0, predict the endpoint in one Euler
    step of length (1 - t_k): a_k = clip(z_k + (1 - t_k) v(z_k, t_k)). Each probe is differentiable
    through a single network evaluation. Returns (a_exec, probes): a_exec is the detached K-step
    endpoint (what is executed), probes has shape (K - k0, B, act_dim). k0 defaults to K // 2; at
    k0 = K - 1 the single probe is the executed action and the rule equals `decode_flow` with m = 1.
    """
    K = int(flow_steps)
    k0 = (K // 2) if endpoint_from is None else max(0, min(int(endpoint_from), K - 1))
    dt = 1.0 / K
    sg = jax.lax.stop_gradient

    def _v(obs, zz, t):
        tt = jnp.full((*obs.shape[:-1], 1), t)
        if params is None:
            return actor_apply(obs, zz, tt, is_encoded=True)
        return actor_apply(obs, zz, tt, params=params, is_encoded=True)

    zz = z
    probes = []
    for i in range(K):
        t = i / K
        zd = sg(zz)
        v = _v(observations, zd, t)
        if i >= k0:
            probes.append(jnp.clip(zd + v * (1.0 - t), act_min, act_max))
        zz = zd + sg(v) * dt
    a_exec = jnp.clip(zz, act_min, act_max)
    return a_exec, jnp.stack(probes)


def uses_flow_decode(cfg):
    return bool(cfg.get('sfg_flow_decode', False))


def sfg_decode(cfg, actor_apply, observations, z, act_min, act_max, params=None, grad_steps=None):
    """The decode SFG executes, chosen from the config. Every decode site calls this."""
    if uses_flow_decode(cfg):
        return decode_flow(actor_apply, observations, z, act_min, act_max, params=params,
                           flow_steps=int(cfg['flow_steps']), grad_steps=grad_steps)
    return decode_meanflow(actor_apply, observations, z, act_min, act_max, params=params)


# ---------------------------------------------------------------------------------------------------
# actor loss
# ---------------------------------------------------------------------------------------------------

def min_of_random_heads(qs, rng, nsub, num_qs):
    """qs: (num_qs, B). Minimum over `nsub` heads drawn without replacement."""
    idx = jax.random.permutation(rng, num_qs)[:nsub]
    return qs[idx].min(axis=0)


def sfg_actor_loss(cfg, actor_apply, critic_apply, obs_encoded, critic_obs_encoded, grad_params, rng,
                   act_dim, act_min, act_max, critic_is_encoded, data_actions=None, n_demo_rows=0,
                   ref_params=None):
    """Direct Q-gradient actor loss, -mean(q / sg(mean|q|)).

    The loss value is identically -1; size any added coefficient against gradient norms, not against
    the loss value.

    data_actions / n_demo_rows: rows [0, n_demo_rows) of a mixed batch are demonstration rows. Only
    read when `sfg_anchor_beta > 0`.
    ref_params: actor parameters at the start of the current episode. Only read when
    `sfg_ref_pen > 0`.
    """
    actor_bs = int(cfg['sfg_actor_bs'])
    nsub = int(cfg['sfg_nsub'])
    num_qs = int(cfg['num_qs'])
    latent_dist = cfg['meanflow_latent_dist']
    flow = uses_flow_decode(cfg)
    endpoint = flow and str(cfg.get('sfg_grad_mode', 'bptt')) == 'endpoint'
    grad_steps = int(cfg.get('sfg_grad_steps', 0) or 0) or None

    B = obs_encoded.shape[0]
    rng, idx_rng, z_rng, head_rng = jax.random.split(rng, 4)
    n = min(actor_bs, B)
    idx = jax.random.choice(idx_rng, B, (n,), replace=False)
    obs = obs_encoded[idx]
    c_obs = critic_obs_encoded[idx]
    z = sample_latent_dist(z_rng, (n, act_dim), latent_dist)

    if endpoint:
        _k0 = int(cfg.get('sfg_endpoint_from', -1))
        a, probes = decode_flow_endpoint(
            actor_apply, obs, z, act_min, act_max, params=grad_params,
            flow_steps=int(cfg['flow_steps']), endpoint_from=(_k0 if _k0 >= 0 else None))
        a_raw = a
        P = probes.shape[0]
        qs = critic_apply(c_obs, actions=a, is_encoded=critic_is_encoded)
        c_tile = jnp.tile(c_obs, (P,) + (1,) * (c_obs.ndim - 1))
        q_probe = min_of_random_heads(
            critic_apply(c_tile, actions=probes.reshape((P * n, -1)), is_encoded=critic_is_encoded),
            head_rng, nsub, num_qs).reshape((P, n))
        # normalised per probe: the earliest probe's gradient is several times larger than the last
        qn = q_probe / (jax.lax.stop_gradient(jnp.abs(q_probe).mean(axis=1, keepdims=True)) + 1e-6)
        loss = -qn.mean()
        q = min_of_random_heads(qs, head_rng, nsub, num_qs)        # reported Q of the executed action
    else:
        if flow:
            a = decode_flow(actor_apply, obs, z, act_min, act_max, params=grad_params,
                            flow_steps=int(cfg['flow_steps']), grad_steps=grad_steps)
            a_raw = a
        else:
            a, a_raw = decode_meanflow(actor_apply, obs, z, act_min, act_max, params=grad_params,
                                       ste=bool(cfg.get('sfg_act_ste', False)), return_raw=True)
        qs = critic_apply(c_obs, actions=a, is_encoded=critic_is_encoded)          # (num_qs, n)
        q = min_of_random_heads(qs, head_rng, nsub, num_qs)
        qn = q / (jax.lax.stop_gradient(jnp.abs(q).mean()) + 1e-6)
        loss = -qn.mean()

    info = {}
    ref_pen = float(cfg.get('sfg_ref_pen', 0.0))
    if ref_params is not None and ref_pen > 0.0:
        # loss += lambda * mean_rows sum_dims (a - a_ref)^2, a_ref decoded by the episode-start actor on
        # the same rows and latents. sqrt of this term is in the units of the functional step.
        a_ref = jax.lax.stop_gradient(sfg_decode(cfg, actor_apply, obs, z, act_min, act_max,
                                                 params=ref_params))
        ref_term = jnp.mean(jnp.sum(jnp.square(a - a_ref), axis=-1))
        loss = loss + ref_pen * ref_term
        info['sfg_ref_dist_rms'] = jnp.sqrt(jax.lax.stop_gradient(ref_term))

    anchor_beta = float(cfg.get('sfg_anchor_beta', 0.0))
    if anchor_beta > 0.0 and data_actions is not None and n_demo_rows > 0:
        # loss += beta * mean over demo rows of mean_dims (a - a_demo)^2: a FIXED action-space anchor
        m = (idx < n_demo_rows).astype(jnp.float32)
        d = ((a - data_actions[idx]) ** 2).mean(axis=-1)
        anchor = (d * m).sum() / jnp.maximum(m.sum(), 1.0)
        loss = loss + anchor_beta * anchor
        info['sfg_anchor'] = jax.lax.stop_gradient(anchor)

    info.update({
        'sfg_loss': loss,
        'sfg_q_mean': q.mean(),
        'sfg_q_absmean': jnp.abs(q).mean(),
        'actions': a.mean(), 'actions_std': a.std(axis=0).mean(),
        'actions_min': a.min(), 'actions_max': a.max(),
        # fraction of clipped action coordinates (they receive no Q-gradient)
        'clip_frac': jnp.mean((jnp.abs(a) >= act_max - 1e-6).astype(jnp.float32)),
        'raw_absmax': jnp.max(jnp.abs(a_raw)),
    })
    return loss, info


# ---------------------------------------------------------------------------------------------------
# probe bank and the per-episode projection
# ---------------------------------------------------------------------------------------------------

def build_probe_bank(observations_pool, n_states, n_lats, act_dim, latent_dist, seed):
    """Fixed bank of n_states demonstration states x n_lats latent draws, drawn once."""
    rs = np.random.RandomState(seed)
    pool = np.asarray(observations_pool)
    si = rs.randint(0, len(pool), size=n_states)
    S = np.repeat(pool[si], n_lats, axis=0).astype(np.float32)               # (n_states*n_lats, obs)
    W = np.asarray(sample_latent_dist(jax.random.PRNGKey(seed), (len(S), act_dim), latent_dist),
                   dtype=np.float32)
    return jnp.asarray(S), jnp.asarray(W)


def rms_rows(d):
    """Functional distance: sqrt( mean over bank rows of sum over action dims of d^2 )."""
    return float(jnp.sqrt(jnp.mean(jnp.sum(jnp.square(d), axis=-1))))


def make_bank_decoder(agent, bank_S, bank_W):
    """f(full actor params) -> decoded actions on the bank (jitted once)."""
    act_min, act_max = agent.config.act_min, agent.config.act_max
    select = agent.actor_network.select('actor')
    cfg = agent.config

    @jax.jit
    def _f(full_params):
        return sfg_decode(cfg, select, bank_S, bank_W, act_min, act_max, params=full_params)

    return _f


def blend_actor(full_pre, full_now, alpha):
    """theta_pre + alpha * (theta_now - theta_pre) on modules_actor only (critic, targets untouched)."""
    out = dict(full_now)
    out['modules_actor'] = jax.tree_util.tree_map(
        lambda p, q: p + alpha * (q - p), full_pre['modules_actor'], full_now['modules_actor'])
    return out


def project_functional_step(bank_f, full_pre, full_now, f_pre, func_delta, max_iter=8, tol=0.05,
                            bisect=False, bisect_iter=24):
    """Find alpha in (0, 1] with rms(f(theta_alpha) - f_pre) = func_delta (Appendix F).

    Proportional iteration alpha <- alpha * delta / d(alpha) from alpha = delta / d(1), accepted once
    within +-tol of delta; after max_iter the best iterate is used. Nothing is changed when the
    proposal's step is already <= delta. Returns (params_to_commit, info).

    bisect=True (multi-step decoders, Appendix G): when the proportional iteration misses the
    tolerance, fall back to a bisection in log(alpha) on [1e-12, 1]. Through K decoder evaluations
    the step is far from linear in alpha and the solutions lie near alpha ~ 1e-7.
    """
    delta_raw = rms_rows(bank_f(full_now) - f_pre)
    info = {'delta_raw': delta_raw, 'alpha': 1.0, 'iters': 0, 'resid': 0.0,
            'projected': bool(delta_raw > func_delta)}
    if not (delta_raw > func_delta and delta_raw > 1e-12):
        return full_now, info
    alpha = min(func_delta / delta_raw, 1.0)
    best = (alpha, float('inf'))
    resid = float('inf')
    for k in range(1, max_iter + 1):
        cand = blend_actor(full_pre, full_now, alpha)
        cur = rms_rows(bank_f(cand) - f_pre)
        resid = abs(cur - func_delta) / func_delta
        if resid < best[1]:
            best = (alpha, resid)
        info['iters'] = k
        if resid <= tol:
            break
        alpha = min(alpha * func_delta / max(cur, 1e-12), 1.0)
    if resid > tol and bisect:
        lo, hi = 1e-12, 1.0
        for k in range(1, bisect_iter + 1):
            mid = math.sqrt(lo * hi)
            cur = rms_rows(bank_f(blend_actor(full_pre, full_now, mid)) - f_pre)
            r = abs(cur - func_delta) / func_delta
            if r < best[1]:
                best = (mid, r)
            info['bisect_iters'] = k
            if r <= tol:
                break
            if cur > func_delta:
                hi = mid
            else:
                lo = mid
    if resid > tol:
        alpha, resid = best
    info['alpha'] = float(alpha)
    info['resid'] = float(resid)
    return blend_actor(full_pre, full_now, alpha), info


def actor_param_dist(full_a, full_b):
    """||theta_a - theta_b||_2 over modules_actor, accumulated in float64 on the host."""
    la = jax.tree_util.tree_leaves(full_a['modules_actor'])
    lb = jax.tree_util.tree_leaves(full_b['modules_actor'])
    assert len(la) == len(lb), (len(la), len(lb))
    s = 0.0
    for x, y in zip(la, lb):
        d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
        s += float(np.sum(d * d))
    return math.sqrt(s)


def project_param_step(full_pre, full_now, param_delta):
    """Parameter-space trust region (ablation): alpha = min(1, eps / ||theta~ - theta||_2)."""
    n = actor_param_dist(full_now, full_pre)
    if not (n > param_delta and n > 1e-12):
        return full_now, {'alpha': 1.0, 'projected': False}
    alpha = float(param_delta / n)
    return blend_actor(full_pre, full_now, alpha), {'alpha': alpha, 'projected': True}
