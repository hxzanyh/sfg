"""Unit tests for the SFG decoders and the per-episode projection (no simulator or dataset needed)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.modules.sfg_helper import (actor_param_dist, blend_actor, decode_flow,
                                           decode_flow_endpoint, decode_meanflow, project_functional_step,
                                           project_param_step, rms_rows)

OBS, ACT, K = 5, 3, 6


def _velocity(params, obs, a, t, is_encoded=True):
    h = jnp.tanh(jnp.concatenate([obs, a, t], -1) @ params['modules_actor']['w1'])
    return h @ params['modules_actor']['w2']


def _params(key, scale=0.5):
    k1, k2 = jax.random.split(key)
    return {'modules_actor': {'w1': scale * jax.random.normal(k1, (OBS + ACT + 1, 16)),
                              'w2': scale * jax.random.normal(k2, (16, ACT))},
            'modules_critic': {'w': jnp.ones(2)}}


def _apply(obs, a, t, params=None, is_encoded=True):
    return _velocity(params, obs, a, t)


@pytest.fixture
def data():
    k = jax.random.PRNGKey(0)
    return _params(k), jax.random.normal(jax.random.PRNGKey(1), (4, OBS)), jax.random.normal(jax.random.PRNGKey(2), (4, ACT))


def test_truncated_backprop_keeps_the_forward_pass(data):
    p, obs, z = data
    outs = [decode_flow(_apply, obs, z, -1.0, 1.0, params=p, flow_steps=K, grad_steps=m) for m in (1, 3, K, None)]
    for o in outs[1:]:
        np.testing.assert_array_equal(np.asarray(o), np.asarray(outs[0]))


def test_truncated_backprop_changes_the_gradient(data):
    p, obs, z = data
    g = {m: jax.grad(lambda q: decode_flow(_apply, obs, z, -1e9, 1e9, params=q, flow_steps=K,
                                           grad_steps=m).sum())(p) for m in (1, K)}
    assert not np.allclose(np.asarray(g[1]['modules_actor']['w1']), np.asarray(g[K]['modules_actor']['w1']))


def test_endpoint_rule_at_last_step_equals_one_step_backprop(data):
    p, obs, z = data
    a_exec, probes = decode_flow_endpoint(_apply, obs, z, -1.0, 1.0, params=p, flow_steps=K, endpoint_from=K - 1)
    a_bptt = decode_flow(_apply, obs, z, -1.0, 1.0, params=p, flow_steps=K, grad_steps=1)
    assert probes.shape == (1, 4, ACT)
    np.testing.assert_allclose(np.asarray(probes[0]), np.asarray(a_exec), atol=1e-6)
    np.testing.assert_allclose(np.asarray(a_exec), np.asarray(a_bptt), atol=1e-6)
    g1 = jax.grad(lambda q: decode_flow_endpoint(_apply, obs, z, -1e9, 1e9, params=q, flow_steps=K,
                                                 endpoint_from=K - 1)[1].sum())(p)
    g2 = jax.grad(lambda q: decode_flow(_apply, obs, z, -1e9, 1e9, params=q, flow_steps=K, grad_steps=1).sum())(p)
    np.testing.assert_allclose(np.asarray(g1['modules_actor']['w2']), np.asarray(g2['modules_actor']['w2']), rtol=1e-5)


def test_meanflow_decode_clips_and_straight_through(data):
    p, obs, z = data

    def big(o, zz, t, u, params=None, is_encoded=True):
        return 3.0 * zz * params['modules_actor']['w2'].sum()
    a, a_raw = decode_meanflow(big, obs, z, -1.0, 1.0, params=p, return_raw=True)
    assert float(jnp.abs(a).max()) <= 1.0 and float(jnp.abs(a_raw).max()) > 1.0
    g_clip = jax.grad(lambda q: decode_meanflow(big, obs, z, -1.0, 1.0, params=q).sum())(p)
    g_ste = jax.grad(lambda q: decode_meanflow(big, obs, z, -1.0, 1.0, params=q, ste=True).sum())(p)
    assert abs(float(g_ste['modules_actor']['w2'].sum())) > abs(float(g_clip['modules_actor']['w2'].sum()))


def _bank_f(obs, z):
    return jax.jit(lambda q: decode_flow(_apply, obs, z, -1.0, 1.0, params=q, flow_steps=K))


def test_projection_commits_delta(data):
    p, obs, z = data
    p_new = jax.tree.map(lambda x: x + 0.3, p)
    bank_f = _bank_f(obs, z)
    f_pre = bank_f(p)
    raw = rms_rows(bank_f(p_new) - f_pre)
    delta = raw / 7.0
    commit, info = project_functional_step(bank_f, p, p_new, f_pre, delta)
    assert info['projected'] and 0.0 < info['alpha'] < 1.0
    assert abs(rms_rows(bank_f(commit) - f_pre) - delta) <= 0.05 * delta + 1e-9
    np.testing.assert_array_equal(np.asarray(commit['modules_critic']['w']), np.asarray(p_new['modules_critic']['w']))


def test_projection_is_a_no_op_below_the_bound(data):
    p, obs, z = data
    p_new = jax.tree.map(lambda x: x + 1e-4, p)
    bank_f = _bank_f(obs, z)
    commit, info = project_functional_step(bank_f, p, p_new, bank_f(p), 10.0)
    assert not info['projected'] and commit is p_new


def test_bisection_fallback_never_worse():
    # a step that saturates in alpha (d ~ alpha^0.1): the proportional iteration cannot reach the tolerance
    pre = {'modules_actor': {'w': jnp.zeros(1)}}
    now = {'modules_actor': {'w': jnp.ones(1)}}
    bank_f = lambda q: jnp.abs(q['modules_actor']['w'])[None] ** 0.1
    f_pre = bank_f(pre)
    _, plain = project_functional_step(bank_f, pre, now, f_pre, 0.3)
    _, bis = project_functional_step(bank_f, pre, now, f_pre, 0.3, bisect=True)
    assert plain['resid'] > 0.05 and bis['resid'] <= plain['resid'] and bis['resid'] <= 0.05


def test_parameter_trust_region(data):
    p, _, _ = data
    p_new = jax.tree.map(lambda x: x + 0.1, p)
    eps = actor_param_dist(p_new, p) / 4
    commit, info = project_param_step(p, p_new, eps)
    assert info['projected']
    assert abs(actor_param_dist(commit, p) - eps) < 1e-5 * eps
    np.testing.assert_allclose(np.asarray(blend_actor(p, p_new, 1.0)['modules_actor']['w1']),
                               np.asarray(p_new['modules_actor']['w1']))


def test_resume_refuses_a_checkpoint_of_another_configuration(tmp_path):
    from ogpo.main import _guard_resume_config
    cfg = {'agent': {'sfg_func_delta': 0.01}, 'training': {'online_steps': 10}, 'experiment': {'run_name': 'a'}}
    _guard_resume_config(str(tmp_path), cfg)                       # first start: records the configuration
    (tmp_path / '100_meta.json').write_text('{}')                  # a rolling checkpoint now exists
    _guard_resume_config(str(tmp_path), {**cfg, 'training': {'online_steps': 20},
                                         'experiment': {'run_name': 'b'}})   # budget / names may change
    with pytest.raises(ValueError):
        _guard_resume_config(str(tmp_path), {**cfg, 'agent': {'sfg_func_delta': 0.0}})


def test_rolling_checkpoint_cleanup_keeps_the_run_config(tmp_path, monkeypatch):
    from ogpo.utils import main_helper
    monkeypatch.setenv('OGPO_CHECKPOINT_ROOT', str(tmp_path))
    d = main_helper.get_checkpoint_dir(0, 'env', 'ogpo')
    import os
    os.makedirs(d)
    for f in ('run_config.json', '100_agent.pkl', '100_buffer.pkl', '100_meta.json'):
        open(os.path.join(d, f), 'w').write('{}')
    main_helper.save_rolling_checkpoint({'w': 1}, [1], 200, 50, 0, 'env', algo_name='ogpo', episode_counter=3)
    assert sorted(os.listdir(d)) == ['200_agent.pkl', '200_buffer.pkl', '200_meta.json', 'run_config.json']
