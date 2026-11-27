"""Episode-boundary bookkeeping of SFG for the online runner.

At every episode end the proposal actor theta~ (trained during the episode) is compared with theta, the
actor committed at the previous boundary, on a fixed probe bank. With `sfg_func_delta > 0` the commit is
projected so that the functional step is at most delta (+-5%); with `sfg_param_delta > 0` the parameter
step is bounded instead (ablation). The same bookkeeping logs the drift in action space for every SFG
run, bounded or not, to <save_dir>/sfg_steps.jsonl:

  delta_raw         functional step of the proposal, d(theta, theta~)
  alpha, projected  blend factor of the commit theta + alpha (theta~ - theta)
  step_committed    functional step actually committed
  path_length       sum of committed steps since the start of the online phase
  net_distance      functional distance of the committed actor from the actor at the start of the online phase
  param_step_raw / param_step_committed   the same steps in parameter space (L2 over the actor)
  ep_success, ep_len                       the training episode that just ended (collected by theta~)
"""
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.modules.sfg_helper import (actor_param_dist, blend_actor, build_probe_bank,
                                           make_bank_decoder, project_functional_step,
                                           project_param_step, rms_rows)


class SFGProjection:

    def __init__(self, agent, config, train_dataset, save_dir, is_resumed, start_online_step):
        self.func_delta = float(config.get('sfg_func_delta', 0.0))
        self.param_delta = float(config.get('sfg_param_delta', 0.0))
        self.ref_pen = float(config.get('sfg_ref_pen', 0.0))
        self.bisect = bool(config.get('sfg_proj_bisect', False))
        assert self.param_delta == 0.0 or (self.func_delta == 0.0 and self.ref_pen == 0.0), (
            "sfg_param_delta is exclusive with sfg_func_delta and sfg_ref_pen")
        assert train_dataset is not None, "the probe bank is drawn from the demonstration states"

        bank_S, bank_W = build_probe_bank(
            train_dataset['observations'], int(config['sfg_probe_states']), int(config['sfg_probe_latents']),
            int(agent.config['full_act_dim']), config['meanflow_latent_dist'], int(config['sfg_probe_seed']))
        self.bank_f = make_bank_decoder(agent, bank_S, bank_W)

        self.log_path = os.path.join(save_dir, 'sfg_steps.jsonl')
        fork_path = os.path.join(save_dir, 'sfg_fork_actions.npy')
        self.theta_pre = agent.actor_network.params
        self.f_pre = self.bank_f(self.theta_pre)
        self.path_length = 0.0
        if is_resumed:
            # net_distance is measured from the actor at the start of the online phase, saved at the start
            assert os.path.exists(fork_path), f"resumed SFG run but no {fork_path}"
            self.f_fork = jnp.asarray(np.load(fork_path))
            if os.path.exists(self.log_path):
                # the run went past its last checkpoint before stopping: drop the rows after it
                with open(self.log_path) as fh:
                    rows = [json.loads(l) for l in fh if l.strip()]
                keep = [r for r in rows if int(r['online_step']) <= int(start_online_step) - 1]
                tmp = self.log_path + '.tmp'
                with open(tmp, 'w') as fh:
                    for r in keep:
                        fh.write(json.dumps(r) + '\n')
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(tmp, self.log_path)
                if keep:
                    self.path_length = float(keep[-1]['path_length'])
        else:
            os.makedirs(save_dir, exist_ok=True)
            if os.path.exists(self.log_path):   # left by an earlier run in the same directory
                os.replace(self.log_path, self.log_path + time.strftime('.bak_%Y%m%d_%H%M%S'))
            self.f_fork = self.f_pre
            np.save(fork_path, np.asarray(self.f_fork))
        self.log = open(self.log_path, 'a')

        what = (f"functional bound delta={self.func_delta}" if self.func_delta > 0 else
                f"parameter bound eps={self.param_delta}" if self.param_delta > 0 else
                "NO bound (measuring the drift only)")
        print(f"[sfg] episode-boundary projection: {what}; probe bank {bank_S.shape[0]} rows "
              f"({config['sfg_probe_states']} states x {config['sfg_probe_latents']} latents)"
              + (f"; penalty sfg_ref_pen={self.ref_pen}" if self.ref_pen > 0 else ""), flush=True)

    @property
    def bounded(self):
        """True when commits happen at episode boundaries, so evals/saves wait for a boundary."""
        return self.func_delta > 0 or self.param_delta > 0

    def init_agent(self, agent):
        if self.ref_pen > 0:
            agent = agent.replace(sfg_ref_params=jax.tree.map(lambda x: x.copy(), agent.actor_network.params))
        return agent

    def commit(self, agent, online_step, episode, extra=None):
        """Project the proposal actor (if bounded), log the step, and move the reference to the commit."""
        theta_now = agent.actor_network.params
        if self.func_delta > 0:
            commit, info = project_functional_step(self.bank_f, self.theta_pre, theta_now, self.f_pre,
                                                   self.func_delta, bisect=self.bisect)
        elif self.param_delta > 0:
            commit, info = project_param_step(self.theta_pre, theta_now, self.param_delta)
            info = {'delta_raw': rms_rows(self.bank_f(theta_now) - self.f_pre), **info}
        else:
            commit = theta_now
            info = {'delta_raw': rms_rows(self.bank_f(theta_now) - self.f_pre), 'alpha': 1.0,
                    'projected': False}
        if info['projected']:
            # the actor's optimizer state is kept as produced by the unprojected updates
            agent = agent.replace(actor_network=agent.actor_network.replace(params=commit))
        f_now = self.bank_f(agent.actor_network.params)
        step = rms_rows(f_now - self.f_pre)
        self.path_length += step
        p_raw = actor_param_dist(theta_now, self.theta_pre)
        p_com = p_raw if agent.actor_network.params is theta_now else \
            actor_param_dist(agent.actor_network.params, self.theta_pre)
        self.log.write(json.dumps({
            'online_step': int(online_step), 'episode': int(episode), **info,
            'step_committed': step, 'path_length': self.path_length,
            'net_distance': rms_rows(f_now - self.f_fork),
            'param_step_raw': p_raw, 'param_step_committed': p_com, **(extra or {})}) + '\n')
        self.log.flush()
        self.theta_pre, self.f_pre = agent.actor_network.params, f_now
        return self.init_agent(agent)
