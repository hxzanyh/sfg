"""Episode-boundary projection of the policy's functional step (SFG's bound).

With f_theta(s, eps) the clipped one-step decode and B = {(s_i, eps_i)} a fixed probe bank,

    D(theta, theta') = sqrt( mean_i || f_theta(s_i, eps_i) - f_theta'(s_i, eps_i) ||^2 )

(squared norm over the flattened action chunk, mean over the bank). At the end of an episode the
proposal theta (after all of the episode's actor updates) is compared with the committed policy
theta_prev of the episode start. If D > delta, the step is shrunk along the parameter path,
theta_prev + alpha (theta - theta_prev), with alpha found by proportional iteration
alpha <- min(alpha * delta / D(alpha), 1) from alpha_0 = delta / D(1): at most `max_iters`
evaluations, accepted at |D(alpha) - delta| <= tol * delta, otherwise the best evaluated alpha.
alpha never exceeds 1, so the step is never extrapolated. The actor optimizer state is not touched.
"""
import torch


def rms(d):
    """sqrt(mean over rows of the squared row norm)."""
    return float(d.pow(2).sum(-1).mean().sqrt())


def project_functional_step(params, theta_prev, f_prev, decode_bank, delta, tol=0.05, max_iters=8):
    """Project in place. params: trainable tensors (proposal on entry); theta_prev: detached copies
    at the episode start; f_prev: decode_bank() at theta_prev.

    Returns (delta_raw, alpha, iters, resid). params hold theta_prev + alpha (theta - theta_prev).
    """
    delta_raw = rms(decode_bank() - f_prev)
    iters, resid, alpha = 0, 0.0, 1.0
    if delta_raw > delta and delta_raw > 1e-12:
        d_par = [p.detach() - q for p, q in zip(params, theta_prev)]
        alpha = min(delta / max(delta_raw, 1e-12), 1.0)
        best = (alpha, float('inf'))
        for iters in range(1, max_iters + 1):
            with torch.no_grad():
                for p, q, d in zip(params, theta_prev, d_par):
                    p.copy_(q + alpha * d)
            cur = rms(decode_bank() - f_prev)
            resid = abs(cur - delta) / delta
            if resid < best[1]:
                best = (alpha, resid)
            if resid <= tol:
                break
            alpha = min(alpha * delta / max(cur, 1e-12), 1.0)
        if resid > tol:                                       # not converged: take the best alpha
            alpha, resid = best
            with torch.no_grad():
                for p, q, d in zip(params, theta_prev, d_par):
                    p.copy_(q + alpha * d)
    return delta_raw, alpha, iters, resid
