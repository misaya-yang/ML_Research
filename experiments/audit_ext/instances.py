"""Synthetic finite-state pairs for X9 (normalized so that E_Q W = 1)."""

import numpy as np


def _normalize(w, q, floor=1e-14):
    q = np.asarray(q, dtype=float)
    if floor:
        q = np.maximum(q, floor)
    q = q / q.sum()
    w = np.asarray(w, dtype=float)
    return w / (q @ w), q


def random_instance(rng, kind, n):
    """Generators of experiments/finite_state/risk_bound.py (kinds 0-2) and risk_bound_refined.py (kind 3)."""
    if kind == 0:
        w = np.exp(rng.normal(0, rng.uniform(1, 5), n))
    elif kind == 1:
        w = rng.pareto(rng.uniform(.3, 1.5), n) + rng.uniform(1e-4, 1)
    elif kind == 2:
        w = np.r_[np.full(n // 2, rng.uniform(1e-4, 1e-2)), rng.uniform(1, 10**rng.uniform(1, 4), n - n // 2)]
    else:
        m = int(rng.integers(1, 4))
        w = np.r_[rng.uniform(.2, 3, n - m), 10**rng.uniform(2, 7, m)]
    q = rng.dirichlet(np.full(n, rng.uniform(.3, 3)))
    if kind == 3:
        theta = rng.uniform(.01, .6)
        P = q * w
        far = w > 50
        P[far] *= theta / P[far].sum()
        P[~far] *= (1 - theta) / P[~far].sum()
        q = P / w
    return _normalize(w, q)


def pareto_low(eta0, r, bins=160, tail=1e-12, dark_w=1e-9):
    """P: Pareto(scale 1/eta0, index r) w.p. 1-eta0/2, W=0.5 w.p. eta0/2.

    Q = P/W on these states plus one dark atom carrying the remaining Q mass
    (needed for a probability law; its P-mass is ~1e-9). Pareto discretized on
    geometric tail-probability bins down to `tail`, at the P-conditional mean in each bin.
    """
    D1 = 1 / eta0
    edges_tail = np.geomspace(1.0, tail, bins + 1)
    v = D1 * edges_tail**(-1 / r)
    mass = edges_tail[:-1] - edges_tail[1:]
    # conditional mean of W within [v_i, v_{i+1}) for Pareto: use the midpoint in tail-prob space
    wbin = D1 * np.sqrt(edges_tail[:-1] * edges_tail[1:])**(-1 / r)
    Pp = (1 - eta0 / 2) * mass / mass.sum()
    w = np.r_[wbin, 0.5]
    P = np.r_[Pp, eta0 / 2]
    q = P / w
    qd = 1 - q.sum()
    if qd <= 0:
        raise ValueError("No room for a dark atom")
    w = np.r_[w, dark_w]
    q = np.r_[q, qd]
    return _normalize(w, q)


def lognormal(sigma, bins=256):
    """W=exp(sigma Z - sigma^2/2) under Q, Z~N(0,1), Q discretized into equal-mass bins."""
    from scipy.stats import norm
    u = (np.arange(bins) + .5) / bins
    z = norm.ppf(u)
    w = np.exp(sigma * z - sigma**2 / 2)
    return _normalize(w, np.full(bins, 1 / bins))
