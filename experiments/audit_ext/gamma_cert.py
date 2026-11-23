"""Corollary R2: the Gamma-certificate (no chain paths), vectorized over replicates."""

import math

import numpy as np

from experiments.audit_lab.auditors import empirical_bernstein
from experiments.audit_lab.counts import multinomial


def _quantile_from_counts(score, counts, u):
    order = np.argsort(score, kind="stable")
    n = int(counts[0].sum())
    rank = math.ceil(u * n)
    index = (np.cumsum(counts[:, order], axis=1) >= rank).argmax(axis=1)
    return score[order[index]]


def bernoulli_lcb(successes, m, log_budget):
    """Maurer-Pontil lower bound for a Bernoulli mean from m iid draws."""
    mean = successes / m
    var = mean * (1 - mean) * m / max(m - 1, 1)
    radius = np.sqrt(2 * var * log_budget / m) + 7 * log_budget / (3 * max(m - 1, 1))
    return np.maximum(0, mean - radius)


def A_gamma(rng, score, h, P, Qm, p, q, t, k, delta, reps, C0s=(44,)):
    """Return {C0: issue indicator array}, plus diagnostics. Three banks at delta/3 each.

    P0: n0=ceil(32 log(6/delta)) target rows for the clip level c (5/8 quantile of S).
    P1, P2: halves of the remaining target rows. Q: all q proposal rows.
    """
    n0 = math.ceil(32 * math.log(6 / delta))
    zero = {C0: np.zeros(reps) for C0 in C0s}
    if p < n0 + 4 or q < 2:
        return zero, dict(n0=n0, reason="budget")
    p1 = (p - n0) // 2
    p2 = p - n0 - p1
    lb = math.log(6 / delta)  # one-sided Maurer-Pontil at delta/3
    c = _quantile_from_counts(score, multinomial(rng, n0, P, reps), 5 / 8)
    q_counts = multinomial(rng, q, Qm, reps)
    lo, _ = empirical_bernstein(q_counts, np.minimum(score[None, :], c[:, None]) / c[:, None], lb)
    ZL = c * lo
    ok = ZL > 0
    safeZ = np.where(ok, ZL, 1.0)
    y = np.minimum(score[None, :] / safeZ[:, None], t) / t
    _, UG = empirical_bernstein(multinomial(rng, p1, P, reps), y, lb)
    # Disjoint pairs of iid labels: (h_i-h_j)^2 is Bernoulli(2 Var) for binary h.
    pi = float(P @ h)
    pairs = p2 // 2
    discord = rng.binomial(pairs, 2 * pi * (1 - pi), size=reps)
    VL = bernoulli_lcb(discord, pairs, lb) / 2
    out = {C0: (ok & (C0 * UG < VL / k)).astype(float) for C0 in C0s}
    return out, dict(n0=n0, mean_UG=float(np.mean(UG)), mean_VL=float(np.mean(VL)),
                     frac_ZL_pos=float(ok.mean()))
