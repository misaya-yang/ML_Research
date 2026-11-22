"""Audit decisions from the observed scores, labels and cold-path tallies."""

import math

import numpy as np
from scipy.stats import beta

from .counts import log_ep_ratios
from .imh import root_bounds


def hoeffding(paths, gamma):
    """One lower bound for a and one upper bound for d, sharing gamma."""
    a, d = paths.averages()
    radius = math.sqrt(math.log(2 / gamma) / (2 * paths.n))
    return np.maximum(0, a - radius), np.minimum(1, d + radius)


def _safe_roots(paths, delta, k):
    lo, hi = root_bounds(*hoeffding(paths, delta), k)
    good = (0 < lo) & (lo < hi) & (hi < 1)
    return lo, hi, good


def A_ep(score, h, p_counts, q_pool, paths, delta, k, gamma):
    out = np.zeros(len(q_pool))
    if paths.n == 0:
        return out
    lo, hi, good = _safe_roots(paths, gamma, k)
    if not np.any(good):
        return out
    p = int(p_counts[0].sum())
    pool = p_counts[good] + q_pool[good]
    log_plus, log_minus = log_ep_ratios(score, (score * h, score * (1 - h)), pool, p)
    log_value = np.minimum(log_plus - p * np.log(lo[good]),
                           log_minus - p * np.log1p(-hi[good]))
    out[good] = np.exp(np.minimum(0, math.log(delta - gamma) + log_value))
    return out


def A_knownZ(score, h, mean_counts, paths, Z, delta, k):
    n = int(mean_counts[0].sum())
    if n == 0 or paths.n == 0:
        return np.zeros(len(mean_counts))
    cap = n**0.25
    w = score / Z
    radius = cap * math.sqrt(math.log(8 / delta) / (2 * n))
    mu_lo = np.sum(mean_counts * np.minimum(w * h, cap), axis=1) / n - radius
    mu_hi = 1 - (np.sum(mean_counts * np.minimum(w * (1 - h), cap), axis=1) / n - radius)
    lo, hi, good = _safe_roots(paths, delta / 2, k)
    return (good & (lo < mu_lo) & (mu_hi < hi)).astype(float)


def empirical_bernstein(counts, values, log_budget):
    n = int(counts[0].sum())
    if n < 2:
        return np.zeros(len(counts)), np.ones(len(counts))
    mean = np.sum(counts * values, axis=1) / n
    variance = np.sum(counts * (values - mean[:, None])**2, axis=1) / (n - 1)
    radius = np.sqrt(2 * variance * log_budget / n) + 7 * log_budget / (3 * (n - 1))
    return np.maximum(0, mean - radius), np.minimum(1, mean + radius)


def _gate(mu_lo, mu_hi, paths, delta, k):
    a, d = hoeffding(paths, delta)

    def upper_gap(mu):
        return d - (2 * a + 1 / k) * mu + (1 + 1 / k) * mu**2

    return ((mu_lo <= mu_hi) & (np.maximum(upper_gap(mu_lo), upper_gap(mu_hi)) < 0)).astype(float)


def A_clip(score, h, p_counts, pilot, rest, paths, delta, k, c0=16):
    p, n_pilot, n_rest = (int(counts[0].sum()) for counts in (p_counts, pilot, rest))
    if p == 0 or n_pilot == 0 or n_rest == 0 or paths.n == 0:
        return np.zeros(len(p_counts))
    order = np.argsort(score, kind="stable")
    # ceil(31 n / 32) without rounding an integer tally through float.
    rank = n_pilot - n_pilot // 32
    index = (np.cumsum(pilot[:, order], axis=1) >= rank).argmax(axis=1)
    cutoff = (n_rest / p) * score[order[index], None]
    head = np.minimum(1, score / cutoff)
    tail = np.maximum(0, 1 - cutoff / score)
    ell = math.log(c0 / delta)
    u1_lo, u1_hi = empirical_bernstein(rest, head * h, ell)
    u0_lo, u0_hi = empirical_bernstein(rest, head * (1 - h), ell)
    c1_lo, c1_hi = empirical_bernstein(p_counts, tail * h, ell)
    c0_lo, c0_hi = empirical_bernstein(p_counts, tail * (1 - h), ell)
    lo_denom, hi_denom = u1_lo + u0_hi, u1_hi + u0_lo
    lo = np.divide(u1_lo, lo_denom, out=np.zeros_like(u1_lo), where=lo_denom > 0)
    hi = np.divide(u1_hi, hi_denom, out=np.ones_like(u1_hi), where=hi_denom > 0)
    mu_lo = np.clip(lo + (1 - lo) * c1_lo - lo * c0_hi, 0, 1)
    mu_hi = np.clip(hi + (1 - hi) * c1_hi - hi * c0_lo, 0, 1)
    return _gate(mu_lo, mu_hi, paths, delta / 2, k)


def A_replay(h, p_counts, paths, delta, k):
    p = int(p_counts[0].sum())
    if p == 0 or paths.n == 0:
        return np.zeros(len(p_counts))
    successes = p_counts[:, h == 1].sum(axis=1)
    lo = np.where(successes > 0, beta.ppf(delta / 4, successes, p - successes + 1), 0)
    hi = np.where(successes < p, beta.ppf(1 - delta / 4, successes + 1, p - successes), 1)
    return _gate(lo, hi, paths, delta / 2, k)
