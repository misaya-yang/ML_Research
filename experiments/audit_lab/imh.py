"""Exact cold IMH moments and binary window distributions."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Cells:
    w: np.ndarray
    q: np.ndarray
    h: np.ndarray

    @property
    def p(self):
        mass = self.q * self.w
        return mass / mass.sum()


def _prefix(values):
    out = np.empty(values.shape[:-1] + (values.shape[-1] + 1,))
    out[..., 0] = 0
    np.cumsum(values, axis=-1, out=out[..., 1:])
    return out


def _suffix(values):
    return _prefix(values[..., ::-1])[..., ::-1]


class SortedKernel:
    """O(N) applications of K and its transpose after sorting."""

    def __init__(self, w, qm):
        if (w.ndim != 1 or qm.shape != w.shape or not w.size
                or not np.all(np.isfinite(w) & (w > 0))
                or not np.all(np.isfinite(qm) & (qm >= 0))
                or not math.isclose(float(qm.sum()), 1, rel_tol=1e-12)):
            raise ValueError("Expected positive finite scores and a proposal law")
        self.order = np.argsort(w, kind="stable")
        self.w, self.q = w[self.order], qm[self.order] / qm.sum()
        self.first = np.searchsorted(self.w, self.w, side="left")
        self.after = np.searchsorted(self.w, self.w, side="right")
        self.qw = self.q * self.w
        alpha = _suffix(self.q)[self.first] + _prefix(self.qw)[self.first] / self.w
        self.stay = np.clip(1 - alpha, 0, 1)

    def apply(self, values):
        head = _prefix(self.qw * values)[self.first] / self.w
        tail = _suffix(self.q * values)[self.first]
        return head + tail + self.stay * values

    def forward(self, mass):
        """Propagate row distributions with the state index on the last axis."""
        head = _prefix(mass)[..., self.after]
        tail = self.w * _suffix(mass / self.w)[..., self.after]
        return self.q * (head + tail) + mass * self.stay


def exact_path_moments(w, qm, h, b, t):
    """Return E[A], E[A²], Ph and Ph² for a window starting at time b."""
    if b < 0 or t < 1:
        raise ValueError("Expected b >= 0 and t >= 1")
    kernel = SortedKernel(np.asarray(w, dtype=float), np.asarray(qm, dtype=float))
    labels = np.asarray(h, dtype=float)[kernel.order]
    mass = kernel.q
    for _ in range(b):
        mass = kernel.forward(mass)

    # Conditional moments of a sum of length j. Expanding (h + next_sum)²
    # accumulates the same triangular lag terms using O(N) working memory.
    square = labels * labels
    first, second = labels, square
    for _ in range(1, t):
        future = kernel.apply(first)
        first = labels + future
        second = square + 2 * labels * future + kernel.apply(second)
    target = kernel.qw / kernel.qw.sum()
    return (np.sum(mass * first) / t, np.sum(mass * second) / t**2,
            np.sum(target * labels), np.sum(target * square))


def G_k(a, d, mu, nu, k):
    return d - 2 * mu * a + mu**2 - (nu - mu**2) / k


def cell_moments(cells, b, t):
    return exact_path_moments(cells.w, cells.q, cells.h, b, t)


def cell_gap(cells, b, t, k):
    return float(G_k(*cell_moments(cells, b, t), k))


def window_sum_pmf(cells, b, t):
    """DP on the binary sum and the last state, starting at X_0 ~ Q."""
    if b < 0 or t < 1 or not np.all((cells.h == 0) | (cells.h == 1)):
        raise ValueError("Expected a positive window and binary labels")
    kernel = SortedKernel(cells.w, cells.q)
    zero = cells.h[kernel.order] == 0
    initial = kernel.q
    for _ in range(b):
        initial = kernel.forward(initial)
    mass = np.zeros((2, initial.size))
    mass[0, zero], mass[1, ~zero] = initial[zero], initial[~zero]
    for _ in range(1, t):
        moved = kernel.forward(mass)
        mass = np.zeros((moved.shape[0] + 1, initial.size))
        mass[:-1, zero] = moved[:, zero]
        mass[1:, ~zero] = moved[:, ~zero]
    pmf = mass.sum(axis=1)
    return pmf / pmf.sum()


def root_bounds(a, d, k):
    linear = 2 * np.asarray(a) + 1 / k
    leading = 1 + 1 / k
    disc = linear**2 - 4 * leading * d
    top = linear + np.sqrt(np.where(disc > 0, disc, np.nan))
    return 2 * d / top, top / (2 * leading)


def roots_Fk(a, d, k):
    lo, hi = root_bounds(a, d, k)
    return (float(lo), float(hi)) if np.isfinite(lo) else None


def kappa(mu0, lo, hi):
    return min(1 - lo / mu0 if lo > 0 else 1,
               1 - (1 - hi) / (1 - mu0) if hi < 1 else 1)


def Pi_p(delta, kap, p):
    if p == 0:
        return delta
    if kap == 1:
        return 1.0
    return min(1.0, math.exp(min(0, math.log(delta) - p * math.log1p(-kap))))
