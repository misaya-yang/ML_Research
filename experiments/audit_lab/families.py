"""The prespecified synthetic cell families and exact gate calibrations."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from .imh import Cells, G_k, cell_gap, cell_moments, kappa, roots_Fk


def layered(weights, masses, plus):
    q, plus = np.asarray(masses), np.asarray(plus)
    return Cells(np.repeat(weights, 2),
                 np.column_stack((q * (1 - plus), q * plus)).ravel(),
                 np.tile([0., 1.], len(q)))


def fair_coin():
    return Cells(np.ones(2), np.full(2, 0.5), np.array([0., 1.]))


def tail_profile(D, D0, r, *, knee=None, right_r=None):
    if knee is not None and D > knee:
        return (D0 / knee)**r * (knee / D)**right_r
    return min(1., (D0 / D)**r)


@dataclass(frozen=True)
class ThreeLayers:
    J: float
    K: float
    m: float
    bulk_mass: float

    def cells(self, beta, theta):
        qb, qh = self.bulk_mass / self.J, self.m / self.K
        ql = 1 - qb - qh
        low_score = (1 - self.bulk_mass - self.m) / ql
        low_plus = (0.5 - qb * beta - qh * theta) / ql
        return layered([low_score, self.J, self.K], [ql, qb, qh], [low_plus, beta, theta])


@dataclass(frozen=True)
class WorldPair:
    safe: Cells
    unsafe: Cells
    beta: float
    s: float
    g_safe: float
    g_unsafe: float
    bracket: float


def calibrated_pair(family, b, t, k, epsilon, *, max_tilt, tilt_tolerance):
    def gap(beta, theta):
        return cell_gap(family.cells(beta, theta), b, t, k)

    if not gap(.5, .5) < 0 < gap(1., .5):
        raise ValueError("The three-layer root is not bracketed in beta")
    beta = brentq(lambda beta: gap(beta, .5), .5, 1., xtol=1e-14)

    def crosses(s):
        return gap(beta, .5 - s) <= -epsilon and gap(beta, .5 + s) >= epsilon

    lo, hi = 0., max_tilt
    if not crosses(hi):
        raise ValueError("The two worlds cannot both cross the requested margins")
    while hi - lo > tilt_tolerance:
        md = (lo + hi) / 2
        if crosses(md):
            hi = md
        else:
            lo = md
    safe, unsafe = family.cells(beta, .5 - hi), family.cells(beta, .5 + hi)
    a0, d0, mu0, nu0 = cell_moments(safe, b, t)
    a1, d1, mu1, nu1 = cell_moments(unsafe, b, t)
    bracket = -2 * a0 + (1 + 1 / k) * (mu0 + mu1) - 1 / k
    return WorldPair(safe, unsafe, beta, hi, float(G_k(a0, d0, mu0, nu0, k)),
                     float(G_k(a1, d1, mu1, nu1, k)), float(bracket))


def gap_and_kappa(cells, b, t, k):
    a, d, mu, nu = cell_moments(cells, b, t)
    gap = float(G_k(a, d, mu, nu, k))
    roots = roots_Fk(a, d, k)
    if gap >= 0 or roots is None or not roots[0] < mu < roots[1]:
        raise ValueError("Expected a strictly safe cell family")
    return gap, float(kappa(mu, *roots)), float(mu), *roots


def i2_cells(family, b, t, k, *, kap=None, epsilon=None):
    if (kap is None) == (epsilon is None):
        raise ValueError("Supply exactly one of kappa and epsilon")

    def gap(beta):
        return cell_gap(family.cells(beta, .5), b, t, k)

    beta_root = brentq(gap, .5, 1., xtol=1e-14)
    if epsilon is not None:
        beta = brentq(lambda beta: gap(beta) + epsilon, .5, beta_root, xtol=1e-14)
    else:
        def distance(beta):
            a, d, mu, _ = cell_moments(family.cells(beta, .5), b, t)
            return kappa(mu, *roots_Fk(a, d, k)) - kap

        beta = brentq(distance, .5, beta_root, xtol=1e-14)
    return family.cells(beta, .5), beta


def heavy_tail(r, j_min, j_max, extra_exponent, label_amplitude):
    j = np.arange(j_min, j_max + 1)
    weights = np.exp2(j.astype(float))
    low = j < 0
    low_mean = weights[low].mean()
    tail = np.exp2(-j[~low] * (1 + r + extra_exponent))
    tail_scale = (1 - low_mean) / np.sum(tail * (weights[~low] - low_mean))
    q = np.zeros(len(j))
    q[~low] = tail_scale * tail
    q[low] = (1 - q.sum()) / np.count_nonzero(low)
    plus = .5 + label_amplitude * j / (1 + np.abs(j))
    return layered(weights, q, plus)


def signed_limit_root(k, t):
    return math.sqrt((1 / k - 1 / t) / (1 + 1 / k))


def center_floor(epsilon, sign, alpha, t, k, heavy_mass, numerical_margin):
    """Select a signed target mean with a margin paid for by the coupling bound."""
    eta = 1 - heavy_mass
    low_w = eta / (1 - alpha)
    budget = epsilon + 4 * t * alpha + numerical_margin
    mu = math.sqrt((sign * budget - 1 / t + 1 / k) / (1 + 1 / k))
    plus = (1 + mu / (heavy_mass - alpha * low_w)) / 2
    low_plus = (.5 - alpha * plus) / (1 - alpha)
    cells = layered([low_w, heavy_mass / alpha], [1 - alpha, alpha], [low_plus, plus])
    signed = Cells(cells.w, cells.q, 2 * cells.h - 1)
    return cells, cell_gap(signed, 0, t, k), plus
