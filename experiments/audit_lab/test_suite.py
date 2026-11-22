"""Deterministic numerical checks for Plan §2."""

from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import binom

from .counts import LR_oracle, NORMAL_THRESHOLD, e_p_ratio, multinomial
from .imh import Cells, G_k, cell_moments, exact_path_moments, window_sum_pmf


def _dense_moments(w, q, h, b, t):
    kernel = w[None, :] / w[:, None]
    np.minimum(kernel, 1, out=kernel)
    kernel *= q
    kernel.flat[:: len(w) + 1] += 1 - kernel.sum(axis=1)
    pi = q
    for _ in range(b):
        pi = np.einsum("i,ij->j", pi, kernel)
    future = [h]
    for _ in range(1, t):
        future.append(np.einsum("ij,j->i", kernel, future[-1]))
    a, d = 0, 0
    for i in range(t):
        a += np.sum(pi * h)
        d += np.sum(pi * h * h)
        for lag in range(1, t - i):
            d += 2 * np.sum(pi * h * future[lag])
        pi = np.einsum("i,ij->j", pi, kernel)
    p = q * w / np.sum(q * w)
    return np.array([a / t, d / t**2, np.sum(p * h), np.sum(p * (h**2))])


def _check_dense(rng, check_memory):
    worst = 0.0
    for n in (1, 2, 7, 19, 47, 127, 509, 2000):
        check_memory()
        w = np.exp2(rng.integers(-12, 13, n).astype(float))
        h = rng.uniform(0.01, 1, n)
        q = rng.dirichlet(np.ones(n))
        b, t = int(rng.integers(0, 6)), int(rng.integers(2, 11))
        reference = _dense_moments(w, q, h, b, t)
        value = exact_path_moments(w, q, h, b, t)
        rescaled = exact_path_moments(8 * w[::-1], q[::-1], h[::-1], b, t)
        worst = max(worst, np.max(np.abs((value - reference) / reference)),
                    np.max(np.abs((rescaled - reference) / reference)))
    assert worst < 1e-10, worst
    print(f"dense: N<=2000, max relative error {worst:.6g}")


def _check_mc(check_memory):
    rng = np.random.default_rng(42027)
    cells = Cells(np.array([0.125, 1., 1., 7.5, 18.]),
                  np.array([.2, .19, .32, .25, .04]),
                  np.array([.05, .25, .9, .3, 1.]))
    b, t, n = 3, 7, 10**6
    sums, squares = np.zeros(4), np.zeros(4)
    for _ in range(n // 50000):
        check_memory()
        state = rng.choice(len(cells.q), p=cells.q, size=50000)
        window = np.zeros(50000)
        for i in range(b + t):
            if i:
                y = rng.choice(len(cells.q), p=cells.q, size=50000)
                state = np.where(rng.random(50000) < cells.w[y] / cells.w[state], y, state)
            if i >= b:
                window += cells.h[state]
        avg = window / t
        target = rng.choice(len(cells.p), p=cells.p, size=50000)
        h = cells.h[target]
        value = np.array([avg, avg**2, h, h**2])
        sums += value.sum(axis=1)
        squares += (value**2).sum(axis=1)
    mean = sums / n
    se = np.sqrt((squares / n - mean**2) / (n - 1))
    errors = np.abs(mean - cell_moments(cells, b, t)) / se
    assert np.all(errors < 4), errors
    print(f"MC: {n} independent chains, |error|/SE " + ", ".join(f"{x:.3f}" for x in errors))


def _check_cap():
    source = Path(__file__).resolve().parents[1] / "finite_state" / "cap_equality.py"
    oracle = subprocess.check_output([sys.executable, str(source)], text=True)
    latest = float(oracle.splitlines()[-1].split()[-1])
    kap = 1 / math.sqrt(6)
    x = 1e-10
    cells = Cells(np.array([1., 1., 1 + kap / ((1 - kap) * x)]),
                  np.array([.5 - x, .5, x]), np.array([0., 1., 0.]))
    ratio = G_k(*cell_moments(cells, 0, 4), 2) / x
    assert abs(latest - 0.0063138) < 1e-7, latest
    assert abs(ratio - latest) < 2e-6, (ratio, latest)
    print(f"cap_equality: G/x {ratio:.8f}, mpmath {latest:.8f} at x={x:g}")


def _fraction_e(scores, counts, p):
    coefficients = [Fraction(1)] + [Fraction(0)] * p
    for z, n in zip(scores, counts):
        for _ in range(n):
            for j in range(p, 0, -1):
                coefficients[j] += z * coefficients[j - 1]
    return coefficients[p]


def _check_ep(rng, check_memory):
    worst_error = 0.
    for i in range(96):
        check_memory()
        counts = [int(x) for x in rng.integers(0, 6, 8)]
        x = [Fraction(int(k), int(l)) for k, l in rng.integers(1, 24, (8, 2))]
        if i % 3 == 0:
            x[-1] *= 2**40
        y = [z * Fraction(int(k), 4) for z, k in zip(x, rng.integers(0, 5, 8))]
        p = int(rng.integers(0, min(sum(counts), 16) + 1))
        reference = float(_fraction_e(y, counts, p) / _fraction_e(x, counts, p))
        value = e_p_ratio(np.array(x, dtype=float), np.array(y, dtype=float), counts, p)
        error = abs((value - reference) / reference) if reference else abs(value)
        worst_error = max(worst_error, error)
    assert worst_error < 1e-9, worst_error
    for p in (0, 1, 8, 16, 128):
        n, successes = 2**30, 2**29 - 1
        value = e_p_ratio(np.ones(2), np.array([0., 1.]), [n - successes, successes], p)
        expected = math.prod((successes - i) / (n - i) for i in range(p))
        assert abs((value - expected) / expected) < 1e-9, (p, value, expected)
    print(f"e_p: max relative error vs fraction DP {worst_error:.6g}; hypergeometric check passed")


def _check_counts(rng):
    h = np.array([0., 1., 1.])
    q = np.array([.5, .3, .2])
    t = 12
    fair = window_sum_pmf(Cells(np.ones(3), q, h), 3, t)
    assert np.max(np.abs(fair - binom.pmf(np.arange(t + 1), t, .5))) < 1e-14
    weighted = Cells(np.array([.1, 3., 30.]), q, h)
    pmf = window_sum_pmf(weighted, 4, t)
    values = np.arange(t + 1) / t
    a, d, _, _ = cell_moments(weighted, 4, t)
    assert np.allclose([np.sum(pmf * values), np.sum(pmf * values**2)], [a, d],
                       rtol=1e-13, atol=0.)

    n = 2**60
    p = np.array([2.0**-50, .2, .3, .5 - 2.0**-50])
    cnt = multinomial(rng, n, p, 10000, normal_threshold=NORMAL_THRESHOLD)
    assert np.all(cnt.sum(axis=1) == n)
    centered = cnt - n * p
    for i in range(4):
        se = math.sqrt(n * p[i] * (1 - p[i]) / len(cnt))
        assert abs(centered[:, i].mean()) < 5 * se
    corr = np.corrcoef(centered, rowvar=False)[1, 3]
    expected = -math.sqrt(p[1] * p[3] / ((1 - p[1]) * (1 - p[3])))
    assert abs(corr - expected) < 0.03, corr

    same = LR_oracle(q, q, q, q, 4, 32, .05, 1000, rng=rng)
    assert math.isclose(same.power, .05) and same.null_calibration == .05
    print("counts: binary DP, constrained large-count covariance and ROC ties passed")


def run(check_memory):
    rng = np.random.default_rng(20261122)
    _check_dense(rng, check_memory)
    _check_mc(check_memory)
    _check_cap()
    _check_ep(rng, check_memory)
    _check_counts(rng)
