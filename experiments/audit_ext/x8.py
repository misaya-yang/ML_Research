"""X8: tail-dependent P-jump cost with the full two-time replicate (Lemma E4, P10)."""

import math
import time

import numpy as np


def bulk_scale(eta0, r):
    """D_{1/2}=inf{D: P(W>D)<=1/2} for (1-eta0/2) Pareto(1/eta0, r) + eta0/2 at W=0.5."""
    return (1 / eta0) * (2 * (1 - eta0 / 2))**(1 / r)


def overlap(eta0, r):
    """eta=E_P min(1,1/W): Pareto part (1-eta0/2) * eta0 * r/(r+1), low atom eta0/2."""
    return (1 - eta0 / 2) * eta0 * r / (r + 1) + eta0 / 2


def jump_tests(rng, k, D1, r, low):
    """Tests of the high-only lag jump of jump_cost.py at lag k (no height cap).

    Heights V_n=k/Gamma_n >= 1 in descending order; each test uses a fresh target draw
    and succeeds when W <= V. Returns (number of tests, exhausted flag).
    """
    if k <= 0:
        return 0, False
    tests, G, chunk = 0, 0.0, 16
    while True:
        Gs = G + np.cumsum(rng.exponential(size=chunk))
        nv = int(np.searchsorted(Gs, k, side="right"))
        if nv:
            W = D1 * rng.random(nv)**(-1 / r)
            W[rng.random(nv) < low] = 0.5
            ok = np.flatnonzero(W <= k / Gs[:nv])
            if ok.size:
                return tests + int(ok[0]) + 1, False
            tests += nv
        if nv < chunk:
            return tests, True
        G = float(Gs[-1])
        chunk = min(chunk * 2, 1 << 16)


def cell(task):
    start = time.process_time()
    rng = np.random.default_rng(task["seed"])
    eta0, r, t, reps = task["eta0"], task["r"], task["t"], task["reps"]
    D1, low = 1 / eta0, eta0 / 2
    costs = np.empty(reps)
    exhausted = 0
    for n in range(reps):
        i, j = rng.integers(0, t, size=2)
        total = 0
        for lag in (min(i, j), abs(int(i) - int(j))):
            tests, gone = jump_tests(rng, int(lag), D1, r, low)
            total += tests
            exhausted += gone
        costs[n] = total
    d_half = bulk_scale(eta0, r)
    x = d_half / (eta0 * t)
    return dict(eta0=eta0, r=r, t=t, reps=reps, mean_cost=float(costs.mean()),
                se=float(costs.std(ddof=1) / math.sqrt(reps)), median_cost=float(np.median(costs)),
                exhausted_fraction=exhausted / (2 * reps), D_half=d_half, x=x,
                eta=overlap(eta0, r), target_samples=int(costs.sum()),
                cpu_seconds=time.process_time() - start)
