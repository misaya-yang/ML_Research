"""Acceptance tests for the audit_ext components."""

import math

import numpy as np

from experiments.audit_lab.counts import multinomial, path_tallies
from experiments.audit_lab.families import ThreeLayers, calibrated_pair, tail_profile
from experiments.audit_lab.imh import Cells, cell_moments, exact_path_moments, window_sum_pmf
from .gamma_cert import A_gamma
from .instances import lognormal, pareto_low, random_instance
from .risk import cold_risks, envelopes
from .x10 import gate, instance, sigma_K, x10a_task
from .x1e import A_clip_grid, DarkThreeLayers, clip_interval, _quantile
from .x8 import jump_tests


def _risk_vs_moments(rng):
    worst = 0.
    for i in range(40):
        w, q = random_instance(rng, i % 4, int(rng.integers(5, 60)))
        b, t = int(rng.integers(0, 30)), int(rng.integers(1, 40))
        H = np.column_stack([(w >= np.median(w)).astype(float), rng.random(len(w))])
        R, mu, _ = cold_risks(w, q, H, b, t)
        for j in range(2):
            a, d, m, _ = exact_path_moments(w, q, H[:, j], b, t)
            ref = d - 2 * m * a + m * m
            worst = max(worst, abs(R[j] - ref) / max(ref, 1e-300))
    assert worst < 1e-8, worst
    print(f"risk: dense batched R vs exact_path_moments, max relative error {worst:.3g}")


def _gamma_floor(rng):
    low = np.inf
    for i in range(200):
        w, q = random_instance(rng, i % 4, int(rng.integers(5, 80)))
        for t in (2, 8, 512):
            g, _, _ = envelopes(w, q, 0, t)
            low = min(low, g * t)
    for s in (1, 2, 3):
        w, q = lognormal(s)
        assert abs(q @ w - 1) < 1e-12
    for e in (2**-8, 2**-10):
        w, q = pareto_low(e, .5)
        assert abs(q @ w - 1) < 1e-12 and abs(q.sum() - 1) < 1e-12
    assert low >= 1 - 1e-9, low
    print(f"envelopes: min over random instances of E_P min(W,t) = {low:.4f} (>= 1 for t >= 2); families normalized")


def _x10(rng):
    cfg = dict(eps_den=16, tilt=.25, fit_thetas=[.1 + .1 * i for i in range(9)])
    row = x10a_task(dict(k=2, beta=2, c1=1, c2=4, t=1024, rho=.01, cfg=cfg))["row"]
    assert abs(row["G_safe_over_eps"] + 1.98) < .01 and abs(row["G_unsafe_over_eps"] - 1.99) < .01, row
    assert abs(row["KL_P_const"] - 140.7) < .2 and abs(row["KL_Q_const"] - 352.7) < .2, row
    # matrix-power burn-in equals the stepwise burn-in
    th = np.array([.4, .6])
    s0 = sigma_K(2, 2, 1, 4) * .01
    w, q, h, _ = instance(th, .01, s0, 1, 4, 64)
    g = gate(th, .01, s0, 1, 4, 64, 37, 2)
    for a in range(2):
        R, _, _ = cold_risks(w, q[a], h[:, None], 37, 64)
        assert abs(R[0] - g["R"][a]) < 1e-12 * max(1, R[0]), (R, g["R"])
    print("x10: reproduces variance_floor.log (k=2, beta=2, c=(1,4), t=1024, rho=.01); batched gate equals dense risk")


def _naive_jump(rng, k, D1, r, low):
    G = np.cumsum(rng.exponential(size=4 * k + 64))
    V = k / G
    V = V[V >= 1]
    W = D1 * rng.random(len(V))**(-1 / r)
    W[rng.random(len(V)) < low] = .5
    ok = np.flatnonzero(W <= V)
    return ok[0] + 1 if ok.size else len(V)


def _x8(rng):
    n = 20000
    a = np.array([jump_tests(rng, 1, 1., 1., 1.)[0] for _ in range(n)])
    assert abs(a.mean() - (1 - math.exp(-1))) < 4 * a.std() / math.sqrt(n), a.mean()
    for k in (16, 300):
        x = np.array([jump_tests(rng, k, 256., .5, 1 / 512)[0] for _ in range(4000)])
        y = np.array([_naive_jump(rng, k, 256., .5, 1 / 512) for _ in range(4000)])
        se = math.sqrt(x.var() / len(x) + y.var() / len(y))
        assert abs(x.mean() - y.mean()) < 4 * se, (k, x.mean(), y.mean())
    print("x8: chunked jump sampler agrees with the all-heights sampler and the k=1 closed form")


def _x1e(rng):
    psi = tail_profile(2**20, 2048, .5)
    fam = DarkThreeLayers(ThreeLayers(2048, 2**20, psi / 12, .75), 2**-8)
    cells = fam.cells(.8, .5)
    mu = float(cells.p @ cells.h)
    reps, p, q = 2000, 4000, 4000 * 2**20
    anchor = multinomial(rng, 163, cells.p, reps)
    cert = multinomial(rng, p - 163, cells.p, reps)
    qc = multinomial(rng, q // 2, cells.q, reps, normal_threshold=10**12)
    a = _quantile(cells.w, anchor, .25)
    js = range(-3, math.ceil(math.log2(2**20)) + 2)
    ell = math.log(32 * len(js) / .05)
    lo_all, hi_all = np.zeros(reps), np.ones(reps)
    miss_level = 0
    for j in js:
        lo, hi = clip_interval(cells.w, cells.h, cert, qc, a * 2**20 * 2.0**(-j), ell)
        miss_level = max(miss_level, np.mean((lo > mu) | (hi < mu)))
        lo_all, hi_all = np.maximum(lo_all, lo), np.minimum(hi_all, hi)
    miss = np.mean((lo_all > mu) | (hi_all < mu))
    assert miss <= .05 / 4 + 3 * math.sqrt(.0125 / reps), miss
    # validity of the full certificate on an unsafe world
    worlds = calibrated_pair(fam, 0, 8, 2, psi / 2000, max_tilt=.072, tilt_tolerance=1e-12)
    pmf = window_sum_pmf(worlds.unsafe, 0, 8)
    paths = path_tallies(rng, worlds.unsafe, 0, 8, q // 2, reps, pmf=pmf, normal_threshold=10**12)
    issue = A_clip_grid(worlds.unsafe.w, worlds.unsafe.h, multinomial(rng, 163, worlds.unsafe.p, reps),
                        multinomial(rng, p, worlds.unsafe.p, reps),
                        multinomial(rng, q // 2, worlds.unsafe.q, reps, normal_threshold=10**12),
                        paths, .05, 2, 2**20)
    assert issue.mean() <= .05, issue.mean()
    print(f"x1e: intersected interval misses mu in {miss:.4f} of reps (max per level {miss_level:.4f}); "
          f"unsafe issue rate {issue.mean():.4f}")


def _gamma(rng):
    w, q = lognormal(1)
    P = q * w
    h = (w >= np.median(w)).astype(float)
    out, diag = A_gamma(rng, w, h, P, q, 2**14, 2**14, 4096, 2, .05, 2000)
    # validity ingredients hold on almost every replicate
    assert diag["frac_ZL_pos"] > .99
    g, _, _ = envelopes(w, q, 0, 4096)
    assert diag["mean_UG"] >= g, (diag, g)
    # an unsafe instance (small t) never issues
    w, q = pareto_low(.25, .5)
    P = q * w
    h = (w >= np.quantile(w[P > 1e-6], .5)).astype(float)
    R, mu, nu = cold_risks(w, q, h[:, None], 0, 8)
    assert R[0] - (nu[0] - mu[0]**2) / 2 >= 0, (R, nu - mu**2)
    out2, _ = A_gamma(rng, w, h, P, q, 2**16, 2**18, 8, 2, .05, 2000)
    assert out2[44].sum() == 0
    print(f"gamma_cert: mean U_Gamma {diag['mean_UG']:.3e} >= Gamma {g:.3e}; unsafe instance issued 0/2000")


def _x11():
    from .x11 import chain_R, sof
    t, d, rho = 2**11, 64., 0.1
    m = sof.moments(*sof.base(t, d, rho), t)
    th = np.array([.3, .6, .2, .9, .4])
    a, b = sof.R(m, th), chain_R(m, th, t)
    assert abs(a - b) / b < 1e-12, (a, b)
    print(f"x11: quadratic-form R {a:.15f} vs 10-cell chain {b:.15f} (small_overlap_floor.py cross-check)")


def run():
    rng = np.random.default_rng(20261122)
    _risk_vs_moments(rng)
    _gamma_floor(rng)
    _x10(rng)
    _x8(rng)
    _x1e(rng)
    _gamma(rng)
    _x11()
