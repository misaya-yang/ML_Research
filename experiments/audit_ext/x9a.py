"""X9a: exact check of R_Q(b,t) <= 44 Gamma_t and <= 1520 Xi_{b,t} (Lemma R, R'; P11)."""

import time

import numpy as np

from experiments.audit_lab.families import ThreeLayers, tail_profile
from .instances import pareto_low, random_instance, _normalize
from .risk import cold_risks, dense_kernel, envelopes


def three_layer(rho, r, beta=0.8, J=2048, bulk=.75, mass_divisor=12, dark_eta0=None):
    psi = tail_profile(rho, J, r)
    cells = ThreeLayers(J, rho, psi / mass_divisor, bulk).cells(beta, .5)
    w, q, h = cells.w, cells.q, cells.h
    if dark_eta0 is not None:
        q = q * dark_eta0
        dark = 1e-6 * w.min()
        w = np.r_[w, dark, dark]
        q = np.r_[q, (1 - dark_eta0) / 2, (1 - dark_eta0) / 2]
        h = np.r_[h, 0., 1.]
    w, q = _normalize(w, q, floor=0)
    return w, q, h


def build(spec):
    rng = np.random.default_rng(spec["seed"])
    family = spec["family"]
    extra = []
    if family == "random":
        n = int(rng.integers(20, 121))
        w, q = random_instance(rng, spec["kind"], n)
    elif family == "three_layer":
        w, q, h = three_layer(spec["rho"], spec["r"], dark_eta0=spec.get("dark_eta0"))
        extra.append(h)
    elif family == "pareto_low":
        w, q = pareto_low(spec["eta0"], spec["r"])
    else:
        raise ValueError(family)
    n = len(w)
    ws = np.sort(np.unique(w))
    grid = ws[::max(1, len(ws) // 12)]
    hs = [(w >= v).astype(float) for v in grid] + [(w <= v).astype(float) for v in grid]
    hs += [rng.integers(0, 2, n).astype(float) for _ in range(3)]
    hs += [rng.random(n) for _ in range(2)]
    hs += extra
    H = np.column_stack([h for h in hs if h.min() != h.max()])
    return w, q, H


def instance_task(task):
    start = time.process_time()
    w, q, H = build(task["spec"])
    K = dense_kernel(w, q)
    rows = []
    for t in task["t_grid"]:
        for b_rule in task["b_rules"]:
            b = {"0": 0, "t/4": t // 4, "4t": 4 * t}[b_rule]
            R, _, _ = cold_risks(w, q, H, b, t, K=K)
            gamma, xi, eta = envelopes(w, q, b, t)
            best = int(np.argmax(R))
            rows.append(dict(t=t, b=b, b_rule=b_rule, n_states=len(w), n_h=H.shape[1],
                             max_R=float(R.max()), max_R_over_Gamma=float(R.max() / gamma),
                             max_R_over_Xi=float(R.max() / xi), Gamma=gamma, Xi=xi,
                             Gamma_over_Xi=gamma / xi, eta=eta, argmax_h=best))
    return dict(rows=rows, cpu_seconds=time.process_time() - start)
