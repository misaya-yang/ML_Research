"""X9b/X9c: Gamma-certificate issuance versus Var/(k Gamma_t) and its target cost (P12)."""

import math
import time

import numpy as np

from experiments.audit_lab.bench import Bank
from experiments.audit_lab.imh import Cells
from .gamma_cert import A_gamma
from .instances import lognormal, pareto_low
from .risk import cold_risks, dense_kernel, envelopes, p_quantile_threshold


def build(spec):
    if spec["family"] == "lognormal":
        return lognormal(spec["sigma"])
    return pareto_low(spec["eta0"], spec["r"])


def thresholds(w, P, design):
    hs, names = [], []
    for u in design["upper"]:          # h=1{W>=v}, P(h=1) about 1-u
        v = p_quantile_threshold(w, P, u)
        hs.append((w >= v).astype(float)); names.append(f"W>=q{u}")
    for u in design["lower"]:          # h=1{W<=v}, P(h=1) about u
        v = p_quantile_threshold(w, P, u)
        hs.append((w <= v).astype(float)); names.append(f"W<=q{u}")
    keep = [i for i, h in enumerate(hs) if 0 < P @ h < 1]
    return np.column_stack([hs[i] for i in keep]), [names[i] for i in keep]


def task_fn(task):
    start = time.process_time()
    cfg = task["cfg"]
    rng = np.random.default_rng(task["seed"])
    w, q = build(task["spec"])
    P = q * w
    t, delta, reps = task["t"], cfg["delta"], cfg["reps"]
    H, names = thresholds(w, P, cfg["h_design"])
    K = dense_kernel(w, q)
    R, mu, nu = cold_risks(w, q, H, 0, t, K=K)
    gamma, xi, eta = envelopes(w, q, 0, t)
    rows, target, proposal = [], 0, 0
    for j, name in enumerate(names):
        h = H[:, j]
        var = float(nu[j] - mu[j]**2)
        for k in cfg["k_grid"]:
            G = float(R[j] - var / k)
            meta = dict(family=task["spec"]["name"], t=t, k=k, h=name, mu=float(mu[j]), var=var,
                        Gamma=gamma, R=float(R[j]), G=G, x=var / (k * gamma), unsafe=G >= 0, eta=eta)
            compare = k == cfg["compare"]["k"] and t in cfg["compare"]["t_grid"]
            bank = Bank(Cells(w, q, h), 0, t, k) if compare else None
            for p in cfg["p_grid"]:
                for qq in cfg["q_grid"]:
                    issued, diag = A_gamma(rng, w, h, P, q, p, qq, t, k, delta, reps,
                                           C0s=(cfg["C0"], cfg["C0_explore"]))
                    row = dict(meta, auditor="A_gamma", p=p, q=qq, reps=reps,
                               rate=float(issued[cfg["C0"]].mean()),
                               rate_C0_explore=float(issued[cfg["C0_explore"]].mean()),
                               issues=int(np.count_nonzero(rng.random(reps) < issued[cfg["C0"]])),
                               **{k2: v for k2, v in diag.items() if k2 != "reason"})
                    rows.append(row)
                    target += p * reps
                    proposal += qq * reps
                    if compare:
                        values = bank.sample(rng, "A_clip", p, qq, reps, delta,
                                             normal_threshold=None, c0=16)
                        actual_q = qq // 2 + ((qq - qq // 2) // t) * t
                        rows.append(dict(meta, auditor="A_clip_paths", p=p, q=qq, q_actual=actual_q,
                                         reps=reps, rate=float(values.mean()),
                                         issues=int(np.count_nonzero(rng.random(reps) < values))))
                        target += p * reps
                        proposal += actual_q * reps
    return dict(rows=rows, target_samples=target, proposal_samples=proposal,
                cpu_seconds=time.process_time() - start)
