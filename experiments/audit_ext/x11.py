"""X11: exact check of the small-overlap floor (Lemma F, Theorem V_eta, Proposition N, Theorem S; P14).

Reuses experiments/finite_state/small_overlap_floor.py unchanged (imported by path).
"""

import importlib.util
import math
import time

import numpy as np

from experiments.audit_lab.counts import LR_oracle
from experiments.audit_lab.run_io import ROOT
from .risk import cold_risks

_spec = importlib.util.spec_from_file_location(
    "small_overlap_floor", ROOT / "experiments" / "finite_state" / "small_overlap_floor.py")
sof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sof)


def _cells(m, th):
    w2 = np.repeat(m["w"], 2)
    q2 = np.ravel(np.c_[m["q"] * th, m["q"] * (1 - th)])
    h2 = np.tile([1., 0.], len(th))
    return w2, q2, h2


def chain_R(m, th, t):
    w2, q2, h2 = _cells(m, th)
    keep = q2 > 0
    R, _, _ = cold_risks(w2[keep], q2[keep], h2[keep][:, None], 0, t)
    return float(R[0])


def _tilted(m, ref, j, D):
    lo, hi = ref.copy(), ref.copy()
    lo[j] -= D
    hi[j] += D
    if j >= 3:
        lo[4], hi[4] = 1 - lo[3], 1 - hi[3]
    lo, hi = sof.compensate(m, lo, ref), sof.compensate(m, hi, ref)
    inside = all(v.min() >= 0 and v.max() <= 1 and 0 < v[0] < 1 and 0 < v[2] < 1 for v in (lo, hi))
    return lo, hi, inside


def _cell_law(m, th, which):
    mass = m[which]
    return np.r_[mass * th, mass * (1 - th)]


def x11_task(task):
    start = time.process_time()
    cfg = task["cfg"]
    c1, c2 = task["c"]
    t, rho, fr = task["t"], task["rho"], task["fr"]
    rB = fr * rho
    d = rB * t
    Delta = cfg["tilt"]
    base_meta = dict(c1=c1, c2=c2, t=t, rho=rho, rB_over_rho=fr, rB=rB, d=d)
    rows = []
    m = None
    for k in cfg["k_grid"]:
        meta = dict(base_meta, k=k)
        Pbar0 = (sof.Phi0(c1) + sof.Phi0(c2)) / 2
        if k * Pbar0 <= 1:
            continue                                   # combination excluded by k*Phibar0>1 (not a row)
        if not (2 * k * rB < 1 and d * rho >= 5):
            rows.append(dict(meta, status="n/a", reason="2k rho_B<1 and d rho>=5 required"))
            continue
        if m is None:
            w, q = sof.base(t, d, rho, (float(c1), float(c2)))
            m = sof.moments(w, q, t)
            eta = float(q @ np.minimum(1, w))
            Gam = float(m["p"] @ np.minimum(w, t) / t)
        refS = sof.calibrate(m, np.array([.5, .5, 0., 1., 1.]), k)
        refV = sof.calibrate(m, np.array([.5, .5, 0., .5, .5]), k)
        if refS is None or refV is None:
            rows.append(dict(meta, status="n/a", reason="no bulk label gives G=0", eta=eta, Gamma=Gam))
            continue
        loS, hiS, inS = _tilted(m, refS, 1, Delta)
        loV, hiV, inV = _tilted(m, refV, 3, Delta)
        if not (inS and inV):
            rows.append(dict(meta, status="tilt_infeasible", S_inside=inS, V_inside=inV, eta=eta, Gamma=Gam))
            continue
        gm, gp = sof.gate(m, loS, k), sof.gate(m, hiS, k)
        kp, kq = sof.kl(m, loS, hiS, "p"), sof.kl(m, loS, hiS, "q")
        a = (gp - gm) / (2 * Delta)
        eps = min(abs(gm), abs(gp))
        sB = refS[2]
        A1 = sum(.5 * (1 - sof.phi0(c)) for c in (c1, c2))
        pred = rB * ((.25 + .5 + .5 - 2.5 * sB) * rB - (1 - sB) * A1 * rho) + 2 * rho * Pbar0 * (1 - sB) / d
        vm, vp = sof.gate(m, loV, k), sof.gate(m, hiV, k)
        vkp, vkq = sof.kl(m, loV, hiV, "p"), sof.kl(m, loV, hiV, "q")
        epsV = min(abs(vm), abs(vp))
        K = (vkp / epsV**2) / (kp / eps**2)
        inv = 0.
        for v in (loS, hiS, loV, hiV):
            ref = refS if v is loS or v is hiS else refV
            inv = max(inv, abs(m["p"] @ (v - ref)), abs(m["q"] @ (v - ref)))
        # quadratic form vs explicit 10-cell chain at the S pair and its reference
        rel = 0.
        for th in (refS, loS, hiS):
            Rq, Rc = sof.R(m, th), chain_R(m, th, t)
            rel = max(rel, abs(Rq - Rc) / abs(Rc))
        # Proposition N ingredients, as in small_overlap_floor.py
        rng = np.random.default_rng(int(t + 1000 * rho + fr * 10 + 100000 * k))
        worst, used = 0.0, 0
        for _ in range(cfg["cs_pairs"]):
            x0 = rng.uniform(.05, .95, 5)
            x1 = x0 + rng.normal(0, .1, 5)
            x1[[0, 2]] = x0[[0, 2]]
            x1 = sof.compensate(m, np.clip(x1, .01, .99), x0)
            if x1.min() <= 0 or x1.max() >= 1:
                continue
            used += 1
            D = x1 - x0
            worst = max(worst, abs(sof.R(m, x1) - sof.R(m, x0)) / (18.17 * np.sqrt(Gam * np.sum(D**2 * m["M1"]))))
        lam_d = 1 - q @ np.minimum(1, w / d)
        e_low = (1 - lam_d**t) / (t * (1 - lam_d))
        low = w < d
        exc = float(np.sum(np.maximum(m["M1"][low] - 2 * m["p"][low], 0)))
        rows.append(dict(meta, status="ok", eta=eta, Gamma=Gam, sB=float(sB),
                         a_over_pred=float(a / pred), S_Gm_over_eps=float(gm / eps), S_Gp_over_eps=float(gp / eps),
                         eps_S=float(eps), KL_P_d_over_D2=float(kp * d / Delta**2), KL_Q_d_over_D2=float(kq * d / Delta**2),
                         KL_P_over_eps2=float(kp / eps**2),
                         V_Gm_over_eps=float(vm / epsV), V_Gp_over_eps=float(vp / epsV), eps_V=float(epsV),
                         V_KL_P_over_eps2=float(vkp / epsV**2), K=float(K), inv=float(inv), R_rel_diff=float(rel),
                         cs_ratio=float(worst), cs_pairs_used=used, excess_over_elow=float(exc / e_low),
                         soft_row=bool(fr >= .5)))
        if task.get("x11b") and fr in task["x11b"]["fr_grid"] and t in task["x11b"]["t_grid"] and k == task["x11b"]["k"] \
                and (c1, c2) == tuple(task["x11b"]["c"]) and rho == task["x11b"]["rho"]:
            rows[-1].update(_x11b(task, m, loS, hiS, loV, hiV, eps, epsV, K))
    return dict(rows=rows, cpu_seconds=time.process_time() - start)


def _min_p(rng, unsafe, safe, reps, delta, cap):
    target = 1 - delta
    probes = []

    def power(n):
        roc = LR_oracle(unsafe, safe, unsafe, safe, n, 0, delta, reps, rng=rng)
        probes.append(n)
        return roc.power

    low, high = 0, 1
    while power(high) < target:
        low, high = high, 2 * high
        if high > cap:
            return None, probes
    while high - low > 1:
        mid = (low + high) // 2
        if power(mid) >= target:
            high = mid
        else:
            low = mid
    return high, probes


def _x11b(task, m, loS, hiS, loV, hiV, eps, epsV, K):
    cfg = task["x11b"]
    rng = np.random.default_rng(task["seed"])
    out = {}
    total = 0
    for name, lo, hi, e in (("S", loS, hiS, eps), ("V", loV, hiV, epsV)):
        # which side is safe does not matter for a symmetric two-sided requirement; use lo as the null
        n, probes = _min_p(rng, _cell_law(m, lo, "p"), _cell_law(m, hi, "p"), cfg["reps"], cfg["delta"], cfg["cap"])
        out[f"x11b_n_{name}"] = n
        out[f"x11b_n_eps2_{name}"] = n * e**2 if n else None
        total += 2 * cfg["reps"] * sum(probes)
    if out["x11b_n_S"] and out["x11b_n_V"]:
        out["x11b_ratio"] = out["x11b_n_eps2_S"] / out["x11b_n_eps2_V"]  # plan: n_S eps_S^2 : n_V eps_V^2 (fixed after X11-20261122-3565)
        out["x11b_ratio_over_K"] = out["x11b_ratio"] / K
    out["x11b_target_samples"] = total
    return out
