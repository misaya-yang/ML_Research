"""X10: variance-scaled light-window floor family (Theorem V, P13)."""

import math
import time

import numpy as np

from experiments.audit_lab.counts import LR_oracle

W_D, S_D0 = 0.1, 0.1


def Phi(c, beta):
    return 2 * (c - 1 + math.exp(-c)) / c**2 - math.exp(-c * beta) * (2 / c**2) * (1 - (1 + c) * math.exp(-c))


def sigma_K(k, beta, c1, c2):
    return (k * (Phi(c1, beta) + Phi(c2, beta)) / 2 - 1) / 2 - S_D0 * W_D / (1 - W_D)


def instance(theta, rho, s0, c1, c2, t):
    """Same construction as experiments/finite_state/variance_floor.py; theta may be an array."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    q1, q2 = rho * c1 / (2 * t), rho * c2 / (2 * t)
    qD = (rho - q1 - q2) / (1 - W_D)
    qB = 1 - q1 - q2 - qD
    pB, pD = qB, qD * W_D
    a = rho / 2
    YP = s0 * pB + S_D0 * pD
    YQ = s0 * qB + S_D0 * qD - a * (theta - .5) * (c1 - c2) / t
    A = np.array([[qB, qD], [pB, pD]])
    sol = np.linalg.solve(A, np.vstack([YQ, np.full_like(YQ, YP)]))
    sB, sD = sol
    w = np.array([1., 1., W_D, W_D, t / c1, t / c1, t / c2, t / c2])
    wm = w[4]
    q = np.column_stack([(1 - sB) * qB, sB * qB, (1 - sD) * qD, sD * qD,
                         theta * a / w[4], (rho / 2 - theta * a) / w[4],
                         (1 - theta) * a / w[6], (rho / 2 - (1 - theta) * a) / w[6]])
    h = np.array([0, 1, 0, 1, 1, 0, 1, 0.])
    feasible = (q.min(axis=1) > 0) & (np.abs(q.sum(axis=1) - 1) < 1e-12)
    return w, q, h, feasible


def gate(theta, rho, s0, c1, c2, t, b, k):
    """Exact G_k and statistics for each theta (batched 8-state chain)."""
    w, q, h, feasible = instance(theta, rho, s0, c1, c2, t)
    n = len(w)
    K = q[:, None, :] * np.minimum(1, w[None, :] / w[:, None])[None]
    idx = np.arange(n)
    K[:, idx, idx] += 1 - K.sum(axis=2)
    P = q * w
    P = P / P.sum(axis=1, keepdims=True)
    mu = P @ h
    f = h[None, :] - mu[:, None]
    nu = q.copy()
    if b:
        Kb = np.linalg.matrix_power(K, b)
        nu = np.einsum("ai,aij->aj", nu, Kb)
    g1 = np.zeros_like(f)
    g2 = np.zeros_like(f)
    for _ in range(t):
        Kg1 = np.einsum("aij,aj->ai", K, g1)
        g2 = f * f + 2 * f * Kg1 + np.einsum("aij,aj->ai", K, g2)
        g1 = f + Kg1
    R = np.einsum("ai,ai->a", nu, g2) / t**2
    V = mu * (1 - mu)
    Gam = (P * np.minimum(w, t)).sum(axis=1) / t
    eta = (q * np.minimum(1, w)).sum(axis=1)
    return dict(G=R - V / k, R=R, V=V, Gam=Gam, eta=eta, mu=mu, Qh=q @ h, P=P, q=q, w=w,
                feasible=feasible)


def _level_law(w, mass):
    levels, inverse = np.unique(w, return_inverse=True)
    return np.bincount(inverse, mass)


def kl(a, b):
    return float(np.sum(a * np.log(a / b)))


def calibrate(k, beta, c1, c2, t, rho, eps, fit_thetas):
    """theta* from an exact quadratic fit refined by Newton steps on exact values."""
    P1, P2 = Phi(c1, beta), Phi(c2, beta)
    sigma = sigma_K(k, beta, c1, c2)
    b = int(round(beta * t))
    s0 = sigma * rho
    thetas = np.asarray(fit_thetas, dtype=float)
    g = gate(thetas, rho, s0, c1, c2, t, b, k)
    if not g["feasible"].all():
        return dict(status="infeasible_fit_points")
    coef = np.polyfit(thetas - .5, g["G"], 2)
    resid = float(np.max(np.abs(np.polyval(coef, thetas - .5) - g["G"])))
    scale = float(np.max(np.abs(g["G"])))
    roots = np.roots(coef)
    roots = roots[np.isreal(roots)].real + .5
    if not roots.size:
        return dict(status="no_root", resid=resid, scale=scale)
    th = float(roots[np.argmin(np.abs(roots - .5))])
    for _ in range(3):
        hstep = 1e-4
        gg = gate([th, th - hstep, th + hstep], rho, s0, c1, c2, t, b, k)["G"]
        slope = (gg[2] - gg[1]) / (2 * hstep)
        th -= gg[0] / slope
    return dict(status="ok", theta=th, resid=resid, scale=scale, coef=coef.tolist(),
                b=b, s0=s0, sigma=sigma, P1=P1, P2=P2)


def x10a_task(task):
    start = time.process_time()
    k, beta, c1, c2, t, rho = (task[key] for key in ("k", "beta", "c1", "c2", "t", "rho"))
    cfg = task["cfg"]
    P1, P2 = Phi(c1, beta), Phi(c2, beta)
    dPhi = P1 - P2
    eps = rho * abs(dPhi) / cfg["eps_den"]
    Delta = cfg["tilt"]
    meta = dict(k=k, beta=beta, c1=c1, c2=c2, t=t, rho=rho, eps=eps, Delta=Delta,
                sigma=sigma_K(k, beta, c1, c2), Phi1=P1, Phi2=P2)
    cal = calibrate(k, beta, c1, c2, t, rho, eps, cfg["fit_thetas"])
    if cal["status"] != "ok":
        return dict(row=dict(meta, status=cal["status"], resid=cal.get("resid"), scale=cal.get("scale")),
                    cpu_seconds=time.process_time() - start)
    th, sg = cal["theta"], np.sign(dPhi)
    gg = gate([th - sg * Delta, th + sg * Delta, th], rho, cal["s0"], c1, c2, t, cal["b"], k)
    if not gg["feasible"].all():
        return dict(row=dict(meta, status="infeasible_worlds", theta=th),
                    cpu_seconds=time.process_time() - start)
    Pm, Pp, Qm, Qp = gg["P"][0], gg["P"][1], gg["q"][0], gg["q"][1]
    w = gg["w"]
    lawdiff = max(np.abs(_level_law(w, Pm) - _level_law(w, Pp)).max(),
                  np.abs(_level_law(w, Qm) - _level_law(w, Qp)).max())
    klP = max(kl(Pp, Pm), kl(Pm, Pp))
    klQ = max(kl(Qp, Qm), kl(Qm, Qp))
    row = dict(meta, status="ok", theta=th, b=cal["b"],
               G_safe_over_eps=float(gg["G"][0] / eps), G_unsafe_over_eps=float(gg["G"][1] / eps),
               G_center=float(gg["G"][2]),
               quad_resid=cal["resid"], quad_scale=cal["scale"], quad_rel=cal["resid"] / cal["scale"],
               R_over_rho=float(gg["R"][2] / rho), Phibar_half=(P1 + P2) / 4,
               var_over_kR=float(gg["V"][2] / (k * gg["R"][2])), Gamma_over_rho=float(gg["Gam"][2] / rho),
               eta=float(gg["eta"][2]),
               inv_W_law=float(lawdiff), inv_Ph=float(abs(gg["mu"][0] - gg["mu"][1])),
               inv_Qh=float(abs(gg["Qh"][0] - gg["Qh"][1])),
               KL_P_const=klP * rho * dPhi**2 / eps**2, KL_Q_const=klQ * rho * t * dPhi**2 / eps**2,
               KL_P=klP, KL_Q=klQ)
    return dict(row=row, cpu_seconds=time.process_time() - start)


def _min_budget(rng, unsafe_P, safe_P, unsafe_Q, safe_Q, which, target, reps, delta, cap):
    probes = []

    def power(n):
        p, q = (n, 0) if which == "P" else (0, n)
        roc = LR_oracle(unsafe_P, safe_P, unsafe_Q, safe_Q, p, q, delta, reps, rng=rng)
        probes.append((n, roc.power, roc.null_calibration))
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


def x10b_task(task):
    start = time.process_time()
    cfg = task["cfg"]
    k, beta, c1, c2, t, rho = (task[key] for key in ("k", "beta", "c1", "c2", "t", "rho"))
    P1, P2 = Phi(c1, beta), Phi(c2, beta)
    dPhi = P1 - P2
    eps = cfg["rho_min"] * abs(dPhi) / cfg["eps_den"]
    Delta = 4 * eps / (rho * abs(dPhi))
    cal = calibrate(k, beta, c1, c2, t, rho, eps, cfg["fit_thetas"])
    meta = dict(k=k, beta=beta, c1=c1, c2=c2, t=t, rho=rho, eps=eps, Delta=Delta)
    if cal["status"] != "ok":
        return dict(row=dict(meta, status=cal["status"]), cpu_seconds=time.process_time() - start)
    th, sg = cal["theta"], np.sign(dPhi)
    gg = gate([th - sg * Delta, th + sg * Delta], rho, cal["s0"], c1, c2, t, cal["b"], k)
    safe_P, unsafe_P, safe_Q, unsafe_Q = gg["P"][0], gg["P"][1], gg["q"][0], gg["q"][1]
    rng = np.random.default_rng(task["seed"])
    target = 1 - cfg["delta"]
    p_min, p_probes = _min_budget(rng, unsafe_P, safe_P, unsafe_Q, safe_Q, "P", target,
                                  cfg["reps"], cfg["delta"], cfg["cap"])
    q_min, q_probes = _min_budget(rng, unsafe_P, safe_P, unsafe_Q, safe_Q, "Q", target,
                                  cfg["reps"], cfg["delta"], cfg["cap"])
    row = dict(meta, status="ok", theta=th, G_safe_over_eps=float(gg["G"][0] / eps),
               G_unsafe_over_eps=float(gg["G"][1] / eps), p_min=p_min, q_min=q_min,
               p_probes=len(p_probes), q_probes=len(q_probes))
    target_samples = 2 * cfg["reps"] * sum(n for n, _, _ in p_probes)
    proposal_samples = 2 * cfg["reps"] * sum(n for n, _, _ in q_probes)
    return dict(row=row, target_samples=target_samples, proposal_samples=proposal_samples,
                cpu_seconds=time.process_time() - start)
