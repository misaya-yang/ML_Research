"""X1e: grid-intersection clipped certificate at small overlap, and saturation (E1, E1'; P9)."""

import math
import time

import numpy as np
from scipy.optimize import brentq

from experiments.audit_lab.auditors import A_clip, _gate, empirical_bernstein
from experiments.audit_lab.counts import multinomial, path_tallies
from experiments.audit_lab.families import ThreeLayers, calibrated_pair, layered, tail_profile
from experiments.audit_lab.imh import Cells, cell_gap, window_sum_pmf


def _quantile(score, counts, u):
    order = np.argsort(score, kind="stable")
    n = int(counts[0].sum())
    rank = max(1, math.ceil(u * n))
    index = (np.cumsum(counts[:, order], axis=1) >= rank).argmax(axis=1)
    return score[order[index]]


def clip_interval(score, h, p_counts, q_counts, H, ell):
    """Pro's clipped center interval at clip level H (per replicate), as in A_clip."""
    head = np.minimum(1, score[None, :] / H[:, None])
    tail = np.maximum(0, 1 - H[:, None] / score[None, :])
    u1_lo, u1_hi = empirical_bernstein(q_counts, head * h, ell)
    u0_lo, u0_hi = empirical_bernstein(q_counts, head * (1 - h), ell)
    c1_lo, c1_hi = empirical_bernstein(p_counts, tail * h, ell)
    c0_lo, c0_hi = empirical_bernstein(p_counts, tail * (1 - h), ell)
    lo_den, hi_den = u1_lo + u0_hi, u1_hi + u0_lo
    lo = np.divide(u1_lo, lo_den, out=np.zeros_like(u1_lo), where=lo_den > 0)
    hi = np.divide(u1_hi, hi_den, out=np.ones_like(u1_hi), where=hi_den > 0)
    return (np.clip(lo + (1 - lo) * c1_lo - lo * c0_hi, 0, 1),
            np.clip(hi + (1 - hi) * c1_hi - hi * c0_lo, 0, 1))


def A_clip_grid(score, h, anchor, p_counts, q_counts, paths, delta, k, ratio):
    """Anchor a = empirical 1/4-quantile of S over an independent target bank.

    Levels H_j = a * ratio * 2^-j, j=-3..ceil(log2 ratio)+1, ratio=q/p. Each level's
    interval has error delta/(4J) (eight one-sided Bernstein bounds at delta/(32J)).
    Intervals are intersected; the unchanged path test uses delta/2.
    """
    reps = len(p_counts)
    if int(p_counts[0].sum()) < 2 or int(q_counts[0].sum()) < 2 or paths.n == 0:
        return np.zeros(reps)
    a = _quantile(score, anchor, .25)
    js = range(-3, math.ceil(math.log2(ratio)) + 2)
    J = len(js)
    ell = math.log(32 * J / delta)
    mu_lo, mu_hi = np.zeros(reps), np.ones(reps)
    for j in js:
        lo, hi = clip_interval(score, h, p_counts, q_counts, a * ratio * 2.0**(-j), ell)
        mu_lo, mu_hi = np.maximum(mu_lo, lo), np.minimum(mu_hi, hi)
    return _gate(mu_lo, mu_hi, paths, delta / 2, k)


class DarkThreeLayers:
    """Theorem A three-layer family mixed with a dark layer of Q-mass 1-eta0.

    Dark raw score = 1e-6 * the family's minimum raw score; the dark layer is split
    evenly into h=0 and h=1 so Q(h=1)=1/2 is unchanged. Original Q-masses are scaled by eta0.
    eta0=1 returns the original family.
    """

    def __init__(self, base, eta0):
        self.base, self.eta0 = base, eta0

    def cells(self, beta, theta):
        cells = self.base.cells(beta, theta)
        if self.eta0 == 1:
            return cells
        dark = 1e-6 * cells.w.min()
        rest = (1 - self.eta0) / 2
        return Cells(np.r_[cells.w, dark, dark], np.r_[cells.q * self.eta0, rest, rest],
                     np.r_[cells.h, 0., 1.])

    @property
    def m(self):
        return self.base.m


def _sample_and_audit(rng, cells, pmf, b, t, k, p, q, reps, delta, ratio, n_anchor, normal_threshold, c0):
    """Shared banks: A_clip gets all p target rows and the pilot/rest split of the Q center
    half; A_clip_grid gets the same Q center rows (pilot+rest) and splits its p rows into
    an anchor bank (n_anchor) and a certificate bank (p - n_anchor)."""
    n_center = q // 2
    paths = path_tallies(rng, cells, b, t, q - n_center, reps, pmf=pmf, normal_threshold=normal_threshold)
    pilot_n = n_center // 8

    def draw(n, probs):
        return multinomial(rng, n, probs, reps, normal_threshold=normal_threshold)

    anchor = draw(n_anchor, cells.p)
    cert = draw(p - n_anchor, cells.p)
    pilot = draw(pilot_n, cells.q)
    rest = draw(n_center - pilot_n, cells.q)
    clip = A_clip(cells.w, cells.h, anchor + cert, pilot, rest, paths, delta, k, c0)
    grid = A_clip_grid(cells.w, cells.h, anchor, cert, pilot + rest, paths, delta, k, ratio)
    actual_q = n_center + paths.n * (b + t)
    return clip, grid, actual_q


def _estimate(values):
    if np.all(values == values[0]):
        return float(values[0]), 0.
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def lowoverlap_task(task):
    """One design point (profile r, rho, eta0): calibrate the pair, then scan c for both auditors."""
    start = time.process_time()
    cfg = task["cfg"]
    rng = np.random.default_rng(task["seed"])
    r, rho, eta0 = task["r"], task["rho"], task["eta0"]
    t, k, delta = cfg["t"], cfg["k"], cfg["delta"]
    b = 0 if eta0 == 1 else math.ceil(cfg["burn_in_over_eta0"] / eta0)
    psi = tail_profile(rho, cfg["J"], r)
    eps = psi / cfg["eps_divisor"]
    meta = dict(r=r, rho=rho, eta0=eta0, b=b, t=t, psi=psi, epsilon=eps)
    family = DarkThreeLayers(ThreeLayers(cfg["J"], rho, psi / cfg["mass_divisor"], cfg["bulk_mass"]), eta0)
    try:
        worlds = calibrated_pair(family, b, t, k, eps, max_tilt=cfg["max_tilt"],
                                 tilt_tolerance=cfg["tilt_tolerance"])
    except ValueError as error:
        return dict(rows=[dict(meta, kind="calibration_failed", error=str(error))],
                    cpu_seconds=time.process_time() - start)
    rows = [dict(meta, kind="calibration", beta=worlds.beta, s=worlds.s, G_safe=worlds.g_safe,
                 G_unsafe=worlds.g_unsafe)]
    pmf_safe = window_sum_pmf(worlds.safe, b, t)
    pmf_unsafe = window_sum_pmf(worlds.unsafe, b, t)
    base_p = psi * math.log(1 / delta) / eps**2
    reached = {"A_clip": None, "A_clip_grid": None}
    target = proposal = 0
    for c in cfg["c_grid"]:
        if all(reached.values()):
            break
        p = math.ceil(c * base_p)
        q = rho * p
        if q > cfg["q_cap"]:
            rows.append(dict(meta, kind="skipped_q_cap", c=c, p=p, q=q))
            continue
        clip_s, grid_s, actual_q = _sample_and_audit(rng, worlds.safe, pmf_safe, b, t, k, p, q, cfg["reps"],
                                                     delta, rho, cfg["n_anchor"], cfg["normal_threshold"], cfg["c0"])
        clip_u, grid_u, _ = _sample_and_audit(rng, worlds.unsafe, pmf_unsafe, b, t, k, p, q, cfg["reps"],
                                              delta, rho, cfg["n_anchor"], cfg["normal_threshold"], cfg["c0"])
        target += 2 * p * cfg["reps"]
        proposal += 2 * actual_q * cfg["reps"]
        for name, safe_v, unsafe_v in (("A_clip", clip_s, clip_u), ("A_clip_grid", grid_s, grid_u)):
            power, se = _estimate(safe_v)
            issues = int(np.count_nonzero(rng.random(cfg["reps"]) < unsafe_v))
            rows.append(dict(meta, kind="probe", auditor=name, c=c, p=p, q=q, q_actual=actual_q,
                             power=power, se=se, unsafe_mean=float(unsafe_v.mean()),
                             unsafe_issues=issues, reps=cfg["reps"]))
            if reached[name] is None and power >= cfg["power_target"]:
                reached[name] = c
    for name, c in reached.items():
        rows.append(dict(meta, kind="minimum", auditor=name, c_star=c,
                         status="reached" if c else "not_reached"))
    return dict(rows=rows, target_samples=target, proposal_samples=proposal,
                cpu_seconds=time.process_time() - start)


def _beta_for_gap(family, b, t, k, gap_target, lo, hi):
    return brentq(lambda beta: cell_gap(family.cells(beta, .5), b, t, k) - gap_target, lo, hi, xtol=1e-14)


def saturation_task(task):
    """Fixed epsilon; safe base (G=-eps) and unsafe base (G=+eps) at theta=1/2 by tuning beta."""
    start = time.process_time()
    cfg = task["cfg"]
    rng = np.random.default_rng(task["seed"])
    rho = task["rho"]
    t, k, delta, eps = cfg["t"], cfg["k"], cfg["delta"], cfg["epsilon"]
    psi = tail_profile(rho, cfg["J"], cfg["r"])
    meta = dict(rho=rho, psi=psi, epsilon=eps, psi_over_eps=psi / eps)
    family = ThreeLayers(cfg["J"], rho, psi / cfg["mass_divisor"], cfg["bulk_mass"])

    def gap(beta):
        return cell_gap(family.cells(beta, .5), 0, t, k)

    try:
        root = brentq(gap, .5, 1., xtol=1e-14)
        beta_safe = _beta_for_gap(family, 0, t, k, -eps, .5, root)
        beta_unsafe = _beta_for_gap(family, 0, t, k, eps, root, 1.)
    except ValueError as error:
        return dict(rows=[dict(meta, kind="calibration_failed", error=str(error))],
                    cpu_seconds=time.process_time() - start)
    safe, unsafe = family.cells(beta_safe, .5), family.cells(beta_unsafe, .5)
    rows = [dict(meta, kind="calibration", beta_safe=beta_safe, beta_unsafe=beta_unsafe,
                 G_safe=cell_gap(safe, 0, t, k), G_unsafe=cell_gap(unsafe, 0, t, k))]
    pmf_safe, pmf_unsafe = window_sum_pmf(safe, 0, t), window_sum_pmf(unsafe, 0, t)
    unit = math.log(1 / delta) / eps
    found = None
    target = proposal = 0
    for c in cfg["c_grid"]:
        p = math.ceil(c * unit)
        q = rho * p
        if q > cfg["q_cap"]:
            rows.append(dict(meta, kind="skipped_q_cap", c=c, p=p, q=q))
            continue
        _, grid_s, actual_q = _sample_and_audit(rng, safe, pmf_safe, 0, t, k, p, q, cfg["reps"], delta, rho,
                                                cfg["n_anchor"], cfg["normal_threshold"], cfg["c0"])
        _, grid_u, _ = _sample_and_audit(rng, unsafe, pmf_unsafe, 0, t, k, p, q, cfg["reps"], delta, rho,
                                         cfg["n_anchor"], cfg["normal_threshold"], cfg["c0"])
        target += 2 * p * cfg["reps"]
        proposal += 2 * actual_q * cfg["reps"]
        power, se = _estimate(grid_s)
        rows.append(dict(meta, kind="probe", auditor="A_clip_grid", c=c, p=p, q=q, power=power, se=se,
                         unsafe_mean=float(grid_u.mean()),
                         unsafe_issues=int(np.count_nonzero(rng.random(cfg["reps"]) < grid_u)),
                         reps=cfg["reps"]))
        if power >= cfg["power_target"]:
            found = (c, p)
            break
    rows.append(dict(meta, kind="minimum", status="reached" if found else "not_reached",
                     c_star=found[0] if found else None, p_min=found[1] if found else None,
                     p_eps_over_log=(found[1] * eps / math.log(1 / delta)) if found else None,
                     A_psi=psi * math.log(1 / delta) / eps**2))
    return dict(rows=rows, target_samples=target, proposal_samples=proposal,
                cpu_seconds=time.process_time() - start)
