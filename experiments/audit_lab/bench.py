"""Shared simulation and reporting operations for the local benchmarks."""

from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, t as student_t

from .auditors import A_clip, A_ep, A_knownZ
from .counts import LR_oracle, multinomial, path_tallies
from .imh import window_sum_pmf


@dataclass(frozen=True)
class Estimate:
    power: float
    se: float


class Bank:
    """Simulate independent half-banks before calling an auditor."""

    def __init__(self, cells, b, t, k):
        self.cells, self.b, self.t, self.k = cells, b, t, k
        self.pmf = window_sum_pmf(cells, b, t)
        self.p_prob = cells.p

    def sample(self, rng, method, p, q, reps, delta, *, normal_threshold=None, c0=16):
        cells = self.cells
        n_center = q // 2

        def draw(n, probs):
            return multinomial(rng, n, probs, reps, normal_threshold=normal_threshold)

        paths = path_tallies(rng, cells, self.b, self.t, q - n_center, reps,
                            pmf=self.pmf, normal_threshold=normal_threshold)
        if method == "A_clip":
            pilot = n_center // 8
            return A_clip(cells.w, cells.h, draw(p, self.p_prob),
                          draw(pilot, cells.q), draw(n_center - pilot, cells.q), paths,
                          delta, self.k, c0)
        if method == "A_ep":
            return A_ep(cells.w, cells.h, draw(p, self.p_prob), draw(n_center, cells.q),
                        paths, delta, self.k, 1 / q)
        if method == "A_knownZ" and p == 0:
            z = float(np.sum(cells.q * cells.w))
            return A_knownZ(cells.w, cells.h, draw(n_center, cells.q), paths, z, delta, self.k)
        raise ValueError(f"Unsupported auditor/budget: {method}, p={p}")


def evaluate_audit(run, rng, section, bank, method, p, q, reps, delta,
                   *, unsafe=False, normal_threshold=None, c0=16, **fields):
    run.guard.check()
    values = bank.sample(rng, method, p, q, reps, delta,
                         normal_threshold=normal_threshold, c0=c0)
    if np.all(values == values[0]):
        est = Estimate(float(values[0]), 0.)
    else:
        est = Estimate(float(values.mean()), float(values.std(ddof=1) / math.sqrt(reps)))
    issued = ""
    if unsafe:
        issued = int(np.count_nonzero(rng.random(reps) < values))
    actual_q = q // 2 + ((q - q // 2) // (bank.b + bank.t)) * (bank.b + bank.t)
    run.record(section, method=method, world="unsafe" if unsafe else "safe",
               p=p, q=q, reps=reps, power=est.power, se=est.se, observed_issues=issued,
               target_samples=p * reps, proposal_samples=actual_q * reps, named_scores=0,
               **fields)
    return est


def minimum_lr(run, rng, section, unsafe, safe, budget, target, reps, delta,
               *, max_p=None, normal_threshold=None, **fields):
    """Bracket true power's monotone crossing using finite-MC power estimates."""
    q0, q1, p0, p1 = unsafe.q, safe.q, unsafe.p, safe.p

    def probe(p):
        run.guard.check()
        q = budget(p)
        answer = LR_oracle(p0, p1, q0, q1, p, q, delta, reps,
                           rng=rng, normal_threshold=normal_threshold)
        run.record(section, method="LR_oracle", kind="probe",
                   p=p, q=q, reps=reps, power=answer.power, se=answer.se,
                   null_calibration=answer.null_calibration,
                   target_samples=2 * reps * p, proposal_samples=2 * reps * q,
                   named_scores=0, **fields)
        return answer

    low = 0
    if budget(0):
        result = probe(0)
        if result.power >= target:
            return 0, result, "reached"
    if max_p == 0:
        return 0, None, "q_cap"
    high = 1
    high_result = probe(high)
    while high_result.power < target:
        if high == max_p:
            return high, high_result, "q_cap"
        low, high = high, 2 * high
        if max_p is not None:
            high = min(high, max_p)
        high_result = probe(high)
    while high - low > 1:
        mid = (low + high) // 2
        result = probe(mid)
        if result.power >= target:
            high, high_result = mid, result
        else:
            low = mid
    return high, high_result, "reached"


def slope(x, y, *, ci_level=None):
    fit = linregress(np.log(x), np.log(y))
    result = dict(slope=float(fit.slope), points=len(x))
    if ci_level is not None:
        scale = student_t.ppf((1 + ci_level) / 2, len(x) - 2) * fit.stderr
        result.update(ci_lo=float(fit.slope - scale), ci_hi=float(fit.slope + scale))
    return result


def save_figure(run, fig, name):
    fig.tight_layout()
    fig.savefig(run.path / "figs" / name, dpi=150)
    plt.close(fig)
