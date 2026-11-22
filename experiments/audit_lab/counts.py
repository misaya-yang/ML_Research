"""Cell counts, elementary symmetric polynomials, and simple-world ROC."""

from dataclasses import dataclass
import math

import numpy as np

from .imh import window_sum_pmf


NORMAL_THRESHOLD = 10**12


def multinomial(rng, n, probabilities, reps, *, normal_threshold=None):
    """Use tallies only. For huge cells preserve the multinomial covariance
    by approximating sequential conditional binomials, keeping the sum exact.
    """
    probability = np.asarray(probabilities, dtype=float)
    probability = probability / probability.sum()
    if normal_threshold is None or n * probability.max() <= normal_threshold:
        return rng.multinomial(n, probability, size=reps)
    order = np.flatnonzero(probability > 0)
    order = order[np.argsort(probability[order], kind="stable")]
    suffix = np.cumsum(probability[order][::-1])[::-1]
    counts = np.zeros((reps, probability.size), dtype=np.int64)
    remaining = np.full(reps, n, dtype=np.int64)
    for index, mass in zip(order[:-1], suffix[:-1]):
        chance = probability[index] / mass
        if n * probability[index] > normal_threshold:
            mean = remaining * chance
            draw = np.rint(rng.normal(mean, np.sqrt(mean * (1 - chance))))
            counts[:, index] = np.clip(draw.astype(np.int64), 0, remaining)
        else:
            counts[:, index] = rng.binomial(remaining, chance)
        remaining -= counts[:, index]
    counts[:, order[-1]] = remaining
    return counts


@dataclass(frozen=True)
class PathTallies:
    counts: np.ndarray

    @property
    def n(self):
        return int(self.counts[0].sum())

    def averages(self):
        t = self.counts.shape[1] - 1
        values = np.arange(t + 1) / t
        return (np.sum(self.counts * values, axis=1) / self.n,
                np.sum(self.counts * values**2, axis=1) / self.n)


def path_tallies(rng, cells, b, t, q, reps, *, pmf=None, normal_threshold=None):
    if pmf is None:
        pmf = window_sum_pmf(cells, b, t)
    counts = multinomial(rng, q // (b + t), pmf, reps,
                         normal_threshold=normal_threshold)
    return PathTallies(counts)


def _positive_log_e(log_scores, counts, p):
    """Coefficient of z^p in the product of (1 + score*z)^count."""
    coefficients = np.full((p + 1, len(counts)), -np.inf)
    coefficients[0] = 0
    for j, score in enumerate(log_scores.T):
        n = counts[:, j]
        active = (n > 0) & np.isfinite(score)
        if not np.any(active):
            continue
        log_factor = np.zeros(len(counts))
        updated = coefficients.copy()
        for degree in range(1, min(p, int(n.max())) + 1):
            active &= n >= degree
            log_factor[active] += np.log(n[active] - degree + 1) - math.log(degree) + score[active]
            log_factor[~active] = -np.inf
            updated[degree:] = np.logaddexp(
                updated[degree:], coefficients[:-degree] + log_factor
            )
        coefficients = updated
    return coefficients[p]


def _log_e(scores, counts, scale, p):
    log_result = np.full(len(counts), -np.inf)
    enough = counts[:, scores > 0].sum(axis=1) >= p
    if not np.any(enough):
        return log_result
    counts, scale = counts[enough], scale[enough]
    z = np.where(counts > 0, scale[:, None] * scores, 0)
    if p > 500:
        with np.errstate(divide="ignore"):
            log_result[enough] = _positive_log_e(
                np.log(scale)[:, None] + np.log(scores), counts, p
            )
        return log_result

    # The common rescaling makes sum(x) = p, retaining the ratio and
    # avoiding factorial underflow at the moderately large p in X2d.
    power = z.copy()
    powers = np.empty((p, len(counts)))
    values = np.ones((p + 1, len(counts)))
    errors = np.zeros_like(values)
    eps = np.finfo(float).eps
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(p):
            powers[i] = np.sum(counts * power, axis=1)
            power *= z
        for j in range(1, p + 1):
            terms = powers[:j] * values[j - 1::-1]
            values[j] = (terms[::2].sum(axis=0) - terms[1::2].sum(axis=0)) / j
            propagated = (powers[:j] * errors[j - 1::-1]).sum(axis=0)
            roundoff = (counts.shape[1] + 4 * j) * eps * np.abs(terms).sum(axis=0)
            errors[j] = (propagated + roundoff) / j

    value = values[p]
    reliable = np.isfinite(value) & (value > 0) & (errors[p] <= value * 1e-10)
    result = np.empty(len(counts))
    result[reliable] = np.log(value[reliable])
    if not np.all(reliable):
        with np.errstate(divide="ignore"):
            result[~reliable] = _positive_log_e(
                np.log(scale[~reliable])[:, None] + np.log(scores), counts[~reliable], p
            )
    log_result[enough] = result
    return log_result


def log_ep_ratios(x, ys, cnt, p):
    counts = np.atleast_2d(cnt)
    if p == 0:
        return [np.zeros(len(counts)) for _ in ys]
    if p < 0 or np.any(counts.sum(axis=1) < p):
        raise ValueError("Degree must be between zero and the pool size")
    max_x = np.max(x)
    x = x / max_x
    scale = p / np.sum(counts * x, axis=1)
    log_denom = _log_e(x, counts, scale, p)
    return [_log_e(y / max_x, counts, scale, p) - log_denom for y in ys]


def e_p_ratio(x, y, cnt, p):
    """Return e_p(y)/e_p(x) for score arrays and one or more count vectors."""
    result = np.exp(log_ep_ratios(np.asarray(x), (np.asarray(y),), cnt, p)[0])
    return float(result[0]) if np.ndim(cnt) == 1 else result


def _log_lr(counts, unsafe, safe, n):
    common = (unsafe > 0) & (safe > 0)
    coef = np.log1p((safe[common] - unsafe[common]) / unsafe[common])
    # Subtract one common constant before projection to avoid large offsets.
    score = np.sum((counts[:, common] - n * unsafe[common]) * coef, axis=1)
    unique_safe = (unsafe == 0) & (safe > 0)
    unique_unsafe = (safe == 0) & (unsafe > 0)
    score[np.any(counts[:, unique_safe] > 0, axis=1)] = np.inf
    score[np.any(counts[:, unique_unsafe] > 0, axis=1)] = -np.inf
    return score


@dataclass(frozen=True)
class ROC:
    power: float
    se: float
    null_calibration: float


def LR_oracle(cellsP0, cellsP1, cellsQ0, cellsQ1, p, q, delta, reps=20000,
              *, rng, normal_threshold=None):
    """Calibrate on unsafe world 0 and evaluate on independent safe world 1."""
    target0, target1 = np.asarray(cellsP0), np.asarray(cellsP1)
    cost0, cost1 = np.asarray(cellsQ0), np.asarray(cellsQ1)

    def scores(target, proposal):
        p_counts = multinomial(rng, p, target, reps, normal_threshold=normal_threshold)
        q_counts = multinomial(rng, q, proposal, reps, normal_threshold=normal_threshold)
        return _log_lr(p_counts, target0, target1, p) + _log_lr(q_counts, cost0, cost1, q)

    null, alt = scores(target0, cost0), scores(target1, cost1)
    rank = reps - math.ceil(delta * reps)
    cutoff = np.partition(null, rank)[rank]
    above = np.count_nonzero(null > cutoff)
    ties = np.count_nonzero(null == cutoff)
    tie_probability = np.clip((delta * reps - above) / ties, 0, 1)
    decisions = (alt > cutoff) + tie_probability * (alt == cutoff)
    return ROC(float(decisions.mean()), float(decisions.std(ddof=1) / math.sqrt(reps)),
               float((above + tie_probability * ties) / reps))
