"""No-resampling release for the finite Power-SMC sampler (optimal proposal).

Exact-ramp convention: native updates for t=1..T are first
    w += (alpha(t)-alpha(t-1)) * cumulative_logp
for every slot, including absorbing slots; active slots then add log Z to
w and the sampled base log-probability to cumulative_logp. Native weight
increments and base log-probabilities must be <=0; clamp numerical noise
to 0 in BOTH the reference and accelerated implementations. Use Python
binary64 with identical alpha(t) and update order on both paths.
"""
import math


def ess_lower_log(fixed, caps):
    """Tight real-arithmetic lower ESS on fixed weights and [0, cap] boxes.

    Inputs are log weights; -inf denotes zero. Returns a floating estimate.
    A caller making a threshold decision must leave a numerical margin.
    """
    fixed = tuple(fixed)
    caps = tuple(caps)
    if any(math.isnan(x) or x == math.inf for x in (*fixed, *caps)):
        raise ValueError("Invalid weight")
    fixed = [x for x in fixed if x != -math.inf]
    caps = sorted((x for x in caps if x != -math.inf), reverse=True)
    if not fixed:
        if not caps:
            raise ValueError("All weights vanish")
        return 1.0
    # Scale the fixed-only endpoint independently. An open cap thousands of
    # log-units above every fixed mass would otherwise erase this endpoint.
    scale = max(fixed)
    vals = [math.exp(x - scale) for x in fixed]
    total = math.fsum(vals)
    squares = math.fsum(x * x for x in vals)
    worst = total * total / squares
    for x in caps:
        if x > scale:
            ratio = math.exp(scale - x)
            total *= ratio
            squares *= ratio * ratio
            scale = x
        value = math.exp(x - scale)
        total += value
        squares += value * value
        worst = min(worst, total * total / squares)
    return worst


def release_envelope(logw, logp, done, t, cap, block, ess_fraction, alpha_at, *, check_at_cap=True):
    """Return final log caps (exact on done) iff all future checks clear.

    Call immediately AFTER the native check, if any, at current t. All
    checks at positive multiples of block are honored. Pass check_at_cap=False
    only for a source which has analytically marginalized its last systematic
    resampling and final uniform slot draw. The last ramp is always applied.
    None means inconclusive; it never means resampling is guaranteed.
    When this succeeds, the caller can enter the terminal categorical race.
    A Gumbel-plus-cap proof is still needed before discarding any slot.
    """
    n = len(logw)
    if not (n and len(logp) == len(done) == n and 0 <= t <= cap and block >= 1 and 0 <= ess_fraction <= 1):
        raise ValueError("Invalid SMC state")
    if any(not math.isfinite(x) for x in [*logw, *logp]) or any(p > 0 for p in logp):
        return None
    upper = list(logw)
    last = alpha_at(t)
    if not math.isfinite(last) or last < 1:
        return None
    checkpoints = range((t // block + 1) * block, cap + int(check_at_cap), block)
    threshold = ess_fraction * n
    # Native ESS uses binary64 exp, normalization, summation and reciprocal.
    # Strict slack is much larger than the O(n^2 eps) absolute propagation
    # error at ESS <= n; ties/near-ties fall through to the native check.
    margin = max(1e-10, 256 * n * n * math.ulp(1.0))
    def clear():
        lower = ess_lower_log((w for w,d in zip(upper,done) if d), (w for w,d in zip(upper,done) if not d))
        return lower > threshold + margin
    # Without fixed mass the box can approach just one nonzero particle.
    if len(checkpoints) and not any(done) and threshold + margin >= 1:
        return None
    last_checked = False
    for s in range(t + 1, cap + 1):
        a = alpha_at(s)
        if not math.isfinite(a) or a < last:
            return None
        if a != last:
            # Native binary64 replay amounts to an exact upper envelope for
            # each represented slot: multiplication by nonnegative delta and
            # addition are monotone, actual future logp can only decrease.
            delta = a - last
            upper = [w + delta*p for w,p in zip(upper,logp)]
            if any(not math.isfinite(w) for w in upper):
                return None
            last_checked = False
        last = a
        if s % block == 0 and (s < cap or check_at_cap) and not last_checked:
            if not clear():
                return None
            last_checked = True
    return tuple(upper)
