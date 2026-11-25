"""Exact interval tests for forced ESS-triggered systematic resampling.

certified_parents takes lower/upper *log* weights at one future checkpoint.
It returns the whole parent vector, or None. Only an already certified-dead
slot can be dropped. All survivors must reach that checkpoint on their
original indexed random tapes; multiply selected parents need one prefix.
Never use a possible parent vector as if it were a certified one.
"""
import math


def ess_upper_ratio(lo, hi):
    """Exact maximum ESS as numerator/denominator on an integer weight box.

    Zero lies in some boxes. The all-zero weight vector is excluded.
    Algorithm works unchanged on exact common-scaled nonnegative integers.
    """
    if len(lo) != len(hi) or any(l < 0 or h < l for l,h in zip(lo,hi)):
        raise ValueError("Invalid intervals")
    boxes = [(l,h) for l,h in zip(lo,hi) if h > 0]
    if not boxes:
        raise ValueError("No positive weight")
    if max(l for l,h in boxes) <= min(h for l,h in boxes):
        return len(boxes), 1
    # Maximal ESS has weights clipped to a common water level b:
    # b = sum(w_i^2)/sum(w_i). R(b) decreases across the breakpoints.
    levels = sorted({x for pair in boxes for x in pair})
    def residual(b):
        weights = [min(h,max(l,b)) for l,h in boxes]
        return sum(w*(w-b) for w in weights)
    left, right = 0, len(levels)-1
    while left < right:
        mid = (left+right)//2
        if residual(levels[mid]) > 0:
            left = mid+1
        else:
            right = mid
    high = levels[left]
    if residual(high) == 0:
        w = [min(h,max(l,high)) for l,h in boxes]
        return sum(w)**2, sum(x*x for x in w)
    low = levels[left-1]
    twice_mid = low+high
    bound = [l if twice_mid < 2*l else h for l,h in boxes if twice_mid < 2*l or twice_mid > 2*h]
    free = len(boxes)-len(bound)
    bsum, bsq = sum(bound), sum(b*b for b in bound)
    return free*bsq + bsum*bsum, bsq


def _box(log_lo, log_hi):
    """Outward dyadic approximation scaled to the largest high endpoint.

    At least 80 fractional bits (adapted to the largest finite lower); slack covers
    double exp and normalization. Very small positive upper bounds remain
    positive (one integer unit plus slack) rather than underflowing to zero.
    """
    n = len(log_lo)
    if not n or n != len(log_hi) or any(math.isnan(x) or x == math.inf for x in (*log_lo,*log_hi)):
        raise ValueError("Invalid weight logs")
    if any(l > h for l,h in zip(log_lo,log_hi)) or max(log_hi) == -math.inf:
        raise ValueError("Empty weight box")
    scale = max(log_hi)
    max_lower = max(log_lo)
    gap = scale-max_lower
    bits = 80 + int(min(8112,max(0,gap/math.log(2.0)))) if math.isfinite(gap) else 80
    def endpoint(x, lower):
        if x == -math.inf:
            return 0
        d = x-scale
        if d < -(bits+5)*math.log(2.0):
            v = 0
        elif d > -300:
            num,den = math.exp(d).as_integer_ratio()
            v = (num << bits) // den
        else:
            power = d/math.log(2.0)
            exponent = math.floor(power)
            num,den = math.exp2(power-exponent).as_integer_ratio()
            shift = bits+exponent
            v = (num << shift)//den if shift >= 0 else num//(den << -shift)
        # Includes a margin for range reduction even at 8192 binary digits.
        padding = (v >> 32) + 2
        return max(0,v-padding) if lower else v+padding
    return [endpoint(x,True) for x in log_lo], [endpoint(x,False) for x in log_hi]


def certified_parents(log_lo, log_hi, ess_fraction, offset, *, equivalence=None):
    """Return fixed parents for a strict ESS<threshold and bisect_right grid.

    offset is in [0,1). Native grid is (offset+j)/N, as in powersmc_mbpp.
    A threshold or an interior CDF boundary within the error margin causes
    a conservative fallback. Ties follow the source via that fallback.
    With optional equivalence keys, slots with the same non-None key must
    have identical post-reset observable state and all future evolution.
    Returned representative slots can differ from the native index trace.
    A None key always denotes a singleton, not a shared class.
    """
    n = len(log_lo)
    if not (0 <= offset < 1 and 0 <= ess_fraction <= 1):
        raise ValueError("Invalid SMC parameters")
    if equivalence is not None and len(equivalence) != n:
        raise ValueError("Wrong number of equivalence keys")
    lo,hi = _box(log_lo,log_hi)
    if not any(lo):
        return None
    margin = max(1e-10, 256*n*n*math.ulp(1.0))
    cutoff = ess_fraction*n-margin
    if cutoff <= 1:
        return None
    num, den = ess_upper_ratio(lo,hi)
    a,b = cutoff.as_integer_ratio()
    if num*b >= a*den:
        return None
    # For an interior prefix k, F_lo and F_hi use the opposing extremes
    # of the prefix and suffix. Keep exact common-scaled integer ratios.
    ltot, htot = sum(lo), sum(hi)
    low_cdf, high_cdf = [], []
    lprefix = hprefix = 0
    for l,h in zip(lo[:-1],hi[:-1]):
        lprefix += l
        hprefix += h
        low_den = lprefix+htot-hprefix
        high_den = hprefix+ltot-lprefix
        low_cdf.append((lprefix,low_den) if low_den else (1,1))
        high_cdf.append((hprefix,high_den) if high_den else (0,1))
    cdf_margin = max(1e-13, 64*n*math.ulp(1.0))
    parents = []
    for j in range(n):
        point = (offset+j)/n
        lnum,lden = (point-cdf_margin).as_integer_ratio()
        hnum,hden = (point+cdf_margin).as_integer_ratio()
        # If every possible CDF at a boundary falls strictly below h-eps,
        # that boundary is definitely crossed. A boundary strictly above
        # h+eps is never crossed. Fail on ties/ambiguous bands.
        earliest = sum(x*lden < lnum*y for x,y in high_cdf)
        latest = sum(x*hden <= hnum*y for x,y in low_cdf)
        if earliest != latest:
            if equivalence is None or equivalence[earliest] is None:
                return None
            if any(equivalence[k] != equivalence[earliest] for k in range(earliest+1,latest+1)):
                return None
        parents.append(earliest)
    return tuple(parents)


def native_logz_floor(alpha, vocab_size):
    """Use the same floor in both implemented samplers if opting into finite lowers.

    The analytical bound is (1-alpha)*log(V). The wider floating floor
    tolerates normal FP32 roundoff. Clamping is part of the finite oracle.
    """
    if vocab_size < 1 or alpha < 1:
        raise ValueError("Invalid exponent or vocabulary size")
    return (1.0 - alpha)*math.log(vocab_size) - 1e-4


def next_checkpoint_parents(logw, logp, done, t, checkpoint, ess_fraction,
                            offset, alpha_at, *, vocab_size=None, equivalence=None):
    """No resampling between t and checkpoint; t is after its native check.

    Full per-step ramp includes absorbing particles. Future active log Z
    and sampled log p must be <=0 in the actual implementation. The oracle
    must draw an independent offset for this specific checkpoint in advance.
    Pass vocab_size only when the actual increment on both engines obeys
    native_logz_floor(a,vocab_size). Without a further ramp change an open
    slot then has a finite lower bound even if it terminates before the check.
    For absorbed slots, equivalence can contain (token_tape, logp, reason,
    other observable state) keys. All active slots are kept singleton here.
    """
    if not (t < checkpoint and len(logw) == len(logp) == len(done)):
        raise ValueError("Invalid checkpoint state")
    if any(not math.isfinite(x) for x in (*logw,*logp)) or any(x > 0 for x in logp):
        return None
    a = alpha_at(t)
    if not math.isfinite(a) or a < 1:
        return None
    upper = list(logw)
    lower = list(logw) if vocab_size is not None else None
    for s in range(t+1, checkpoint+1):
        b = alpha_at(s)
        if not math.isfinite(b) or b < a:
            return None
        if b != a:
            upper = [w+(b-a)*p for w,p in zip(upper,logp)]
            if any(not math.isfinite(w) for w in upper):
                return None
        if lower is not None:
            floor = native_logz_floor(b,vocab_size)
            lower = [
                upper[i] if d else
                ((lower[i] + (b-a)*logp[i]) if (s == t+1 or b == a) else -math.inf) + floor
                for i,d in enumerate(done)
            ]
        a = b
    if lower is None:
        lower = [u if d else -math.inf for u,d in zip(upper,done)]
    if equivalence is not None:
        if len(equivalence) != len(done):
            raise ValueError("Wrong number of equivalence keys")
        equivalence = [key if d else None for key,d in zip(equivalence,done)]
    return certified_parents(lower,upper,ess_fraction,offset,equivalence=equivalence)
