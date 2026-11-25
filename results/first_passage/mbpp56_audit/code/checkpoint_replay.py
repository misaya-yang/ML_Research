"""Materialize and reveal a finite SMC window on frozen slot-indexed tapes."""
import math
from resample_certificate import certified_parents, native_logz_floor


def end_intervals(w0, lp0, done0, h, s, alpha_at, vocab=None):
    """Future log weight intervals until next checkpoint, absent resampling."""
    if not h<s or len(w0)!=len(lp0) or len(w0)!=len(done0):
        raise ValueError("invalid window")
    lo,hi = list(w0),list(w0)
    a=alpha_at(h)
    for t in range(h+1,s+1):
        b=alpha_at(t)
        if b<a or a<1 or not math.isfinite(b):
            raise ValueError("invalid exponent schedule")
        if b!=a:
            hi=[x+(b-a)*p for x,p in zip(hi,lp0)]
        floor=native_logz_floor(b,vocab) if vocab is not None else -math.inf
        for i,done in enumerate(done0):
            if done:
                lo[i]=hi[i]
            elif b!=a and t!=h+1:
                lo[i]=-math.inf
            else:
                lo[i]=lo[i]+(b-a)*lp0[i]+floor
        a=b
    return lo,hi


def ranked_cohorts(upper, eligible, cohort_keys, group_size=16, gumbels=None):
    """Greedy whole-prefix packing in descending best possible end score."""
    if group_size<1:
        raise ValueError("invalid cohort size")
    keys=list(range(len(upper))) if cohort_keys is None else cohort_keys
    groups={}
    for i in eligible:
        groups.setdefault(keys[i],[]).append(i)
    def rank(group):
        return max((upper[i]+(gumbels[i] if gumbels is not None else 0.),-i) for i in group)
    waves=[]
    wave=[]
    for group in sorted(groups.values(),key=rank,reverse=True):
        if wave and len(wave)+len(group)>group_size:
            waves.append(wave); wave=[]
        wave.extend(group)
    if wave:
        waves.append(wave)
    return waves


def certify_after_reveal(lower,upper,materialized,*,ess_fraction,offset=None,
                         gumbels=None,equivalence=None):
    """materialized maps *available* slot indices to exact endpoint log weights.

    Returns ('terminal',winner), ('resample',tuple(parents)), or None. A
    named parent without a materialized endpoint STATE must still be run.
    Exact endpoint weights of unsimulated slots are never arguments here.
    """
    lo,hi=list(lower),list(upper)
    for i,w in materialized.items():
        lo[i]=hi[i]=w
    if gumbels is None:
        parent=certified_parents(lo,hi,ess_fraction,offset,equivalence=equivalence)
        return None if parent is None else ('resample',parent)
    if not materialized:
        return None
    k=max(materialized,key=lambda i:(materialized[i]+gumbels[i],-i))
    lower_winner=(materialized[k]+gumbels[k],-k)
    if all(i in materialized or (hi[i]+gumbels[i],-i)<=lower_winner for i in range(len(hi))):
        return 'terminal',k
    return None


def _cost(groups,waves):
    paid_positions=paid_calls=0
    for wave in waves:
        for step in groups:
            per_step={step[i] for i in wave if step[i]>=0}
            paid_positions+=len(per_step)
            paid_calls+=bool(per_step)
    touched={i for wave in waves for i in wave}
    shared_positions=sum(len({step[i] for i in touched if step[i]>=0}) for step in groups)
    return paid_positions,paid_calls,shared_positions


def schedule(window,group_size=16):
    h,s,cap=window['h'],window['s'],window['cap']
    alpha,ramp=window['alpha'],window['ramp']
    def alpha_at(t):
        return 1. if t==0 else 1.+(alpha-1.)*(min(t/ramp,1.) if ramp else 1.)
    lower,upper=end_intervals(window['w0'],window['lp0'],window['done0'],
                              h,s,alpha_at,window.get('vocab'))
    # This dictionary is the only recipient of true endpoint weights. For
    # already absorbed slots, ramped endpoint weights are known from h.
    known={i:upper[i] for i,d in enumerate(window['done0']) if d}
    alive=[i for i,d in enumerate(window['done0']) if not d]
    terminal=s==cap
    g=window['g'] if terminal else None
    ranked=ranked_cohorts(upper,alive,window.get('cohort_keys'),group_size,g)
    waves=[]
    decision=None
    def reveal(ids):
        if ids:
            waves.append(ids)
        for i in ids:
            known[i]=window['wend'][i]
    for j in range(len(ranked)+1):
        decision=certify_after_reveal(lower,upper,known,ess_fraction=window['ess'],
                                      offset=window['offset'],gumbels=g)
        if decision is not None:
            if decision[0]=='resample':
                rem=sorted(set(decision[1])-known.keys())
                # Knowing an identity does not give us its unknown trajectory.
                # All remaining required parent states can be advanced together.
                reveal(rem)
            break
        if j<len(ranked):
            reveal(ranked[j])
    # Only diagnostics beyond this point may inspect unrevealed true outcomes.
    if decision and ('ref_winner' in window if terminal else 'ref_parents' in window):
        actual=window['ref_winner'] if terminal else window['ref_parents']
        if not terminal and actual is not None: actual=tuple(actual)
        if decision[1]!=actual:
            raise AssertionError(f"Certificate mismatch at {s}: {decision} vs {actual}")
    ref_ids=[window['ref_winner']] if terminal else window.get('ref_parents')
    if ref_ids is None: ref_ids=alive
    oracle=set(ref_ids).intersection(alive)
    native,calls,_=_cost(window['groups'],[alive])
    paid,paid_calls,shared=_cost(window['groups'],waves)
    oracle_positions,_,_=_cost(window['groups'],[oracle])
    return dict(h=h,s=s,kind='terminal' if terminal else 'intermediate',active_at_h=len(alive),
                certified=bool(decision),realized_skip=bool(set(alive)-known.keys()),
                skipped_slots=len(set(alive)-known.keys()),waves=list(map(len,waves)),
                native_positions=native,schedule_positions=paid,reused_across_cohorts_positions=shared,
                duplicated_prefix_positions=paid-shared,oracle_positions=oracle_positions,
                native_calls=calls,schedule_calls=paid_calls)
