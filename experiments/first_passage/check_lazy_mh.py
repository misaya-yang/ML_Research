"""Exact rational check: lazy early rejection for suffix-resampling power MH.

Policy (fixed horizon, deterministic padding after EOS, as in the bias-free
PowerMH construction): target pi(x) ∝ prod_s p(x_s | x_<s)^alpha, proposal
q(v|h) = p(v|h)^alpha / Z(h) with Z(h) = sum_v p(v|h)^alpha <= 1, cut c uniform
on {0..T-1}, regenerate x_c..x_{T-1} from q, accept with probability
min(1, prod_{s>c} Z(h'_s) / prod_{s>c} Z(h_s)).

Cost unit: one forward pass at a new non-padding prefix h (yields Z(h) and the
next-token law). Prefixes already read are cached and free.

Lazy rule: draw U first; after each new forward pass, reject as soon as the
partial product of proposal Z factors falls below U times the current state's
suffix product. Remaining factors are <= 1, so the decision is unchanged.

Checks, all in exact rational arithmetic on random small models:
  1. the lazy one-step kernel equals the eager kernel;
  2. for every non-root, non-padding prefix h,
        P(lazy ever reads h) == TV(trajectory law, original vs model altered
        at h to a fresh deterministic token '*'),
     i.e. the lazy read probability equals the counterfactual-influence lower
     bound that binds every trajectory-exact simulator;
  3. the final-state-only TV is <= the trajectory TV; strict gaps are counted
     (the output-only contract can admit cheaper simulators).
A float Monte Carlo pass also checks pathwise identity of decisions under
shared U and token tapes.
"""

from fractions import Fraction as F
import itertools
import random

TOK = ("a", "b", "E")
PAD = "P"
STAR = "*"


def make_model(rng, T, den=5):
    """Random rational next-token laws on every prefix of length < T."""
    table = {}
    for n in range(T):
        for h in itertools.product(TOK, repeat=n):
            if "E" in h:
                continue
            w = [rng.randint(1, den) for _ in TOK]
            s = sum(w)
            table[h] = {v: F(x, s) for v, x in zip(TOK, w)}
    return table


class World:
    def __init__(self, table, T, alpha, star_at=None):
        self.table, self.T, self.alpha, self.star_at = table, T, alpha, star_at
        self.rows = {}

    def law(self, h):
        """Next-token law at prefix h; None marks a free deterministic pad."""
        if "E" in h or (STAR in h and self.star_at is None):
            return None
        if self.star_at is not None and h[: len(self.star_at)] == self.star_at:
            return {STAR: F(1)}
        return self.table[h]

    def Z(self, h):
        lw = self.law(h)
        if lw is None:
            return F(1)
        return sum(p ** self.alpha for p in lw.values())

    def q(self, h):
        lw = self.law(h)
        if lw is None:
            return {PAD: F(1)}
        z = self.Z(h)
        return {v: p ** self.alpha / z for v, p in lw.items()}

    def completions(self, h):
        """All completions of h to length T with their q-probabilities."""
        if len(h) == self.T:
            yield h, F(1)
            return
        for v, pv in self.q(h).items():
            for full, pr in self.completions(h + (v,)):
                yield full, pv * pr

    def suffix_Z(self, x, c):
        out = F(1)
        for s in range(c + 1, self.T):
            out *= self.Z(x[:s])
        return out

    def is_unit(self, h):
        """True when a forward pass at h is a paid unit (non-root, non-pad)."""
        return 0 < len(h) < self.T and self.law(h) is not None


def one_step(world, x):
    """Exact eager kernel row: dict next_state -> prob (memoized per world)."""
    if x in world.rows:
        return world.rows[x]
    T = world.T
    row = {}
    for c in range(T):
        phi = world.suffix_Z(x, c)
        for full, pr in world.completions(x[:c]):
            num = world.suffix_Z(full, c)
            acc = min(F(1), num / phi)
            w = F(1, T) * pr
            row[full] = row.get(full, F(0)) + w * acc
            row[x] = row.get(x, F(0)) + w * (1 - acc)
    world.rows[x] = row
    return row


def one_step_lazy(world, x):
    """Kernel row built from the lazy rule, integrating over U exactly.

    For each (c, tape) the lazy run stops at the first s with
    partial_s < U * phi. It accepts only if it never stops, which happens
    exactly when U <= full/phi; the U-measure is min(1, full/phi).
    The function rebuilds that measure from the stopping thresholds rather
    than from the closed-form acceptance ratio.
    """
    T = world.T
    row = {}
    for c in range(T):
        phi = world.suffix_Z(x, c)
        for full, pr in world.completions(x[:c]):
            thresholds = []
            part = F(1)
            for s in range(c + 1, T):
                part *= world.Z(full[:s])
                thresholds.append(min(F(1), part / phi))
            survive = min(thresholds) if thresholds else F(1)
            w = F(1, T) * pr
            row[full] = row.get(full, F(0)) + w * survive
            row[x] = row.get(x, F(0)) + w * (1 - survive)
    return row


def initial(world):
    return dict(world.completions(()))


def trajectory_law(world, K):
    paths = {(x,): p for x, p in initial(world).items()}
    rows = {}
    for _ in range(K):
        nxt = {}
        for path, p in paths.items():
            x = path[-1]
            if x not in rows:
                rows[x] = one_step(world, x)
            for y, k in rows[x].items():
                if k:
                    nxt[path + (y,)] = nxt.get(path + (y,), F(0)) + p * k
        paths = nxt
    return paths


def tv(a, b):
    keys = set(a) | set(b)
    return sum(abs(a.get(k, F(0)) - b.get(k, F(0))) for k in keys) / 2


def final_marginal(paths):
    out = {}
    for path, p in paths.items():
        out[path[-1]] = out.get(path[-1], F(0)) + p
    return out


def lazy_read_prob(world, h, K):
    """P(the lazy simulator ever pays for a forward pass at h).

    DP over (state, flag). Prefixes of the current state were read when the
    state was generated, so flag is forced to 1 when h is such a prefix.
    """
    T = world.T

    def prefix_of(x):
        return x[: len(h)] == h

    dist = {}
    for x, p in initial(world).items():
        key = (x, prefix_of(x))
        dist[key] = dist.get(key, F(0)) + p
    for _ in range(K):
        nxt = {}
        for (x, flag), p in dist.items():
            if flag:
                for y, k in one_step(world, x).items():
                    if k:
                        nxt[(y, True)] = nxt.get((y, True), F(0)) + p * k
                continue
            for c in range(T):
                phi = world.suffix_Z(x, c)
                for full, pr in world.completions(x[:c]):
                    w = F(1, T) * pr * p
                    acc = min(F(1), world.suffix_Z(full, c) / phi)
                    # threshold at which the forward pass at h would be paid
                    read_thr = None
                    if len(h) > c and full[: len(h)] == h and x[:c] == h[:c]:
                        part = F(1)
                        for s in range(c + 1, len(h)):
                            part *= world.Z(full[:s])
                        read_thr = min(F(1), part / phi)
                    if read_thr is None:
                        nxt[(full, False)] = nxt.get((full, False), F(0)) + w * acc
                        nxt[(x, False)] = nxt.get((x, False), F(0)) + w * (1 - acc)
                    else:
                        # accept implies read (acc <= read_thr)
                        nxt[(full, True)] = nxt.get((full, True), F(0)) + w * acc
                        nxt[(x, True)] = nxt.get((x, True), F(0)) + w * (read_thr - acc)
                        nxt[(x, False)] = nxt.get((x, False), F(0)) + w * (1 - read_thr)
        dist = nxt
    return sum(p for (_, flag), p in dist.items() if flag)


def mc_pathwise(rng, T, alpha, trials=20000):
    """Float check: lazy and eager decisions agree on shared U and tapes."""
    table = make_model(rng, T)
    world = World(table, T, alpha)
    mismatches, saved, total = 0, 0, 0
    states = list(initial(world))
    for _ in range(trials):
        x = rng.choice(states)
        c = rng.randrange(T)
        u = rng.random()
        tape = [rng.random() for _ in range(T)]
        h = x[:c]
        for s in range(c, T):
            dist = world.q(h)
            r, acc, v = tape[s], 0.0, None
            for v, pv in dist.items():
                acc += float(pv)
                if r < acc:
                    break
            h = h + (v,)
        full = h
        phi = float(world.suffix_Z(x, c))
        ratio = float(world.suffix_Z(full, c)) / phi
        eager_accept = u < min(1.0, ratio)
        part, lazy_reject, paid = 1.0, False, 0
        for s in range(c + 1, T):
            if world.is_unit(full[:s]):
                paid += 1
            part *= float(world.Z(full[:s]))
            if part < u * phi:
                lazy_reject = True
                break
        eager_paid = sum(world.is_unit(full[:s]) for s in range(c + 1, T))
        mismatches += eager_accept == lazy_reject
        saved += eager_paid - paid
        total += eager_paid
    return mismatches, saved, total


def main():
    rng = random.Random(20261123)
    cases = units = strict = 0
    worst_gap = F(0)
    for T, K, alpha in [(3, 1, 2), (3, 2, 2), (3, 2, 3), (4, 1, 2), (3, 3, 2), (4, 2, 2)]:
        reps = 3 if T == 4 and K == 2 else 6
        for _ in range(reps):
            table = make_model(rng, T)
            base = World(table, T, alpha)
            for x in initial(base):
                assert one_step(base, x) == one_step_lazy(base, x), "lazy kernel differs"
            law0 = trajectory_law(base, K)
            fin0 = final_marginal(law0)
            for n in range(1, T):
                for h in itertools.product(TOK, repeat=n):
                    if not base.is_unit(h):
                        continue
                    alt = World(table, T, alpha, star_at=h)
                    law1 = trajectory_law(alt, K)
                    d_traj = tv(law0, law1)
                    d_fin = tv(fin0, final_marginal(law1))
                    r = lazy_read_prob(base, h, K)
                    assert r == d_traj, (T, K, alpha, h, r, d_traj)
                    assert d_fin <= d_traj
                    units += 1
                    if d_fin < d_traj:
                        strict += 1
                        worst_gap = max(worst_gap, (d_traj - d_fin) / d_traj)
            cases += 1
    mm, saved, total = mc_pathwise(rng, 6, 4)
    print({
        "models": cases,
        "units_checked": units,
        "read_prob_equals_trajectory_TV": "pass",
        "lazy_kernel_equals_eager": "pass",
        "final_state_TV_strictly_smaller": f"{strict}/{units}",
        "largest_relative_gap": f"{float(worst_gap):.4f}",
        "mc_pathwise_mismatches": mm,
        "mc_T6_alpha4_saved_units": f"{saved}/{total}",
    })
    assert mm == 0


if __name__ == "__main__":
    main()
