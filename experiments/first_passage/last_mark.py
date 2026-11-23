"""Forward-only exact first hit with one amplified terminal transition.

Callers supply the ordinary step and a one-sided terminal routine. The latter
must return each possible successful payload and its original debit with
1/c times their native probabilities. It may return None on the residual mass.
There is no conditional failure sampler. The reference finite-enumeration
driver uses exact rational coins from marked_horizon.
"""

from fractions import Fraction
from math import prod

from marked_horizon import Hit, Move, RationalSampler, enumerate_horizon


def bernoulli(p, choose):
    if not 0 <= p <= 1:
        raise ValueError("Invalid Bernoulli probability")
    if p == 0 or p == 1:
        return bool(p)
    return choose(((True, p), (False, 1 - p)))


def first_hit(caps, initial_state, ordinary_step, terminal_step, choose, *, logical_budget=None):
    """Return (first hit or None, number of ordinary or terminal calls issued).

    A callback receives the native history state, step and accumulated debit.
    The terminal callback also receives the public cap of that step. Future
    horizon randomness must be independent of the native transition randomness.
    """
    caps = tuple(Fraction(c) for c in caps)
    if any(not 0 <= c <= 1 for c in caps):
        raise ValueError("Invalid public cap")
    if logical_budget is not None and logical_budget < 0:
        raise ValueError("Negative logical budget")
    last = next((i for i in range(len(caps), 0, -1) if bernoulli(caps[i-1], choose)), 0)
    state, spent = initial_state, 0
    for i in range(1, last + 1):
        move = (ordinary_step(i, state, spent, choose) if i < last
                else terminal_step(i, state, spent, caps[i-1], choose))
        if move is None:
            if i < last:
                raise ValueError("An ordinary step must report its continuation or stopping state")
            return None, i
        if move.debit < 0:
            raise ValueError("Negative native debit")
        spent += move.debit
        if logical_budget is not None and spent > logical_budget:
            return None, i
        if move.payload is not None:
            return Hit(i, move.payload, spent), i
        if move.stop:
            return None, i
        state = move.state
    return None, last


def from_control(eligible_mass, eligible_control, native_emit):
    """Build the terminal callback using only a conditional *positive control* draw."""
    def terminal(i, state, spent, cap, choose):
        mass = Fraction(eligible_mass(i, state, spent))
        if not 0 <= mass <= cap or not cap:
            raise ValueError("Eligible action mass exceeds the certified cap")
        if not bernoulli(mass / cap, choose):
            return None
        action = eligible_control(i, state, spent, choose)
        return native_emit(i, state, spent, action, choose)
    return terminal


def native(horizon, initial_state, step, choose, logical_budget=None):
    state, cost = initial_state, 0
    for i in range(1, horizon + 1):
        move = step(i, state, cost, choose)
        cost += move.debit
        if logical_budget is not None and cost > logical_budget:
            return None, i
        if move.payload is not None:
            return Hit(i, move.payload, cost), i
        if move.stop:
            return None, i
        state = move.state
    return None, horizon


def check():
    # The native controller draws from its full support; the terminal sampler
    # uses only actions 1 and 2. Action 0 can change all later success chances.
    caps = tuple(map(Fraction, ("1/2", "0", "2/3", "1", "1/4")))
    initial = (0, 0)
    def mass(i, state, cost):
        return caps[i-1] / (1 + (state[0] + cost) % 2)

    def eligible(i, state, cost, choose):
        return choose(((1, Fraction(1, 3)), (2, Fraction(2, 3))))

    def emit(i, state, cost, action, choose):
        flip = int(bernoulli(Fraction(1 + state[0], 4), choose))
        new = ((state[0] + action + flip) % 3, action)
        pay = 1 + ((state[1] + action) % 2)
        success = action > 0 and new[0] == 0
        return Move(new, pay, (action, flip, state) if success else None,
                    i > 2 and state[1] == 2 and action == 0)

    def ordinary(i, state, cost, choose):
        q = mass(i, state, cost)
        action = choose(((0, 1-q), (1, q/3), (2, 2*q/3)))
        return emit(i, state, cost, action, choose)

    terminal = from_control(mass, eligible, emit)
    for budget in (None, 0, 3, 6):
        base, _ = enumerate_horizon(lambda choose: native(len(caps), initial, ordinary, choose, budget))
        actual, _ = enumerate_horizon(lambda choose: first_hit(caps, initial, ordinary, terminal,
                                                               choose, logical_budget=budget))
        assert base == actual
        print(f"joint law: budget={budget}, outcomes={len(base)}, failure={base.get(None, 0)}")

    def metered(choose, mode):
        charged = Fraction(0)

        def record(i, state, cost, action, rng):
            nonlocal charged
            move = emit(i, state, cost, action, rng)
            if mode == "predicted_savings":
                if action == 0:
                    charged += prod(1-c for c in caps[i:]) * move.debit
            else:
                charged += move.debit
            return move

        def forward(i, state, cost, rng):
            q = mass(i, state, cost)
            action = rng(((0, 1-q), (1, q/3), (2, 2*q/3)))
            return record(i, state, cost, action, rng)

        if mode == "one_sided":
            outcome, _ = first_hit(caps, initial, forward, from_control(mass, eligible, record),
                                   choose, logical_budget=6)
        else:
            outcome, _ = native(len(caps), initial, forward, choose, logical_budget=6)
        return outcome, charged

    costs = {mode: enumerate_horizon(lambda choose: metered(choose, mode))[1]
             for mode in ("native", "one_sided", "predicted_savings")}
    assert costs["native"] - costs["one_sided"] == costs["predicted_savings"]
    print("cost identity:", {k: str(v) for k, v in costs.items()})

    positive = (Fraction(1, 4), Fraction(2, 3), Fraction(1, 5), Fraction(1, 3))
    for values in ((), (Fraction(0),) * 3, (Fraction(1),) * 3, positive):
        for private_bit in (None, *range(len(values))):
            def hidden_emit(i, state, flagged):
                armed = state or (i - 1 == private_bit)
                return Move(armed, 1, i if armed and flagged else None)
            def step(i, state, cost, choose):
                return hidden_emit(i, state, bernoulli(values[i-1], choose))
            def forced(i, state, cost, cap, choose):
                return hidden_emit(i, state, True)
            ref, _ = enumerate_horizon(lambda choose: native(len(values), False, step, choose))
            got, calls = enumerate_horizon(lambda choose: first_hit(values, False, step, forced, choose))
            assert ref == got
            if private_bit is None:
                lower = sum(1-prod(1-c for c in values[i:]) for i in range(len(values)))
                assert calls == lower
            else:
                assert 1-got.get(None, 0) == 1-prod(1-c for c in values[private_bit:])
    # Exercise the runnable rational sampler as well as the exhaustive driver.
    rng = RationalSampler(seed=42)
    first_hit(caps, initial, ordinary, terminal, rng, logical_budget=6)
    print("private-address lower bound, empty and degenerate horizons: pass")


if __name__ == "__main__":
    check()
