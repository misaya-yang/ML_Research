"""Exact simulation and finite enumeration for the marked forward horizon."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import lcm, prod
from random import Random


@dataclass(frozen=True)
class Move:
    state: object
    debit: int
    payload: object = None
    stop: bool = False


@dataclass(frozen=True)
class Hit:
    step: int
    payload: object
    cost: int


def _flip(probability, choose):
    if probability == 0:
        return False
    if probability == 1:
        return True
    return choose(((True, probability), (False, 1 - probability)))


def _last_mark(caps, choose):
    for index in range(len(caps), 0, -1):
        if _flip(caps[index - 1], choose):
            return index
    return 0


def run_horizon(
    caps,
    initial_state,
    carrier_mass,
    draw_control,
    emit,
    choose,
    *,
    marked=True,
    logical_budget=None,
):
    """Return (successful record or None, issued emissions).

    Caps and carrier masses are exact rational probabilities. The two control
    draws must be the native distributions conditional on carrier membership.
    The emit callback can draw additional randomness through choose.
    """
    caps = tuple(Fraction(cap) for cap in caps)
    if any(not 0 <= cap <= 1 for cap in caps):
        raise ValueError("Caps must be probabilities.")
    if logical_budget is not None and logical_budget < 0:
        raise ValueError("The logical budget must be nonnegative.")

    horizon = _last_mark(caps, choose) if marked else len(caps)
    state, cost = initial_state, 0
    for step in range(1, horizon + 1):
        cap = caps[step - 1]
        mass = Fraction(carrier_mass(step, state, cost))
        if not 0 <= mass <= cap:
            raise ValueError("The carrier mass exceeds its certified cap.")
        if marked:
            mark = step == horizon or _flip(cap, choose)
            in_carrier = mark and _flip(mass / cap, choose)
        else:
            in_carrier = _flip(mass, choose)
        action = draw_control(step, state, cost, in_carrier, choose)
        move = emit(step, state, cost, action, choose)
        if move.debit < 0:
            raise ValueError("The logical debit must be nonnegative.")
        if move.payload is not None and not in_carrier:
            raise ValueError("A passing move fell outside the carrier.")
        cost += move.debit
        if logical_budget is not None and cost > logical_budget:
            return None, step
        if move.payload is not None:
            return Hit(step, move.payload, cost), step
        if move.stop:
            return None, step
        state = move.state
    return None, horizon


class RationalSampler:
    """Sample each finite rational distribution without float rounding."""

    def __init__(self, seed=None):
        self.rng = Random(seed)

    def __call__(self, options):
        denominator = lcm(*(weight.denominator for _, weight in options))
        threshold = self.rng.randrange(denominator)
        for item, weight in options:
            units = weight.numerator * (denominator // weight.denominator)
            if threshold < units:
                return item
            threshold -= units
        raise ValueError("Choice weights must sum to one.")


class _UnresolvedChoice(Exception):
    def __init__(self, options):
        self.options = options


def enumerate_horizon(run):
    """Replay the same runner over all exact outcomes of its choices."""
    pending = [((), Fraction(1))]
    law = defaultdict(Fraction)
    expected_calls = Fraction(0)
    while pending:
        tape, weight = pending.pop()
        cursor = 0

        def choose(options):
            nonlocal cursor
            if cursor == len(tape):
                raise _UnresolvedChoice(options)
            index = tape[cursor]
            cursor += 1
            return options[index][0]

        try:
            result, issued_calls = run(choose)
        except _UnresolvedChoice as unresolved:
            for index, (_, probability) in enumerate(unresolved.options):
                if probability:
                    pending.append((tape + (index,), weight * probability))
        else:
            law[result] += weight
            expected_calls += weight * issued_calls
    if sum(law.values()) != 1:
        raise AssertionError("The exact output law does not have mass one.")
    return dict(law), expected_calls


def _adaptive_instance(caps, *, marked, budget):
    def carrier_mass(step, state, cost):
        return caps[step - 1] * (1 if (state + cost) % 2 else Fraction(1, 2))

    def draw_control(step, state, cost, in_carrier, choose):
        choices = (2, 3) if in_carrier else (0, 1)
        return choose(tuple((choice, Fraction(1, 2)) for choice in choices))

    def emit(step, state, cost, action, choose):
        noise = int(_flip(Fraction(1, 3), choose))
        next_state = (state + action + noise) % 3
        debit = 1 + (state + action) % 2
        payload = (action, noise) if action >= 2 and next_state == 0 else None
        return Move(next_state, debit, payload, step >= 3 and action == 0)

    return enumerate_horizon(
        lambda choose: run_horizon(
            caps, 0, carrier_mass, draw_control, emit, choose,
            marked=marked, logical_budget=budget,
        )
    )


def _private_bits_instance(caps, bits, *, marked):
    def carrier_mass(step, state, cost):
        return caps[step - 1]

    def draw_control(step, state, cost, in_carrier, choose):
        return in_carrier

    def emit(step, armed, cost, action, choose):
        armed = armed or bits[step - 1]
        return Move(armed, 1, "passed" if armed and action else None)

    return enumerate_horizon(
        lambda choose: run_horizon(
            caps, False, carrier_mass, draw_control, emit, choose, marked=marked,
        )
    )


def _random_scan_instance(*, marked):
    caps = (Fraction(1, 2),) * 4
    initial = (0, 1, 0, 0)

    def violated_scope(state):
        if state[0] != state[1]:
            return (0, 1)
        if state[1] != state[2]:
            return (1, 2)
        if state[3] != 1:
            return (3,)
        return ()

    def carrier_mass(step, state, cost):
        return Fraction(len(violated_scope(state)), len(state))

    def draw_control(step, state, cost, in_carrier, choose):
        scope = violated_scope(state)
        sites = tuple(site for site in range(len(state)) if (site in scope) == in_carrier)
        return choose(tuple((site, Fraction(1, len(sites))) for site in sites))

    def emit(step, state, cost, site, choose):
        draw = int(_flip(Fraction(1 + state[(site + 1) % len(state)], 3), choose))
        next_state = state[:site] + (draw,) + state[site + 1:]
        return Move(next_state, 1, None if violated_scope(next_state) else next_state)

    return enumerate_horizon(
        lambda choose: run_horizon(
            caps, initial, carrier_mass, draw_control, emit, choose, marked=marked,
        )
    )


def main():
    caps = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(1, 3), Fraction(1, 4))
    for budget in (None, 6):
        reference_law, reference_calls = _adaptive_instance(caps, marked=False, budget=budget)
        marked_law, marked_calls = _adaptive_instance(caps, marked=True, budget=budget)
        assert marked_law == reference_law
        assert marked_calls < reference_calls
        print(
            f"adaptive budget={budget}, distinct outcomes={len(marked_law)}, "
            f"failure={marked_law[None]}, expected calls={reference_calls} -> {marked_calls}"
        )

    reference_law, reference_calls = _random_scan_instance(marked=False)
    marked_law, marked_calls = _random_scan_instance(marked=True)
    assert reference_law == marked_law
    assert 0 < marked_law[None] < 1 and marked_calls < reference_calls
    print(f"random scan, failure={marked_law[None]}, expected calls={reference_calls} -> {marked_calls}")

    no_bits = (False,) * len(caps)
    probes = [1 - prod(1 - cap for cap in caps[index:]) for index in range(len(caps))]
    for index, probe in enumerate(probes):
        bits = list(no_bits)
        bits[index] = True
        reference_law, _ = _private_bits_instance(caps, bits, marked=False)
        marked_law, _ = _private_bits_instance(caps, bits, marked=True)
        assert reference_law == marked_law
        assert 1 - marked_law.get(None, Fraction(0)) == probe
    null_law, null_calls = _private_bits_instance(caps, no_bits, marked=True)
    assert null_law == {None: Fraction(1)}
    assert null_calls == sum(probes)
    print(f"private-bit bound attained, address probabilities={probes}, expected calls={null_calls}")

    for edge_cap in (Fraction(0), Fraction(1)):
        _, calls = _private_bits_instance((edge_cap,) * 4, (False,) * 4, marked=True)
        assert calls == 4 * edge_cap
    _, no_calls = _private_bits_instance((), (), marked=True)
    assert no_calls == 0
    print("zero and unit caps and an empty horizon passed")


if __name__ == "__main__":
    main()
