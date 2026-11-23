"""Exact verification in Plackett–Luce order with progressively revealed weights.

The caller owns the generator and verifier. One slot may have several reveal
steps. Its upper bound, including its final log weight, must be finite and may
only decrease. A finished slot carries the actual deliverable string.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import heapq
import math
import operator


@dataclass(frozen=True)
class Candidate:
    log_upper: float
    payload: str | None = None


@dataclass(frozen=True)
class Result:
    answer: str | None
    slot: int | None
    checked: tuple[str, ...]
    opened: tuple[int, ...]


def first_verified(
    candidates: Sequence[Candidate],
    gumbels: Sequence[float],
    budget: int,
    advance: Callable[[int], Candidate],
    verify: Callable[[str], bool],
) -> Result:
    """Return the first passing distinct payload, its slot and both ledgers.

    'checked' is also the logical verification clock. Reusing the same Gumbels
    and generation tapes reproduces the eager ordering, including ties by slot.
    Independent standard Gumbels realize the weighted sampling law.
    """
    budget = operator.index(budget)
    if budget < 0 or len(candidates) != len(gumbels):
        raise ValueError("Expected a nonnegative budget and one key per slot")
    states = list(candidates)

    def entry(i: int) -> tuple[float, int]:
        key = states[i].log_upper + gumbels[i]
        if not math.isfinite(key):
            raise ValueError("Log bounds and perturbed priorities must be finite")
        return -key, i

    queue = [entry(i) for i in range(len(states))]
    heapq.heapify(queue)
    checked = []
    seen = set()
    opened = []
    while queue and len(checked) < budget:
        _, i = heapq.heappop(queue)
        state = states[i]
        if state.payload is None:
            updated = advance(i)
            if updated.log_upper > state.log_upper:
                raise ValueError(f"Upper bound increased at slot {i}")
            states[i] = updated
            opened.append(i)
            heapq.heappush(queue, entry(i))
        elif state.payload not in seen:
            seen.add(state.payload)
            checked.append(state.payload)
            if verify(state.payload):
                return Result(state.payload, i, tuple(checked), tuple(opened))
    return Result(None, None, tuple(checked), tuple(opened))
