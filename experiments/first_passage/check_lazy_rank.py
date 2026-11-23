"""Exhaustive checks of stopped weighted ranking, including the exposure formula."""

from collections import Counter
from itertools import product
import json
import math

from lazy_rank import Candidate, first_verified


def eager(final, keys, budget, omit=None):
    order = sorted(
        (i for i in range(len(final)) if i != omit),
        key=lambda i: (-(final[i].log_upper + keys[i]), i),
    )
    checked = []
    for i in order:
        payload = final[i].payload
        if payload in checked:
            continue
        if len(checked) == budget:
            break
        checked.append(payload)
        if payload.startswith("pass"):
            return payload, i, tuple(checked), (-(final[i].log_upper + keys[i]), i)
        if len(checked) == budget:
            return None, None, tuple(checked), (-(final[i].log_upper + keys[i]), i)
    return None, None, tuple(checked), (math.inf, len(final))


def run():
    cases = 0
    calls = 0
    for payloads in product(("fail", "pass-a", "pass-b"), repeat=3):
        for weights in product((-2., 0.), repeat=3):
            final = [Candidate(w, z) for w, z in zip(weights, payloads)]
            for keys in product((-1., 1.), repeat=3):
                for budget in range(5):
                    # Slot 0 is finished. The others each require two reads.
                    paths = [[final[0]], [Candidate(1.), Candidate(0.), final[1]],
                             [Candidate(2.), Candidate(1.), final[2]]]
                    step = [0] * len(paths)

                    def advance(i):
                        step[i] += 1
                        return paths[i][step[i]]

                    verified = []

                    def verify(payload):
                        verified.append(payload)
                        return payload.startswith("pass")

                    res = first_verified([a[0] for a in paths], keys, budget, advance, verify)
                    oracle = eager(final, keys, budget)
                    assert (res.answer, res.slot, res.checked) == oracle[:3]
                    assert tuple(verified) == res.checked
                    actual_opens = Counter(res.opened)
                    for i, path in enumerate(paths):
                        threshold = eager(final, keys, budget, omit=i)[3]
                        expected = sum(
                            (-(bound.log_upper + keys[i]), i) < threshold
                            for bound in path[:-1]
                        ) if budget else 0
                        assert actual_opens[i] == expected
                        calls += actual_opens[i]
                    cases += 1
    print(json.dumps({"cases": cases, "private_prefix_reads": calls,
                      "endpoint_clock_ledger_and_exposure": "pass"}))


if __name__ == "__main__":
    run()
