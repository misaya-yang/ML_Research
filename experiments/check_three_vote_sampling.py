"""Exact rational sanity checks for the three-route result and collision ceiling."""
from collections import Counter
from fractions import Fraction as F
from itertools import product
from math import prod

GOLD = 3


def plurality(votes):
    counts = Counter(v for v in votes if v is not None)
    if GOLD not in counts or counts[GOLD] < max(counts.values()):
        return F(0)
    return F(1, sum(n == counts[GOLD] for n in counts.values()))


def accuracy(routes):
    return sum(prod(r[v] for r, v in zip(routes, votes)) * plurality(votes)
               for votes in product(*routes))


def collision_only_score(votes):
    n = sum(v is not None for v in votes)
    k = votes.count(GOLD)
    return F(0) if not k else F(1) if k >= 2 else F(1, n)


def main():
    # Uniform route mixture minimizes squared wrong masses over iid mixtures.
    distinct = accuracy([{GOLD: F(2, 5), a: F(3, 5)} for a in range(3)])
    iid = accuracy([{GOLD: F(2, 5), **{a: F(1, 5) for a in range(3)}}] * 3)
    assert distinct == F(62, 125) and iid == F(56, 125)
    for votes in product((None, 0, 1, 2, GOLD), repeat=3):
        wrong = [v for v in votes if v not in (None, GOLD)]
        rescue = len(wrong) == 2 and votes.count(GOLD) == 1 and wrong[0] == wrong[1]
        assert collision_only_score(votes) - plurality(votes) == (F(1, 3) if rescue else 0)
    print(f"iid={float(iid):.3f}; distinct={float(distinct):.3f}; ceiling verified on 125 triples")


if __name__ == '__main__':
    main()
