"""Numerical check (random trees) of the pairwise coupling identity.
Claim: for two exact-marginal samples that are stratified (k=2: one uniform per
half) at every step while they share a prefix, and independent once they split,
   P(S=1,S'=1) - mu^2 = - sum_x c(x) * Var_strata(fbar_x)
where c(x) = P(both at x) and fbar_x(s) = mean of M(child) over the stratum s of
the inverse-CDF at node x (children in the chosen order). Also shows that sorting
children by value M changes the gain in this example (not a global optimum claim)."""
import random, itertools
random.seed(7)
B, D = 3, 4
def build(d):
    if d == D: return {"s": float(random.random() < 0.35)}
    w = [random.random() ** 2 + 1e-3 for _ in range(B)]; z = sum(w)
    return {"p": [x / z for x in w], "c": [build(d + 1) for _ in range(B)]}
def M(n):
    if "s" in n:
        n["M"] = n["s"]; return n["s"]
    n["M"] = sum(p * M(c) for p, c in zip(n["p"], n["c"])); return n["M"]
def order(n, key):
    if "s" in n: return
    idx = sorted(range(B), key=key(n)); n["o"] = idx
    for c in n["c"]: order(c, key)
def pick(n, u):
    acc = 0.0
    for i in n["o"]:
        acc += n["p"][i]
        if u < acc: return i
    return n["o"][-1]
def fbar_var(n, k=2):
    # stratum means of child value under the inverse CDF in order n["o"]
    grid = 4000; vals = [[] for _ in range(k)]
    for g in range(grid):
        u = (g + 0.5) / grid; vals[int(u * k)].append(n["c"][pick(n, u)]["M"])
    m = [sum(v) / len(v) for v in vals]; mu = sum(m) / k
    return sum((x - mu) ** 2 for x in m) / k
def predicted(n, cx):
    if "s" in n: return 0.0
    tot = -cx * fbar_var(n)
    # co-location prob of child i = P(both pick i) under stratified pair
    grid = 2000; same = [0.0] * B
    for g in range(grid):
        u1 = (g + 0.5) / grid / 2; i = pick(n, u1)
        for h in range(grid // 10):
            u2 = 0.5 + (h + 0.5) / (grid // 10) / 2
            if pick(n, u2) == i: same[i] += 1.0 / (grid * (grid // 10))
    # both orders of assigning strata are symmetric for pairs
    return tot + sum(predicted(n["c"][i], cx * same[i]) for i in range(B) if same[i] > 0)
def simulate(root, R=300000):
    hit2 = 0
    for _ in range(R):
        a = b = root; together = True
        while "s" not in a:
            if together:
                s = random.random() < 0.5
                u1 = (int(s) + random.random()) / 2; u2 = (1 - int(s) + random.random()) / 2
            else:
                u1, u2 = random.random(), random.random()
            ia, ib = pick(a, u1), pick(b, u2)
            together = together and ia == ib and a is b
            a, b = a["c"][ia], b["c"][ib]
        hit2 += a["s"] * b["s"]
    return hit2 / R
root = build(0); mu = M(root)
for lab, key in [("probability-sorted", lambda n: (lambda i: -n["p"][i])), ("value-sorted", lambda n: (lambda i: (-n["c"][i]["M"], i)))]:
    order(root, key)
    pred = predicted(root, 1.0); sim = simulate(root) - mu * mu
    print(f"{lab:18s}: mu={mu:.4f}  simulated P(S=S'=1)-mu^2 = {sim:+.5f}   identity predicts {pred:+.5f}")
