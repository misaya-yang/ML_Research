"""Exactness tests on identical random tapes, using a tiny randomly initialised Qwen2 (no weights).

The tiny model exercises the real HF KV-cache path: left-padded batches, the prefix trie,
pausing, cancellation and the coin-conditioned cap.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from spec_engine import MODES, Engine, parity  # noqa: E402

CFG = dict(T=32, K=12, alpha=1.5, seed=7)


class Bursty(torch.nn.Module):
    """Tiny random Qwen2 whose next-token law is nearly flat after a few token ids and sharply
    peaked otherwise, so Renyi entropy arrives in bursts as in real text."""

    def __init__(self, m, cut=4, lo=0.05, hi=12.0):
        super().__init__()
        self.m, self.cut, self.lo, self.hi = m, cut, lo, hi

    def forward(self, input_ids, **kw):
        out = self.m(input_ids=input_ids, **kw)
        last = input_ids[:, -1:]
        lo = torch.tensor(self.lo, dtype=out.logits.dtype)
        hi = torch.tensor(self.hi, dtype=out.logits.dtype)
        out.logits = out.logits * torch.where(last < self.cut, lo, hi)[..., None]
        return out


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=False)
    return Bursty(Qwen2ForCausalLM(cfg).eval().to(torch.float64))


def run(model, row, mode, shape="compact", cfg=CFG):
    with torch.inference_mode():
        return Engine(model, [13, 14, 15, 16], {1}, cfg, row, mode, shape=shape, device="cpu").execute()


def test_all_modes_pathwise_identical(tiny):
    totals = dict(coin=0, coin_shared=0, shared=0, distinct=0, acc=0, rej=0)
    for row in range(40):
        ref = run(tiny, row, "serial_eager")
        for mode in MODES[1:]:
            for shape in (("compact", "fixed") if mode.startswith("tree") else ("compact",)):
                rec = run(tiny, row, mode, shape)
                p = parity(ref, rec)
                assert p["identical"], (row, mode, shape, p)
                if mode == "tree_coin" and shape == "compact":
                    c = rec["control"]
                    totals["coin"] += c["coin_cancelled_accept_children"]
                    totals["shared"] += c["shared_pairs"]
                    totals["distinct"] += c["distinct_pairs"]
                    totals["coin_shared"] += c["coin_cancelled_shared_children"]
        for s in ref["steps"][1:]:
            totals["acc" if s["accepted"] else "rej"] += 1
    # the test is only meaningful if every branch of the logic was exercised
    assert totals["acc"] > 20 and totals["rej"] > 20, totals
    assert totals["shared"] > 10 and totals["distinct"] > 10, totals
    assert totals["coin"] + totals["coin_shared"] > 0, totals


def test_coin_changes_no_commit_tick_and_never_adds_work(tiny):
    """Corollary (C1): identical commit ticks to the latency-safe baseline; (C2) work only falls."""
    fewer = 0
    for row in range(40):
        lazy = run(tiny, row, "tree_safe")
        coin = run(tiny, row, "tree_coin")
        assert [s["commit_tick"] for s in lazy["steps"]] == [s["commit_tick"] for s in coin["steps"]]
        assert coin["live_unique_decode_inputs"] <= lazy["live_unique_decode_inputs"]
        fewer += coin["live_unique_decode_inputs"] < lazy["live_unique_decode_inputs"]
    assert fewer > 0


def test_lazy_never_later_than_eager(tiny):
    """Local early rejection can only move commits earlier (in ticks) on identical tapes."""
    for row in range(20):
        e = run(tiny, row, "tree_eager")
        z = run(tiny, row, "tree_lazy")
        assert all(b["commit_tick"] <= a["commit_tick"] for a, b in zip(e["steps"], z["steps"]))


def test_telescoping_identity():
    """Lemma 1: for d <= c, log A_{k+1}(z | y) = log A_{k+1}(z | x) - log A_k(y | x)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        T = 30
        x = -rng.exponential(1.0, T)              # log Z rows of the current state x
        c = int(rng.integers(1, T - 2))
        d = int(rng.integers(0, c + 1))
        y = np.r_[x[:c + 1], -rng.exponential(1.0, int(rng.integers(1, T - c)))]   # proposal at cut c
        zz = np.r_[x[:d + 1], -rng.exponential(1.0, int(rng.integers(1, T - d)))]  # proposal at cut d
        S = lambda a, cut: a[cut + 1:].sum()  # noqa: E731
        logA_k = S(y, c) - S(x, c)
        rej = S(zz, d) - S(x, d)
        acc = S(zz, d) - S(y, d)
        assert math.isclose(acc, rej - logA_k, abs_tol=1e-9)


def test_coin_lower_bound_distinct_cut():
    """Lemma 2, d > c: on acceptance Phi_A = S(y, d) > log U + Phi_k - S_y(c, d]."""
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(2000):
        T = 30
        x = -rng.exponential(1.0, T)
        c = int(rng.integers(0, T - 3))
        y = np.r_[x[:c + 1], -rng.exponential(0.5, int(rng.integers(2, T - c)))]
        d = int(rng.integers(c + 1, len(y) - 1))
        logu = math.log(rng.random())
        phik = x[c + 1:].sum()
        if not logu < y[c + 1:].sum() - phik:
            continue
        checked += 1
        assert y[d + 1:].sum() > logu + phik - y[c + 1:d + 1].sum() - 1e-12
    assert checked > 100


def test_coin_cap_implies_accept_branch_rejection():
    """Lemma 2 as a decision rule, both cut orders, every partial prefix of the child."""
    rng = np.random.default_rng(3)
    fired = 0
    for _ in range(4000):
        T = 24
        x = -rng.exponential(1.0, T)
        c = int(rng.integers(0, T - 3))
        y = np.r_[x[:c + 1], -rng.exponential(rng.choice([0.2, 1.0, 3.0]), int(rng.integers(2, T - c)))]
        logu, logv = math.log(rng.random()), math.log(rng.random())
        phik = x[c + 1:].sum()
        if not logu < y[c + 1:].sum() - phik:
            continue                                        # the child only matters if y is accepted
        d = int(rng.integers(0, len(y) - 1))
        z = np.r_[y[:d + 1], -rng.exponential(rng.choice([0.2, 1.0, 3.0]), int(rng.integers(1, T - d)))]
        lower = logu + (x[d + 1:].sum() if d <= c else phik - y[c + 1:d + 1].sum())
        final_accept = logv < z[d + 1:].sum() - y[d + 1:].sum()
        for m in range(d + 1, len(z) + 1):
            if z[d + 1:m].sum() - lower < logv:
                fired += 1
                assert not final_accept
                break
    assert fired > 50
