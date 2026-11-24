"""Depth-two speculative execution of single-block, fixed-horizon, uniform-cut PowerMH.

Derived from results/local_runs/powermh-native-73eddb4b/code/gpu_depth_two/depth_two.py (same
tape addressing, kernel, pause rule and cancellation). Changes:

* mode ``tree_coin``: the coin-conditioned cap of theory/proofs/speculative_first_passage.md,
  Lemma 2. A speculative accept-branch child is cancelled for good as soon as
  ``child.partial - L_A < log U_{k+1} - eps`` with ``L_A = log U_k + Phi_k - S_y(c, d]``;
  when the parent is later accepted the ordinary check on the realized denominator must reject
  the child without reading another token, and the engine asserts this (online theorem check);
* every committed step records the engine tick, model-call count, wall time and work counters at
  the moment its decision became known, so the first-verified-answer ledger can be cut at its
  first hit without rerunning;
* the cross certificate (which elides an intermediate state and so breaks the first-hit ledger)
  is not included.

A *tick* is one lockstep advance of every active cursor (the logical critical path). A tick issues
a model call only if some active cursor's next prefix is not already in the prefix trie.
"""
import hashlib
import math
import time

import numpy as np

from prefix_decoder import Decoder

MODES = ("serial_eager", "serial_lazy", "tree_eager", "tree_lazy", "tree_safe", "tree_coin")


def token_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def margin(*xs):
    """Floating-point guard for certificates that compare sums formed in different orders."""
    return 1e-9 * (1.0 + sum(abs(x) for x in xs if math.isfinite(x)))


class Cursor:
    def __init__(self, parent, cut, step, rt):
        self.rt, self.cut, self.step = rt, cut, step
        self.ids = parent.ids[:cut] if parent else []
        self.nodes = parent.nodes[:cut + 1] if parent else [rt.dec.root]
        self.z = parent.z[:cut] if parent else []
        self.partial = 0.0
        self.done = False
        self.cancelled = False
        self.label = f"step{step}"

    def consume(self, node):
        j = len(self.ids)
        u = np.random.default_rng([self.rt.seed, self.rt.row, self.step, j]).random()
        v = node.sample(u)
        self.ids.append(v)
        self.z.append(node.logz)
        if j > self.cut:
            self.partial += node.logz
        self.done = v in self.rt.eos or len(self.ids) == self.rt.T
        self.rt.draws += 1


class Engine:
    def __init__(self, model, prompt, eos, cfg, row, mode, shape="compact", device="mps"):
        if mode not in MODES:
            raise ValueError(mode)
        self.t0 = time.perf_counter()
        self.T, self.seed, self.row, self.eos = cfg["T"], cfg["seed"], row, set(eos)
        self.mode, self.shape = mode, shape
        self.lazy = mode not in ("serial_eager", "tree_eager")
        self.tree = mode.startswith("tree")
        self.coin = mode == "tree_coin"
        # tree_safe / tree_coin never pause a cursor that a still-possible branch needs (latency-neutral)
        self.safe = mode in ("tree_safe", "tree_coin")
        self.dec = Decoder(model, prompt, cfg["alpha"], 3 if (self.tree and shape == "fixed") else 1, device)
        self.tape = []
        for k in range(1, cfg["K"] + 1):
            r = np.random.default_rng([self.seed, row, k, 10 ** 6])
            c = int(r.integers(self.T))
            v = float(r.random())
            self.tape.append((k, c, math.log(v) if v else -math.inf))
        self.draws = 0
        self.steps = []
        self.dead_labels = set()
        self.tick = 0
        self.control = dict(prefetch_pairs=0, shared_pairs=0, distinct_pairs=0, accept_children_launched=0,
                            coincidental_shared_prefixes=0, cancelled_accept_children=0, cancelled_reject_children=0,
                            coin_cancelled_accept_children=0, coin_cancel_at_launch=0,
                            coin_cancelled_shared_children=0, coin_checks_on_realized_branch=0)

    # ----------------------------------------------------------------- mechanics
    def advance(self, cursors):
        self.tick += 1
        self.dec.tick = self.tick
        need = [x for x in cursors if len(x.ids) > x.cut]
        nxt = self.dec.children([(x.nodes[-1], x.ids[-1]) for x in need], [x.label for x in need]) if need else []
        for x, node in zip(need, nxt):
            x.nodes.append(node)
        for x in cursors:
            x.consume(x.nodes[-1])

    @staticmethod
    def denom(parent, cut):
        return sum(parent.z[cut + 1:])

    def decision(self, x, phi, u):
        if x.done:
            return u < x.partial - phi
        if self.lazy and x.partial < u + phi:
            return False
        return None

    def roll(self, cur, phi, u):
        verdict = self.decision(cur, phi, u)
        while verdict is None:
            self.advance([cur])
            verdict = self.decision(cur, phi, u)
        return verdict

    def snapshot(self):
        s = self.dec.stats
        return dict(commit_tick=self.tick, commit_calls=s["decode_calls"], commit_wall_s=time.perf_counter() - self.t0,
                    commit_decode_inputs=s["decode_inputs"], commit_live_inputs=s["live_unique_decode_inputs"],
                    commit_dummy_inputs=s["dummy_decode_inputs"], commit_decode_s=s["decode_s"])

    def record(self, parent, child, t, accepted, post, **extras):
        k, c, u = t
        z = dict(k=k, c=c, logu=u, accepted=accepted,
                 parent_sha256=token_hash(parent.ids) if parent is not None else None,
                 post_ids=list(post.ids) if post is not None else None,
                 observed_proposal_tokens=child.ids[c:] if child else [],
                 observed_proposal_logz=child.z[c:] if child else [],
                 proposal_complete=child.done if child else False, **self.snapshot(), **extras)
        self.steps.append(z)

    # ----------------------------------------------------------------- policy steps
    def single(self, parent, t):
        k, c, u = t
        if c >= len(parent.ids):
            self.record(parent, None, t, True, parent, pad=True)
            return parent
        x = Cursor(parent, c, k, self)
        acc = self.roll(x, self.denom(parent, c), u)
        self.record(parent, x, t, acc, parent if not acc else x, pad=False)
        return x if acc else parent

    def coin_rejects(self, partial, lower_denominator, logv):
        """Lemma 2: on {parent accepted}, child log-ratio <= partial - lower_denominator."""
        if not (math.isfinite(lower_denominator) and math.isfinite(logv)):
            return False
        return partial - lower_denominator < logv - margin(partial, lower_denominator, logv)

    def pair(self, parent, a, b):
        k, c, u = a
        kk, d, v = b
        self.control["prefetch_pairs"] += 1
        self.control["shared_pairs" if d <= c else "distinct_pairs"] += 1
        first = Cursor(parent, c, k, self)
        second = Cursor(parent, d, kk, self) if d < len(parent.ids) else None
        phi1 = self.denom(parent, c)
        phi2 = self.denom(parent, d) if second else None
        if second:
            second.label = f"pair{k}_reject"
        positive = second if d <= c else None
        lower_A = None          # coin lower bound on the accept-branch denominator of a distinct child
        coin_info: dict = dict(pair_type="shared" if d <= c else "distinct")
        while True:
            first_verdict = self.decision(first, phi1, u)
            if first_verdict is not None:
                break
            if self.tree and d > c and positive is None and len(first.ids) > d:
                if second is not None and parent.ids[:d] == first.ids[:d] and parent.nodes[d] is first.nodes[d]:
                    positive = second
                    self.control["coincidental_shared_prefixes"] += 1
                else:
                    positive = Cursor(first, d, kk, self)
                    positive.label = f"pair{k}_accept"
                    self.control["accept_children_launched"] += 1
                    coin_info.update(child_launch_tick=self.tick)
                    if self.coin:
                        lower_A = u + phi1 - sum(first.z[c + 1:d + 1])
            advancing = [first]
            shared = second is not None and d <= c
            if second is not None and not second.cancelled:
                if self.coin and shared and self.coin_rejects(second.partial, phi2 + u, v):
                    # Lemma 2 (d <= c): rejected on the accept branch, hence on both branches.
                    second.cancelled = True
                    self.control["coin_cancelled_shared_children"] += 1
                    coin_info.update(coin_cancel_tick=self.tick, coin_cancel_child_tokens=len(second.ids) - d)
                elif self.decision(second, phi2, v) is None:
                    advancing.append(second)
                elif self.safe and shared and not second.done:
                    advancing.append(second)  # the accept branch may still need it: do not pause
            if positive is not None and positive is not second and not positive.done and not positive.cancelled:
                if self.coin and self.coin_rejects(positive.partial, lower_A, v):
                    positive.cancelled = True
                    self.control["coin_cancelled_accept_children"] += 1
                    self.control["coin_cancel_at_launch"] += len(positive.ids) == d
                    coin_info.update(coin_cancel_tick=self.tick, coin_cancel_child_tokens=len(positive.ids) - d)
                else:
                    advancing.append(positive)
            self.advance(advancing)
        mid = first if first_verdict else parent
        self.record(parent, first, a, first_verdict, mid, pad=False, **coin_info)
        if not self.tree:
            raise AssertionError("pair() is only used by tree modes")
        candidate = positive if first_verdict else second
        if first_verdict and candidate is None and second is not None and d < len(mid.ids) \
                and parent.ids[:d] == mid.ids[:d] and parent.nodes[d] is mid.nodes[d]:
            candidate = second
        killed = second if first_verdict else positive
        if killed is not None and killed is not candidate:
            self.dead_labels.add(killed.label)
            self.control["cancelled_reject_children" if first_verdict else "cancelled_accept_children"] += 1
            self.steps[-1]["eliminated_child"] = dict(label=killed.label, tokens=len(killed.ids) - d,
                                                      complete=killed.done, coin_cancelled=killed.cancelled)
        if candidate is None:
            return self.single(mid, b)
        if d >= len(mid.ids) or candidate.ids[:d] != mid.ids[:d] or candidate.nodes[d] is not mid.nodes[d]:
            raise AssertionError("wrong parent or cutoff selected")
        phi_real = self.denom(mid, d)
        if candidate.cancelled:
            # Online check of Lemma 2: the realized ordinary bound must already reject.
            self.control["coin_checks_on_realized_branch"] += 1
            if self.decision(candidate, phi_real, v) is not False:
                raise AssertionError("coin-conditioned certificate violated")
        second_verdict = self.roll(candidate, phi_real, v)
        post = candidate if second_verdict else mid
        self.record(mid, candidate, b, second_verdict, post, pad=False, selected_subtree=candidate.label,
                    pair_type=coin_info["pair_type"], parent_accepted=bool(first_verdict),
                    child_coin_cancelled=candidate.cancelled)
        return post

    def execute(self):
        initial = Cursor(None, 0, 0, self)
        while not initial.done:
            self.advance([initial])
        self.steps.append(dict(k=0, c=0, logu=None, accepted=True, post_ids=list(initial.ids), pad=False,
                               observed_proposal_tokens=list(initial.ids), **self.snapshot()))
        parent = initial
        i = 0
        while i < len(self.tape):
            a = self.tape[i]
            if not self.tree or i + 1 == len(self.tape) or a[1] >= len(parent.ids):
                parent = self.single(parent, a)
                i += 1
                continue
            parent = self.pair(parent, a, self.tape[i + 1])
            i += 2
        dead_positions = sum(bool(g) and all(lbl in self.dead_labels for lbl in g)
                             for t in self.dec.trace for g in t["real_row_consumers"])
        return dict(mode=self.mode, shape=self.shape, issued_rows_exclusive_to_cancelled_subtrees=dead_positions,
                    init_ids=initial.ids, final_ids=parent.ids, steps=self.steps, control=self.control,
                    drawn_tokens=self.draws, ticks=self.tick, engine_wall_s=time.perf_counter() - self.t0,
                    **self.dec.stats, batch_trace=self.dec.trace)


def parity(ref, var):
    """Pathwise comparison of two executions on the same tape."""
    out: dict = dict(init_equal=ref["init_ids"] == var["init_ids"], final_equal=ref["final_ids"] == var["final_ids"],
               n_steps_equal=len(ref["steps"]) == len(var["steps"]))
    sm = dm = tm = 0
    for a, b in zip(ref["steps"], var["steps"]):
        tm += (a["k"], a["c"], a["logu"]) != (b["k"], b["c"], b["logu"])
        sm += a["post_ids"] != b["post_ids"]
        dm += a["accepted"] != b["accepted"]
    out.update(tape_mismatches=tm, state_mismatches=sm, decision_mismatches=dm)
    out["identical"] = bool(out["init_equal"] and out["final_equal"] and out["n_steps_equal"] and tm == sm == dm == 0)
    return out
