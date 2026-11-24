# Speculative first passage: the information bottleneck, solved, measured and bounded

Claude, 2026-11-23.

**Status.** The theorem and proofs are in [theory/proofs/speculative_first_passage.md](../../theory/proofs/speculative_first_passage.md). The code, tests and native paired results are under `experiments/speculative_mh/` and `results/local_runs/speculative-mh-*`. Nothing is committed.

## 1. The question, and the answer

**The question.** Can a verifier-guided decoder use cheap information that is available during generation to avoid a substantial part of a search? It must still return the same first verified answer, at the same logical step, with the same failure law. How can it know a piece of work is skippable without doing that work?

**The answer, for the PowerMH family, in the token-reveal model.**

- **Only certificates make work skippable.** A certificate is an event, fixed by revealed rows and pre-drawn tape entries, that implies the decision for every completion. Nothing else can be used (Theorem 2).
- **For sharpening samplers, certificates are one-sided.**
  - A proposal can be certified *rejected* early: its Rényi-entropy debt crosses the pre-drawn coin's slack.
  - It can never be certified *accepted* before EOS (O1).
- **The new escape: the parent's coin.** In speculative execution, this asymmetry blocks discarding a child whose parent is undecided. The escape uses the parent's pre-drawn coin as the purchased information.
  - The child matters only if the parent is accepted.
  - Acceptance lower-bounds the parent's unread suffix score by $\log U_k+\Phi_k$.
  - A telescoping identity (Lemma 1) turns that into a cap on the child's ratio that is available before the parent finishes (Lemma 2).
  - Discarding at that cap is exact, latency-neutral, and the earliest irrevocable discard any exact execution can make (Theorem 3).
- **It cannot be substantial on real workloads.** A corollary (C3) says the cap can fire only when the child's rejection margin exceeds the parent's acceptance margin. On real text, accepted proposals clear the threshold by wide margins.

**The native test agrees with every prediction.** On MATH the coin cap removes 13 of 16,183 paid rows (0.08%) and changes no commit tick. All measured gains come from earlier rejections. The remaining speculative waste (4.2% of rows) is irreducible for latency-neutral exact schedules.

So the bottleneck is resolved as a characterization:

- you know a piece of work is skippable exactly when a rejection certificate, or a coin-conditioned certificate, holds;
- everything else is provably unskippable;
- the certifiable remainder is small.

## 2. What is new, and what is not

**Prior art, not claimed.**

- Early rejection in MCMC ([Solonen et al. 2012](https://doi.org/10.1214/12-BA724)).
- Prefetching and speculative MH trees: Brockwell 2006; Angelino et al. 2014, which uses predicted outcomes; PreSTO ([arXiv 2610.13171](https://arxiv.org/abs/2610.13171)).
- Certified thinning for serial work (our [certified_thinning](../../theory/proofs/certified_thinning.md)).
- Composing early rejection with a prefetch tree is engineering on top of these.
- I have not checked the prefetching-MH papers for the coin-conditioned bound line by line. The novelty of Lemma 2 still has to be checked against them.

**New.**

1. **Exactness for speculation.** Exactness of any certificate-respecting speculative schedule on an addressed tape. This covers trajectory and first-hit ledgers for every verifier (Theorem 1).
2. **Two obstructions in the reveal model.**
   - No acceptance certificate, so reject children of accepted parents are irreducible waste (O1).
   - Coin-blind executions cannot discard accept children before their parent completes (O2).
3. **The coin-conditioned cap.** The telescoping identity $\log A_{k+1}(z\mid y)=\log A_{k+1}(z\mid x)-\log A_k(y\mid x)$, and the cap $S_q(z)-\log U_k-\Phi_k+S_y(c,d]$. Exactness, latency neutrality and earliest-discard optimality are proved; the depth-$r$ product-of-coins form is stated.
4. **Falsifiable corollaries, tested natively.**
   - C1: no tick change.
   - C2: saving equals rows after the cap.
   - C3: margin condition.
   - C4: tick savings only from realized rejections.
   - C5: paid-work floor.

## 3. Native experiment

**Policy.** The frozen policy is single-block, fixed-horizon, uniform-cut PowerMH (Karan–Du / PreSTO eq. 6) with absorbing EOS and α=4.

**Tape.** Tokens, cuts and coins are addressed by (step, position) from one seed per problem. Every arm of a problem uses the same tape.

**Arms.** All arms share the same engine (`spec_engine.py`, derived from the repository's depth-two comparator). Each is a strict extension of the previous one:

| Arm | What it adds |
|---|---|
| `serial_lazy` | Serial MH with certified early rejection. This is the paid-work floor (C5) and the parity reference |
| `tree_eager` | Depth-two prefetch. Decisions only on completed scores; the eliminated subtree is cancelled when a decision is known. This is the completed-score cancellation baseline |
| `tree_safe` | + local early rejection. Committed proposals are decided at their certificate. A reject child stops at its own certificate when no other branch needs it. A cursor that a still-possible branch needs is never paused. **The baseline the request specifies** |
| `tree_coin` | + Rule C (coin-conditioned cap). **The mechanism** |

**Two physical shapes.**

- *compact*: rows leave the batch when cancelled.
- *fixed*: every model call is padded to width 3, so contention is held constant.

**Ticks** are lockstep advances until a decision is known: the logical critical path, independent of contention.

**Charging.** All of these costs are inside the measured wall time and itemized per run:

- prefill;
- model forwards;
- KV cache packing;
- float64 prefix scoring and inverse-CDF sampling over the 151,936-token vocabulary (the scoring every certificate uses);
- scheduling and cancellation logic.

### 3.1 MATH (the native PowerMH workload)

**Setup.**

- MATH-500 rows 0–9, Qwen2.5-Math-1.5B, the repository's plain-completion prompt.
- T=512, K=16, fp32 on M4 MPS, one model job at a time, `nice 19`.
- 160 MH steps: 60 accepted, 68 rejected, 32 no-ops.

**Exactness.** 50/50 accelerated runs are pathwise identical to `serial_lazy`: every state, decision and final token.

| Arm (compact unless noted) | Ticks | Model calls | Paid rows | Filler rows | Wall s |
|---|---:|---:|---:|---:|---:|
| serial_lazy | 29,887 | 15,376 | 15,376 | 0 | 611.0 |
| tree_eager (completed-score cancellation) | 25,146 | 17,195 | 20,822 | 0 | 731.2 |
| tree_safe (+ local early rejection; baseline) | 21,933 | 14,035 | 16,183 | 0 | 578.9 |
| **tree_coin (+ coin cap; mechanism)** | **21,933** | **14,035** | **16,170** | 0 | **583.2** |
| tree_safe, fixed width 3 | 21,933 | 14,035 | 16,183 | 25,922 | 729.1 |
| tree_coin, fixed width 3 | 21,933 | 14,035 | 16,170 | 25,935 | 699.0 |

**Where the gains come from.** Changes are paired over the 10 problems; the counts are problems lower / higher.

- **The mechanism (tree_safe → tree_coin)**
  - Ticks and calls are identical on every problem; commit ticks are identical step by step (C1 ✓).
  - Paid rows fall by 13 (−0.08%), from 3 coin cancellations among the 11 accept children launched. No shared child was ever cancelled.
  - The 13 saved rows equal exactly the rows the baseline issued for those children after the cancel tick (C2 ✓).
  - Wall time moves +0.7% compact (3/7) and −4.1% fixed (8/2), on identical work. These figures **are measurement noise**: run-to-run wall noise between arms is about ±4%.
- **Certificate-time decisions vs completed-score cancellation (tree_eager → tree_safe)**
  - Ticks −12.8% (9/0), calls −18.4%, paid rows −22.3% (8/0), wall −20.8% (8/2).
  - Per pair (C4 ✓): all 3,213 ticks saved come from units containing a realized rejection:
    - reject–reject pairs: 1,920;
    - accept–reject: 681;
    - reject–accept: 284;
    - single rejected steps: 328.
  - The 16 accept–accept pairs and 5 accepted singles save exactly 0.
  - This is the "shortened on-path rejection" effect. Ticks isolate it from contention: −12.8% is pure critical-path shortening.
  - The larger wall and row drops add the effect of fewer speculative rows.
  - **This is a measured gain of composing known pieces, not ours to claim as a mechanism.**
- **Contention (fixed → compact at identical work, tree_safe)**
  - Wall −20.6%.
  - Median forward time is 33.8, 44.2 and 48.2 ms at widths 1, 2 and 3 (41.1 ms at forced width 3).
  - On this device a second row costs about 31% of a call, so speculation's tick saving is mostly eaten.
- **Speculation itself (serial_lazy → tree_safe)**: ticks −26.6%, wall −5.3% (8/2), for +5.3% paid rows.

**Speculative waste in tree_coin.** 679 rows, 4.2% of paid rows:

- 143 are reject children of accepted parents. These are irreducible by O1.
- 536 are accept children issued before their parent's rejection certificate. They are irreducible by certified thinning together with Theorem 3(c), unless the launch policy itself speculates less.
- Nothing removable remains for any exact, latency-neutral rule given this launch policy.

### 3.2 MBPP+ fresh panel: degenerate for this kernel, so the run was stopped early

Setup: Qwen2.5-1.5B-Instruct, T=192.

- 109 of 144 steps are no-ops, because the cut lands past the end of a short answer.
- 33 of 35 real proposals are accepted, at ratio 1 on cached text.
- The tree launched no accept child at all.
- 81/81 runs are pathwise identical to serial eager.

The run was stopped after 9/40 tasks (`results/local_runs/speculative-mh-fresh40-20261123/README.md`) and the machine was given to MATH. On short-answer code with this kernel there is no search for any skip mechanism to act on.

### 3.3 Task level: correct answers vs actual cost (same 10 MATH rows, model, dtype, prompt)

**The baseline.** Independent samples from $q_\alpha$ (temperature 1/4, PowerMH's own proposal law), drawn from one cached prompt prefill on the same device in fp32. Answers use the same boxed-string normalization. Every exact PowerMH arm returns identical answers.

| Policy | Correct / 10 | Total wall s | Median per problem s |
|---|---:|---:|---:|
| PowerMH K=16, serial early rejection | 5 | 611.0 | 62.5 |
| PowerMH K=16, tree + coin cap (fastest exact) | 5 | 583.2 | 60.2 |
| one $q_\alpha$ sample | 5 | 173.7 | 18.5 |
| majority vote of 4 $q_\alpha$ samples | 6 | 547.0 | 61.6 |
| majority vote of 8 $q_\alpha$ samples | 6 | 1001.2 | 104.6 |

**Solved sets.**

- PowerMH solves {0, 2, 5, 6, 8}, the same set as the first $q_\alpha$ sample.
- Vote-8 solves those plus problem 3.
- No problem is solved only by PowerMH.

Plot: `frontier_math.png`; table: `frontier_math_table.csv`.

**Budget crossings.**

- At 30 s per problem: one sample 5, PowerMH 2.
- At 120 s: vote-8 6, PowerMH 5.

**What a PowerMH user gets that the cheaper alternative cannot deliver here: nothing.** One sample matches its correct count at 3.4× less wall-clock. A vote of four beats it at slightly less total time.

**Even a perfect accelerator would not change the decision.**

- Under C5, paid work cannot fall below serial early rejection.
- Driving its latency to zero would still deliver 5/10.
- Vote-8 delivers 6/10 by 120 s.

**Consistency.** This matches the repository's earlier native 20-row result on bf16: MH16 10/20 at 2,428 s against vote-8 12/20 at 872 s (`results/local_runs/powermh-native-73eddb4b/math_outcomes.json`).

**Limits.**

- Ten problems, so these counts cannot separate policies statistically.
- They show only that this workload offers no value an exact accelerator could unlock.

## 4. The prior evidence, accounted for

- **PreSTO's 481 exported trees.** The 33.5% removed by cancellation after full scoring is PreSTO's own saving, and we claim none of it.
  - The 365 batches whose longest realized-path proposal was a rejection predicted room to shorten the critical path.
  - Natively, that room is real: −12.8% ticks from certificate-time decisions, all of it from realized rejections.
  - But it belongs to the composition of early rejection with prefetching. On MPS it buys −20.8% wall over completed-score cancellation and −5.3% over serial early rejection.
- **The 56-task MBPP result.** The 15.43% position saving tying the best direct retry at 8 s is consistent with C5 and §3.3. An exact accelerator cannot deliver answers its policy does not find, so it only moves the cost axis.

## 5. Verdict

**Genuinely new.**

- A complete, checkable account of what decision-time information can skip in speculative exact search. This covers one exact mechanism for the speculative case (the parent's coin), two obstructions, and an optimality statement for the cap.
- A native decomposition confirming each falsifiable corollary exactly: C1, C2, C4, and C3's prediction that the cap is nearly inactive.

**The decision it changes.**

- Stop looking for further decision-time information to skip PowerMH work. On these workloads the certifiable remainder beyond early rejection is 0.08% of rows, and the rest is provably unskippable.
- The exact-acceleration line cannot pass the research question's Gate 1 here, whatever the speedup (§3.3). Exactness copies the policy's answers, and the policy does not beat the cheaper alternative.

**Still short of an ICML Oral claim.**

- The positive mechanism is provably weak in practice. The headline would be a characterization plus a negative result.
- Depth two only, 10 MATH problems, a 1.5B model on one MPS device. No CUDA or server batching, where an extra row is cheaper and speculation's tick gain would show up in wall time.
- No workload yet where the frozen policy delivers correct answers that the cheaper alternative cannot.

**The stronger formulation that survives.** The obstructions bind *trajectory-exact* and first-hit-exact executions in the reveal model. They do not bind executions that must match only the law of the returned answer. There, the coupling lower bound is strictly weaker (certified_thinning §3: up to 63.7% on checked units), and the terminal-interval construction in `mh_output_certificate.md` already beats certificate reading on single steps. Two things would make it more than a curiosity:

- an output-law simulator for a multi-step policy;
- a policy that wins on value.

Neither exists yet.

## 6. Reproduction

```sh
cd experiments/speculative_mh
~/miniconda3/envs/aidemo/bin/python -m pytest -q tests -p no:cacheprovider     # 6 exactness tests, tiny random Qwen2, no weights
zsh ../../results/local_runs/speculative-mh-math10-20261123/run.sh            # MATH arms (about 70 min on M4)
zsh ../../results/local_runs/speculative-mh-math10-20261123/run_direct.sh     # matched q_alpha baseline
~/miniconda3/envs/aidemo/bin/python analyze_math.py                            # paired_math.csv, summary_math.json, frontier_math.png
```

**Files.**

| Path | Contents |
|---|---|
| `spec_engine.py` | Arms, Rule C and the online assertion |
| `prefix_decoder.py` | Shared-prefix batched decoder with per-call timing |
| `run_spec_math.py`, `run_direct_math.py` | Native runners |
| `analyze_math.py` | Analysis |
| `run_spec.py`, `build_fresh_panel.py`, `grade_spec.py`, `analyze_spec.py`, `plot_frontier.py` | MBPP path |
| `tests/test_spec.py` | Exactness tests |

The raw per-run JSON files hold every step's decision, commit tick, call count, wall time and work counters, plus every model call's width, live rows and forward seconds.
