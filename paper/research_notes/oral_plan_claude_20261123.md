# ICML 2027 Oral Plan: Assessment, Targets and Route

Claude, 2026-11-23. **Revision 2**, after the user's review the same day. Revision 1 proposed switching the main line to PowerMH. Per the review, that is withdrawn.

- **Main line unchanged.** It remains [RESEARCH_QUESTION.md](../../RESEARCH_QUESTION.md): exact first passage for a frozen verifier-guided policy.
- **Top-priority experiment.** PowerMH early rejection.
- **Unification.** The existing results are not discarded. They become instances of one theorem: [certified_thinning](../../theory/proofs/certified_thinning.md).

## 0. Bottom line

1. **One framework, existing results in the main text.** Exact first passage by *certified thinning*. Every exact accelerator in this repository does the same thing:
   - pre-draw the policy's own decision randomness (an MH uniform, Gumbel keys, carrier marks);
   - pay for a unit of model work only while a certified cap still lets that unit be delivered.

   Theorem 1 (exactness and cost identity) and Theorem 2 (one coupling lower bound) cover all of the following:
   - PowerMH early rejection;
   - stopped weighted ranking;
   - Power-SMC terminal selection;
   - last-mark and one-sided first passage;
   - IMH with a syntax carrier.

   Theorem 2 is instance-wise where caps are attainable and minimax or order-matching otherwise. Caps come from exactly two free sources: syntax and action **carriers**, and the **score monotonicity** of sharpening samplers.
2. **Why language models get the bound for free.** For a sharpening proposal ($\alpha\ge1$), the per-token factor of the PowerMH ratio is
   $$
   \log Z_\alpha(h)=-(\alpha-1)H_\alpha(p(\cdot\mid h))\le0,
   $$
   where $H_\alpha$ is Rényi entropy. The MH test therefore compares accumulated Rényi uncertainty, and uncertainty only accumulates. A rejection is certain as soon as the proposal's running total exceeds the current suffix's total by $|\log U|/(\alpha-1)$. Flattening samplers ($\alpha<1$) have no such bound.
3. **Novelty, stated precisely.**
   - *Exactness.* Early rejection preserves every accept/reject decision and every chain state under the same proposal and random numbers.
   - *Optimality.* It holds only in the access model that reveals one token at a time at additive prices and requires the law of the whole trajectory. Any simulator exact for that law is bound, including one that recouples randomness; seed-identical replay is not required.
   - *Outside that model.* Nothing is claimed for final-answer-only contracts (there the bound is strictly weaker on 270/270 checked units), for batched scoring priced per call, or for white-box access.
   - *Prior art.* Early rejection, thinning and lazy Gumbel search.
   - *What is worth writing.* The free Rényi certificate, the single read-rule theorem with its boundaries, and the measured saving on top of strong exact baselines.
4. **Experiment order (the user's design):**
   - **R0:** replay complete PowerMH logs and find each rejection's earliest deciding token. This gives the headroom before any batching work.
   - Then, with the same model on the same held-out problems:
     - **Gate 1:** composed with PreSTO, counting cache and batching overhead, there is still a measured speedup.
     - **Gate 2:** at the same real budget, accuracy is good enough that a user would choose it over low-temperature sampling or Power-SMC.
   - Passing only Gate 1 gives a valuable exact speedup, not the oral-level claim. **No arXiv until both gates are checked.**
5. **The hardest gate is Gate 2, so test it cheaply first.** [Power-SMC](https://arxiv.org/abs/2602.10273) reports matching or exceeding MH power sampling at 1.4–3.3× the latency of plain decoding, against 16–28× for MH. These figures come from its abstract as mirrored by an aggregator and still need checking against the paper. A 2× exact speedup alone would not close that gap. A **Gate 2 feasibility check (R0b)** on the R0 logs places PowerMH at its best-case accelerated cost on the same accuracy–cost plane as low-temperature sampling and Power-SMC. It should run before any PreSTO engineering.
6. **Odds (judgment).**
   - Oral: about 5–10% if R0 shows large headroom and both gates pass. About 2% for a theory-plus-instances paper with modest constants.
   - Accept: about 35–50% in either case if the general theorem survives a human check.

## 1. Assessment of existing assets

They are real results that did not clear every bar. In the new framework each one is an instance.

| Asset | What it establishes | What it did not establish | Role in the framework |
|---|---|---|---|
| One-sided terminal step and last-mark trimming, with an address-wise minimax bound $1-\prod_{j\ge i}(1-c_j)$ | Exact first passage with a matching lower bound under carrier caps, including adaptive (miss-dependent) policies | A workload with large suffix savings: the coloring cap is 0.961 calls | M1 instance of Theorems 1 and 2 (minimax) |
| Qwen IMH year task: 60/60 identical outputs and verifier records, 14.90% fewer rows than the strong baseline, CI [9.63%, 20.20%] | Exact jump over carrier misses on held-out items, on top of a baseline that already removes 97.5% of chronological work | The prespecified 15% gate (missed by 0.1 point); task value (a source rule solved the six dev items) | M0 carrier instance. Its attainability, and hence the bound type, is still open |
| MBPP Power-SMC, 56 tasks: 15.43% fewer input positions, identical verification paths and 35/6/15 in all three arms | Exact certificates on a finite particle policy with adaptive resampling | A population-level gain over strong direct retries at equal work. The 4 s and 8 s deadline rows in [forward_search](../../results/first_passage/forward_search.md) show individual deliveries, not an aggregate win | The terminal part is an instance-optimal race. The across-checkpoint part is exact, but its 27.62% witness marks a **boundary** of the theorem |
| Stopped weighted ranking: exhaustive 8,640 cases | Instance-wise read-probability optimality | Large measured savings (27 positions on 4 dev tasks) | Prototype of Theorem 2 |
| Forward witness and reverse-kernel blocks | Exact stationary-block first passage | Matching bounds beyond two-step blocks | Order-matching instance |
| Cold-start IMH audit line | Separate theory (two-source frontier, Γ bound) | — | Not part of this paper; separate venue |

## 2. Theory: the single framework

The full statements and proofs are in [certified_thinning.md](../../theory/proofs/certified_thinning.md).

**Theorem 1 (exactness and cost identity).** The read rule $R^*$ is: pay for unit $u$ iff $u$ is reached and the pre-drawn coins lie in the certified cap event $\Gamma_u$. It is pathwise ledger-exact when two conditions hold:

- deliveries are contained in cap events;
- skipped work cannot affect the ledger. That is (M0) miss-oblivious updates, or, under (M1), stopping only when no future delivery is possible.

Its expected cost is $\sum_u a(u)\Pr(E_u)$. The saving over any baseline is $\sum_u a(u)[\Pr(\text{baseline pays }u)-\Pr(E_u)]$, which a replay of complete logs measures directly.

**Theorem 2 (necessity).** If the cap is attainable at $u$, every ledger-exact simulator pays for $u$ with probability at least $\Pr(E_u)$, and $R^*$ attains it. Attainable means some admissible world agrees on everything before $u$ and delivers a new ledger entry on $E_u$. If attainability holds at every unit of an instance, the bound is instance-wise; if only on a hard subfamily, it is minimax.

**Boundaries (each with a witness):**

- final-answer-only contracts: 270/270 strict gaps, up to 63.7%;
- streams coupled through normalization: the SMC 27.62% witness;
- batched or white-box cost models.

| ID | Task | Status | Done when | Week |
|---|---|---|---|---|
| T1 | PowerMH instance ([lazy_power_mh](../../theory/proofs/lazy_power_mh.md)) | Proved, machine-checked, re-derived by the user | Extend to Karan & Du's variable-length block implementation and top-$p$ proposals. The algebra is unchanged and partial sums stay monotone | W1 |
| T2 | Certified-thinning theorem | Drafted today | Human line-by-line check. One checker harness covering MH, rejection sampling and the race | W1–W3 |
| T3 | Carrier attainability for the IMH syntax-carrier instance | Open | Instance-wise bound proved, or the gap order stated | W2–W3 |
| T4 | Output-only contract: a cheaper simulator, or optimality for one MH step | Open, with 270/270 evidence | Either theorem, or keep it as an open problem with the evidence | W3–W4 (desirable) |
| T5 | Restricted matching bound for across-checkpoint SMC | Open | Bound within a stated family, or a clean boundary statement | W4 (desirable) |
| T6 | Rényi-profile prediction of headroom: the saving per rejection as a function of where $H_\alpha$ accumulates | New | Formula checked against R0 logs | W2 |
| T7 | Human verification of every main-text proof | — | Checker name and date recorded in each file | W6–W7 |

## 3. Experiments

General rules from [EXPERIMENTS_README](../../EXPERIMENTS_README.md) apply: freeze configs and SHA-256 before running; write only to `results/local_runs/`; one job at a time; report failures as observed. Because early rejection is pathwise identical, every PowerMH comparison is paired per problem and checked bit for bit.

### R0: replay headroom (first; development rows)

- **Logger.** [powermh_lazy_pilot.py](../../experiments/first_passage/powermh_lazy_pilot.py) runs the eager PowerMH chain and writes a complete log: initial tokens and per-position $\log Z$, and per step the cut, $\log U$, proposal tokens, per-position $\log Z$ and decode seconds.
- **Replay.** [replay_powermh_headroom.py](../../experiments/first_passage/replay_powermh_headroom.py) rebuilds the chain from the logs alone and recomputes every decision (mismatches are reported). For each rejection it finds the earliest decode position at which the rejection was already certain, and sums the decode positions computed after it. It counts two ways:
  - raw;
  - prefix-cache-aware, with separate caches for eager and lazy, so headroom is measured on top of an engine that already gets cached prefixes free.

  It also reports:
  - the distribution of the decided fraction per rejection;
  - per-problem headroom;
  - best-case seconds saved;
  - the numerics of $\log Z$ at the cut;
  - low-temperature accuracy, which is free because the initial state is a temperature-$1/\alpha$ sample.
- **Validation so far.** Both scripts pass a mocked end-to-end test: zero mismatches, final states rebuilt, and counts equal to the in-run tally. No real-model numbers yet.
- **Setup.**
  - Qwen2.5-Math-1.5B with $\alpha=4$, $T=512$, $K=16$ on MATH-500 rows 0–19.
  - Sensitivity runs at $\alpha\in\{2,8\}$ and $T=1024$ on rows 0–9.
  - Before held-out runs, also log Karan & Du's exact released configuration on a few rows.
- **Command.** In [§5](#5-done-today-and-blockers).

### R0b: Gate 2 feasibility check (same development rows, before any PreSTO work)

- **Arms.** On the R0 rows, measure on the same machine and engine:
  - low-temperature sampling: free from the R0 logs;
  - Power-SMC at the published particle counts: reuse and adapt the repository's `powersmc_mbpp.py` to MATH;
  - PowerMH at several $K$.
- **What to compare.** Place PowerMH at its eager cost and at its best-case accelerated cost: eager minus R0 headroom, optionally divided by PreSTO's published median 1.71×.
- **Decision.** If PowerMH stays off the accuracy–cost frontier even at the best case, Gate 2 cannot pass for PowerMH. In that case:
  - say so;
  - keep the exact-speedup result as a Gate 1 contribution;
  - test Gate 2 instead on the policy the framework also accelerates (exact Power-SMC certificates), using the strongest available implementation of each policy as the baseline.

### Gate 1: exact speedup on top of PreSTO (held-out, frozen)

- **Setup.** Same model and frozen held-out problems as Gate 2. Freeze Karan & Du's released policy exactly, plus the fixed-horizon variant if used. Do not alter the policy to fit the theorem.
- **Arms**, all in one engine with identical numerics:
  - PreSTO-style prefetch, plus PreSTO's code if released;
  - prefetch plus early rejection, where prefetched proposals are also cut early and their batch slots freed.
- **Accounting.** Prefill, prefix-cache hits and misses, batch padding and wasted speculative tokens all count.
- **Pass.** Trajectories bit-identical on 100% of problems, and a measured wall-clock speedup with a paired CI lower bound above 1.0. The effect size is pre-registered from R0.

### Gate 2: user value at the same real budget (held-out, frozen)

- **Budget.** Matched per-problem wall-clock on the same machine.
- **Arms.**
  - accelerated PowerMH, at the largest $K$ the budget allows;
  - low-temperature sampling;
  - Power-SMC, in its strongest available exact implementation.
- **Benchmarks.** MATH500, HumanEval and GPQA-Diamond, held-out subsets sized to budget.
- **Pass.** Accelerated PowerMH is on the accuracy–cost frontier on at least one benchmark, with a paired CI. Results are reported whatever the outcome.

### Supporting instances (main text, not appendix)

- **Measured read frequency against cap prediction.** For the MBPP Power-SMC terminal certificates, the weighted ranking, and the Qwen IMH carrier, rerun the replay identity on existing receipts. No new GPU work.
- **Figure.** Saving as a function of the cap source: carrier mass for IMH, Rényi profile for PowerMH, weight ceiling for ranking.

### Compute

- **Local M4 Max.** Shared with the experiment agent. It ran Qwen jobs back to back all session, so R0 needs a scheduled slot or Kaggle permission.
- **Kaggle.** T4 at ≤30 h/week.

| Run | Rough cost |
|---|---|
| R0 | About 20–40 min on MPS |
| R0b Power-SMC arm | About the same again |
| Gate 1 and Gate 2 at 1.5B | Kaggle T4. About 10–15k tokens per problem for single-block $n=1024$, $K=10$–$30$. At ~400 tok/s aggregate batched, ~5 h per 500 problems per eager arm |

## 4. Paper structure (8 pages)

| Section | Content |
|---|---|
| 1 Intro | At finite budgets the sampler is the policy; only exact acceleration is quality-neutral. Fig 1: the read rule, and where carrier and score caps come from |
| 2 Setting | Frozen first-passage policies; ledger contract; trials, coins and certified caps; the M0 and M1 regimes |
| 3 Theory | Theorem 1 (exactness and cost identity); Theorem 2 (necessity under attainability); the Rényi certificate; boundaries with witnesses; table of instances |
| 4 Experiments | PowerMH: R0 headroom, Gate 1, Gate 2. Existing instances: Qwen IMH 14.90%, MBPP Power-SMC 15.43%, ranking. Measured savings against cap predictions |
| 5 Related work | Thinning, early rejection, delayed acceptance, A* and Gumbel sampling, Fagin-style threshold algorithms (deterministic instance optimality, versus our distributional version), speculative decoding, PreSTO, Power-SMC, speculative rejection |
| Appendix | Full proofs, checkers, label-error certification, sensitivity runs |

## 5. Done today and blockers

**Theory.**
- [lazy_power_mh.md](../../theory/proofs/lazy_power_mh.md): PowerMH instance, now with the precise scope paragraph and the Rényi form.
- [certified_thinning.md](../../theory/proofs/certified_thinning.md): the unified theorem, cap sources, the instance table with honest bound types, and the novelty boundaries.

**Checker.** [check_lazy_mh.py](../../experiments/first_passage/check_lazy_mh.py) passes in exact rationals:
- 33 models, 270 units;
- read probability equals trajectory TV on every unit;
- final-state TV is strictly smaller on all 270 units;
- 0 pathwise mismatches.

The three existing first-passage checkers also pass.

**R0 tooling.** The logger and replay scripts described above, both passing the mocked end-to-end test.

**Blocker.** R0 has not run on a real model. The experiment agent ran Qwen jobs on this machine all session (`fixed_verify_k.py`, `human_study.py`, then another), and Kaggle needs the user's request. The Qwen2.5-Math-1.5B download was stopped at about 1.3 of 3.1 GB to free `/restart`; Hugging Face resumes from the partial file. MATH-500 `test.jsonl` is cached. Command, from the repo root, when no other job runs:

```sh
HF_HUB_DISABLE_XET=1 ~/miniconda3/envs/aidemo/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-Math-1.5B', allow_patterns=['*.json','*.safetensors','*.txt'])"
data=$(~/miniconda3/envs/aidemo/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('HuggingFaceH4/MATH-500','test.jsonl',repo_type='dataset'))")
run="results/local_runs/powermh-R0-$(date +%Y%m%d)-$(openssl rand -hex 2)"
OMP_NUM_THREADS=2 nice -n 19 ~/miniconda3/envs/aidemo/bin/python experiments/first_passage/powermh_lazy_pilot.py --out "$run" --data "$data" --rows 0:20 --alpha 4 --T 512 --K 16
python3 experiments/first_passage/replay_powermh_headroom.py "$run"
```

## 6. Timeline (deadline expected late January 2027; confirm on the official site)

| Week | Dates | Theory | Experiments | Checkpoint |
|---|---|---|---|---|
| W1 | 11/23–11/29 | T1; T2 human check starts | R0 run and replay; freeze Karan & Du's code | R0 headroom in hand |
| W2 | 11/30–12/06 | T2 proof final; T6 | R0b Gate 2 feasibility | **Go/no-go for PowerMH as the Gate 2 candidate, 12/04** |
| W3 | 12/07–12/13 | T3 | Prefetch plus early-rejection engine; Kaggle smoke; freeze held-out | — |
| W4 | 12/14–12/20 | T4 | Gate 1 held-out | **Gate 1** |
| W5 | 12/21–12/27 | T5 | Gate 2 held-out | **Gate 2**; holiday buffer |
| W6 | 12/28–01/03 | T7 starts | Supporting instances under the unified replay | Freeze figures; **arXiv decision** |
| W7–W9 | 01/04–01/24 | T7 | Ablations, reproduction bundle | Draft, mock review, submit |

## 7. Gates and fallbacks

| Gate | Pass | If it fails |
|---|---|---|
| R0 headroom | Large enough to matter after overhead (threshold set from R0 itself, before held-out) | PowerMH is reported as a measured instance; the paper leads with the framework and existing instances, aiming for a solid accept |
| R0b feasibility | Best-case accelerated PowerMH reaches the accuracy–cost frontier on development rows | Keep the exact speedup (Gate 1 track); try Gate 2 with exact Power-SMC; no oral-level claim for PowerMH |
| Gate 1 | Bit-identical; paired wall-clock CI above 1.0 over PreSTO alone | Report tokens and calls only; no wall-clock claim |
| Gate 2 | Accelerated PowerMH on the frontier on at least one benchmark at matched real budget | Valuable exact speedup, not oral-level; say so |
| arXiv | Both gates checked | Hold |
| Scoop watch (weekly) | No exact early-rejection PowerMH paper | Lead with Theorems 1–2, the Rényi certificate and the boundaries |

## 8. Risk register

| Risk | Mitigation |
|---|---|
| The lower bound is called trivial | It binds all ledger-exact simulators, including recoupling ones. Its value is the unification and the boundaries where it fails (output contract, normalization coupling, batching), each with a witness |
| Power-SMC dominates PowerMH at matched budget | R0b tests this early; fall back as in §7 |
| Engine fairness | One engine, identical numerics; tokens, calls and wall-clock reported separately; no reconstructed timings |
| Bit identity broken by numerics | Position-indexed float64 sampling on CPU; fixed row shapes; mismatches reported, not hidden. R0 logs the $\log Z$ discrepancy at the cut (prefill versus incremental decode) |
| Machine contention | Agree a slot with the experiment agent, or use Kaggle |

## 9. Decisions for the user

1. **R0 compute.** A local slot coordinated with the experiment agent, or permission for one Kaggle T4 session.
2. **Gate 2 baselines.** Should "Power-SMC" mean its strongest available exact implementation, including this framework's certificates? I recommend yes, as the stronger and fairer baseline.
3. **Paper framing.** Is the certified-thinning framing acceptable as the paper's theoretical spine under the unchanged research question? If yes, I will add one paragraph to RESEARCH_QUESTION.md pointing to it. No other change.

## Sources

- [Karan & Du, Reasoning with Sampling (arXiv 2510.14901)](https://arxiv.org/abs/2510.14901)
- [PreSTO (arXiv 2610.13171)](https://arxiv.org/abs/2610.13171)
- [Parallel Power Tempering (arXiv 2609.38104)](https://arxiv.org/abs/2609.38104)
- [Power-SMC (arXiv 2602.10273)](https://arxiv.org/abs/2602.10273)
- [Speculative rejection (arXiv 2410.20290)](https://arxiv.org/abs/2410.20290)
- [Solonen et al. 2012, early rejection](https://doi.org/10.1214/12-BA724)
- [Lewis & Shedler 1979, thinning](https://doi.org/10.1002/nav.3800260304)
- [Banterle et al., delayed acceptance (arXiv 1503.00996)](https://arxiv.org/abs/1503.00996)
