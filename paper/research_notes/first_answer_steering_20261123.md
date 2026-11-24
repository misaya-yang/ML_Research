# Steering the first answer at requirement forks: a falsification-first experiment

Date: 2026-11-23. Model: Qwen2.5-1.5B-Instruct on the M4 Max. Code: `experiments/first_answer/`.

**Status.** Built, dry-run, and the development-only Stage 0 replay has been run.

**Outcome (§7).**

- Single-fork headroom exists: 5 of 16 development items.
- Executing the visible test picks the right branch at 9 of 10 decisive forks, where confidence picks it at 4–6.
- The cheap pre-commitment signal carries no information.
- The frozen policy turns this into only +1 item over the likelihood control. **Gate G1 fails, so Stages 1–3 stay unrun.** §7 gives the recommended change of mechanism.

## 1. The claim under test and what does not count

**Goal.** Raise the correctness of the first complete answer, one submitted program per task, at the same total inference budget. The budget counts:

- every model position: prompt prefill, decoding, auxiliary streams, and re-entry after a cache crop;
- every visible-test execution, counted separately.

**Excluded as contributions:**

- faster replay of an existing policy (PowerMH early rejection, PreSTO-style prefetch);
- extra retries or extra samples;
- token-level coverage of a reference continuation.

A policy that spends extra compute counts only if it beats a sequence-level baseline that spends the same compute on samples and uses the same visible check.

**Hypothesis.** It has three parts:

1. Where two plausible continuations implement different task requirements, logit confidence does not say which one is right.
2. A cheap signal grounded in the task does say which.
3. That signal can be applied before the code commits to the wrong operation.

**Decomposition used throughout.** Uncertainty can tell us *where* a decision is open; the hypothesis is about *which* branch to take. The experiment measures the two separately.

## 2. Methods that change the answer distribution vs. methods that only accelerate

| Method | Changes the submitted-answer law? | Signal | Task-grounded? | Available before commitment? | Relation to the claim |
|---|---|---|---|---|---|
| η-sampling (Hewitt et al., EMNLP Findings 2022) | Yes (truncation) | Entropy-scaled probability floor | No | Yes | Distribution sharpener. Included in the Stage 1 tuning grid. |
| Min-P (Nguyen et al., ICLR 2025; critique arXiv:2506.13681) | Yes | Probability relative to the top token | No | Yes | In the tuning grid (top_no port). |
| top_no fixed margin, top-nσ, top-p | Yes | Logit margin or spread | No | Yes | In the tuning grid (top_no port, checked equal to the top_no torch code). |
| top_no frequency offset (ν) | Yes | External token frequency | No | Yes | Excluded. The Phase-0 pilot ended `INSUFFICIENT`, with the shift pointing opposite to the implemented rule, and the frequency table is not on this machine. Token coverage ≠ end-to-end accuracy. The sweep is not restarted. |
| PowerMH (Karan & Du 2025), Power-SMC (arXiv:2602.10273) | Yes (targets p^α) | Model likelihood only | No | n/a (whole sequence) | Today's MBPP pilots: no PowerMH-only solves (Instruct: PowerMH 7/16 vs greedy-then-T1 retries 10/16; Math checkpoint 1/16 vs 3–5/16). |
| Entropy-Cut MH (arXiv:2605.30327) | Same target as PowerMH; a better cut kernel changes the finite-budget law | Entropy-weighted cut points | No | n/a | Mixes faster toward p^α, but the target is still model likelihood. |
| PowerMH early rejection, PreSTO | No (pathwise identical) | Rényi bound | n/a | n/a | Acceleration only. Today: 13.3% (Instruct) and 17.4% (Math) fewer measured positions, with identical chains. Not a first-answer contribution. |
| Verifier/value-guided decoding: PG-TD (Zhang et al., ICLR 2023), execution-guided decoding (Wang et al. 2018), CodeT / Coder-Reviewer reranking (2023) | Yes | Executed tests, generated tests, or p(spec \| code) | Yes | Mostly after completion (rerank) or via expensive tree search | Closest in spirit. Hidden tests are never allowed here; the visible test is. |
| AdaDec (arXiv:2506.08980) | Yes | Entropy pause, then lookahead reranked by likelihood | No (likelihood) | Yes (lookahead) | Prior art for "uncertainty says where". Its *which* signal is confidence, which is exactly our `likelihood` control. |
| UnCert-CoT (arXiv:2503.15341) | Yes | Entropy or top-2 gap at line starts triggers CoT | Partly (model reasoning) | Yes | Prior art for uncertainty-triggered deliberation. |
| USCD (arXiv:2409.05923); CAD (Shi et al., NAACL 2024); CFG for LMs (Sanchez et al., ICML 2024) | Yes | Contrast of prompts with vs. without evidence | Yes (examples) | Yes, cheap | USCD already contrasts prompts with and without I/O examples at uncertain tokens. Our `contrast` arm is a USCD-like reimplementation with the expected value masked in place. It is not a novelty claim. |

AdaDec, UnCert-CoT, USCD and CFG were checked by search today. The other references are from memory.

**What is left open.** Prior work already uses uncertainty to say where; what remains open is which signal picks the branch. AdaDec picks by likelihood and USCD by an example contrast. No report I found measures, at same-prefix forks with hidden labels:

- whether a task-grounded check picks the hidden-correct branch where confidence does not;
- whether doing so beats spending the same compute on whole-sequence samples.

That is the measurement this experiment makes.

## 3. Evidence before this experiment

**PowerMH pilots** (`results/local_runs/powermh-mbpp-{instruct,math}-pilot-*`):

- PowerMH added no first-answer value on MBPP.
- The strongest baseline was greedy followed by T=1 retries, filtered by the visible test.

**Existing development traces** (observational; `results/local_runs/first-answer-trace-headroom-20261123/instruct_dev16.json`). Instruct, MBPP development rows 0–15, greedy correct on 6/16:

- Greedy misses 10 items: 8 fail the visible test (11, 13, 14, 15, 19, 20, 24, 26) and 2 pass it but fail a hidden test (16, 22).
- The sample bank (≈45 programs per item) holds a correct program for 7 of the 10.
- Only 3 of those correct programs differ from greedy in one contiguous span:
  - 11: `replace(ch, '', 1)` vs. a two-sided removal;
  - 14: missing `0.5 *` in the prism volume;
  - 22: returns `None` where `-1` is required. The visible test cannot see this one.
- Task 19 diverges at the first token, the function name, which the policy text says to copy from the public test.
- 20, 24 and 26 are rewrites, not single-fork fixes.
- Of programs that pass the visible test, 101/134 (75%) are fully correct.

The bank was not generated from greedy's prefix, so these divergence counts do not establish that steering at that point would have produced the correct program. Stage 0 measures that directly.

**Implication for the signal.** The only reliable task-grounded signal in this setup is execution of the visible example, and it needs a complete program.

- Before commitment it is available only through lookahead, at a cost of about the remaining program length per alternative.
- The cheap pre-commitment alternative is an example contrast. It is a model-internal proxy and its validity is an empirical question.
- Stage 0 tests both against confidence on the same forks.

## 4. The candidate policy: fork-located example check (`flec`)

**Algorithm.**

1. Decode greedily; this is exactly the direct answer.
2. Execute the visible test on it. If it passes, submit it. This adds no model positions and one execution.
3. Otherwise, list the forks of the greedy trace and visit at most 4 of them in increasing-margin order (uncertainty says where). A fork is a code position outside comments and docstrings where another token has at least ¼ of the top token's probability (log ratio ≤ ln 4).
4. At each fork, force each plausible alternative (at most 2) and complete it greedily from the shared KV cache.
5. Submit the first completion that passes the visible test (the task says which).
6. If none passes within the budget, submit greedy. The budget is 2 × the greedy program's generated length in extra positions.

**Property.** The visible test is one of the grading tests. So flec never loses a task that greedy solves: paired regressions against direct are zero by construction. The informative comparisons are therefore against the two controls that isolate each part of the hypothesis.

| Arm | What it is | What the comparison with flec isolates |
|---|---|---|
| `direct` | Greedy (the harness's single low-temperature answer) | Total effect |
| `plan` | Matched generic planning: ≤3 plan lines then code, one call, +96 tokens allowed | Whether thinking first does as well |
| `sampler` | One draw from the development-tuned sampler (Stage 1: greedy, temperature, Min-P, fixed margin, top-p, top-nσ, η) | Whether sharpening or truncation does as well |
| `best_of_n` | Greedy, then sampler draws, first to pass the visible test, same budget rule | **Where:** fork-located alternatives vs. whole-sequence samples at matched compute |
| `likelihood` | Same trigger, same forks, same completions; branch chosen by mean token log-probability (AdaDec's selection signal) | **Which:** task-grounded check vs. confidence on identical candidates |
| `contrast` | USCD-like: at forks, argmax of lp + (lp − lp_masked), where lp_masked conditions on the visible test with its expected value masked; no execution | Whether a cheap pre-commitment signal suffices |

**"Were correct plans implemented?"**

- `analyze_arms.py` exports every plan without code or outcome to `plans_for_annotation_*.csv` for blind labelling. It then reports the four cells (plan correct/wrong × code correct/wrong); a correct plan with failing code is an implementation gap.
- For flec, it reports the fork line, the greedy token and the chosen token for every changed answer. This shows whether the steer landed on an operation-bearing token.

## 5. Protocol

**Panels** (`experiments/first_answer/panels/`, built by `build_panels.py`, frozen with SHA-256):

- **dev16**: MBPP development rows 0–15 (task_ids 11–26), all 16 references validated. Used for Stage 0 and Stage 1 only.
- **pilot40**: 40 MBPP+ tasks (local EvalPlus copy).
  - The draw excludes 271 MBPP ids touched by any earlier experiment in this repository: development 16, the reserved 48 (task_ids 27–74), the 64-task independent screen, the 80-task Power-SMC panel, and a regex scan of earlier MBPP artefacts.
  - Pool: ids 11–600, 145 tasks; seeded order (20261123); every reference passes its own visible and hidden asserts.
  - Visible = first assert; hidden = the remaining original asserts. The plus suite is secondary, since it needs the Linux numpy image.
- **humaneval40**: separate family. HumanEval tasks whose docstring has at least one doctest example that converts to an assert with a maskable value.
  - Pool 69; seeded order; canonical solutions validated; 2 skipped because a docstring example disagrees with the canonical solution.
  - The doctest examples are the visible evidence; `check` is hidden.
- **Reserved 48**: untouched. Panels never contain them (unit-tested).
- **Six-item sets**: I could not find a "six-task first-answer protocol" under that name. The six-item sets I found are the QW15 WikiText year-task calibration and hold-out sets. Nothing in this experiment reads them. If a different six-task set is meant, its ids need to go into the exclusion list before Stage 2.

**Evidence and leakage.**

- Every arm sees the same task text and visible test; the plan arm adds only the generic plan request.
- The engine decodes every prefill input and refuses to run if it contains any hidden assert (`LeakError`, unit-tested).
- Hidden labels are computed only after every arm has fixed its answer for the item.

**Rewording control.**

- Deterministic framing rewrites ("Write a function to X" → "Implement a function that will X"). Requirement clauses, identifiers and literals are unchanged; the same 40 items are run under both variants.
- This is a weak but clean control: it rules out sensitivity to the exact framing, not to paraphrases of the requirement itself. Hand paraphrases can be added as a third variant before Stage 2.

**Frozen choices.** All of the following were fixed before any model output and none is tuned: fork ratio ln 4, ≤2 alternatives, ≤4 forks, budget 2×, contrast λ = 1, plan allowance 96, cap 192. `configs/FROZEN.sha256` covers code, tests, configs and panels, and every real-model stage verifies it before starting.

**Decision rules (frozen).**

- **Stage 0 gate** (development only; decides whether Stages 1–2 are worth running):
  - G0: at least 3/16 items where greedy is hidden-wrong and some single-fork completion is hidden-correct.
  - G1: simulated flec minus simulated likelihood ≥ 2 items. The simulation is exact: the replay generates the same completions the live policies would.
  - If G0 fails, one-fork steering lacks headroom on this workload; change mechanism or workload rather than tuning cutoffs.
  - If G1 fails, the task-grounded choice does not beat confidence; stop.
- **Stage 2 positive pilot:**
  - flec net paired wins ≥ +3/40 against each of direct, plan, sampler, best_of_n and likelihood;
  - flec's mean positions ≤ 1.15 × best_of_n's;
  - both conditions hold on the base variant and again on the reworded variant.
- **Stage 3 transfer:** net wins ≥ +2/40 against direct, best_of_n and likelihood on HumanEval.

## 6. The smallest result that would change the decision

The deciding result is Stage 0: one greedy pass, about 12 forks and up to 24 branch completions per item on 16 development items, all graded offline. It costs minutes on the Mac. It answers both halves of the hypothesis on same-prefix evidence:

- **G0 fails** (fewer than 3 one-fork fixes in 16): steering at a single fork cannot move this model's first answers on MBPP. The hypothesis may still hold elsewhere, but not through this mechanism. Do not run Stage 2.
- **G0 passes, G1 fails**: the right branch exists, but executing the visible test picks it no better than confidence. The "task-grounded signal" half is falsified for this setup.
- **Both pass**: Stage 2 is worth its compute. It is then decided by best_of_n. If fork-located alternatives do not beat whole-sequence samples at the same budget, the gain was "more samples" and does not count.

## 7. Stage 0 results (development only)

**Run.** `results/local_runs/first-answer-stage0-forks-20261123-092b/` (frozen manifest verified before start).

- 16 MBPP development items.
- Greedy plus 63 eligible forks, at most 12 per item.
- 9,720 model positions in 210 s of model time; the Mac had no other model job.

**Files.** `forks.jsonl` holds every fork, candidate and label; `summary.json` holds the gate. `details.json` is a post-hoc descriptive analysis (from `details.py`), which does not change the gate. The real-model dry run of all seven arms on development tasks 11 and 12 is in `first-answer-realdry-20261123-092b/`.

| Quantity | Result |
|---|---|
| Greedy correct | 6/16 |
| **G0** single-fork oracle: greedy hidden-wrong and some alternative at one fork, completed greedily, hidden-correct | **5/16** (11, 14, 20, 22, 24): pass (≥ 3) |
| Decisive forks (candidates with different hidden labels) | 10, in 9 items |
| Correct pick at decisive forks: greedy token / max mean log-prob / USCD-like contrast / visible-test execution | 4 / 6 / 4 / **9** of 10 |
| Same, restricted to the 6 decisive forks where greedy's branch is wrong | 0 / 3 / 0 net change / **5** |
| Margin at decisive vs. other eligible forks (median) | 0.75 vs. 0.625 nats (no separation) |
| Simulated flec / likelihood / direct (exact replay of the live policies) | 8 / 7 / 6 of 16, at 3,493 / 3,521 / 2,760 positions |
| Same-budget best_of_n from the existing bank (T=1 top-p .99; observational, greedy identical on 12/16 items) | 7/16 |
| **G1** flec − likelihood | **+1**: fail (≥ 2) |

**Frozen decision: stop flec as configured. Stage 1 and Stage 2 are not run.**

**What the forks show.** They come from `details.json`.

- **Requirement forks where confidence is wrong and the task check is right.** These are real but rare:
  - Task 11 has `s.replace(ch, "", 1` at margin 0.375 nats. Confidence keeps the count argument `, 1`; the `)` branch passes every test.
  - Task 24 has `return` vs `decimal`, an algorithm choice.
  - Task 14 has the missing `0.5 *` factor. Here confidence over the full completion also finds the fix.
- **A requirement fork that no execution-based arm can see.** Task 22, `return None` vs `return -1`: greedy passes the visible test, so the check never fires. This is the hypothesis's own blind spot when one example under-specifies the requirement.
- **Decisive forks that are not requirement forks.** In tasks 20, 23 and 25, a `#` comment versus a docstring or code token reroutes the whole remaining program. Uncertainty located them, but they are not "two implementations of different requirements".
- **The policy lost fixes to its budget and its fork order, not to its signal.**
  - Task 24's fix costs 54 positions, but the budget is 2 × 18 tokens, because greedy's wrong program was very short.
  - Task 20's fix is at the fourth fork in margin order, at cumulative cost 280 against a budget of 212.
  - Changing either rule after seeing these outcomes would be the cutoff tuning the plan rules out, so I did not rerun with other values.

**Answer to each part of the hypothesis on development data.**

1. *Logit confidence does not tell which branch is right*: supported at decisive forks (greedy token 4/10; mean log-prob 6/10).
2. *A cheap task-grounded signal does*: only the expensive one does. Visible-test execution after a full greedy lookahead picks correctly 9/10. The cheap pre-commitment signal (example contrast, USCD-like) does no better than greedy. **The informative signal is not cheaply available before commitment.**
3. *Steering converts this into first-answer accuracy at matched budget*: not shown. +2 items over direct and +1 over both the likelihood control and same-budget best_of_n, on 16 development items, is inside noise. The pre-registered gate fails.

**What to do next (recommendation, not run).** Of the two options the plan allows, change the mechanism, not the cutoffs.

- **The bottleneck is *where*, not *which*.** Margin ranks the decisive forks poorly, and comment-token forks dominate the candidate list.
- **Candidate mechanism: execution-localized forks.**
  - Run the failing visible test once with line tracing; this is one CPU execution, already paid for.
  - Restrict forks to operation-bearing tokens (call arguments, operators, literals, return expressions, statement starts) on the lines that the failing test executed.
  - Use an absolute extra budget of one cap-length program (192 positions) instead of a multiple of greedy's length.
- **Method.** Develop it on a fresh development set of 24 MBPP+ tasks taken from the 105 fresh-pool tasks not in pilot40, and keep pilot40 and humaneval40 untouched as the test.
- **What would kill it.** If the fresh development gate again shows the fixes are not reachable within one cap-length of extra positions, single-program steering is not worth pursuing on this model, and the decision should move to workloads with richer visible evidence (HumanEval's multi-example docstrings) or stop.
- **Workload note.** Even a perfect visible check cannot move task 22-type items. On MBPP with one visible assert, the ceiling for any execution-steered first answer is the visible-reachable headroom, about 4 of 16 on development.

## 8. What the experiment can and cannot establish

**Can establish:**

- On development forks, with same-prefix branches and hidden labels used offline only:
  - how often a single fork separates a wrong first answer from a correct one;
  - whether visible-test execution, an example contrast, or confidence picks the correct branch.
- On 40 fresh MBPP+ tasks and 40 HumanEval tasks:
  - whether flec beats direct, generic planning, a tuned sampler, best_of_n and the likelihood control at reported compute, with paired wins and regressions;
  - whether the effect survives a framing rewording.

**Cannot establish:**

- Effects at the 1–2 item level. With 40 items, a net +3 is a pilot signal, not a significance claim. Exact two-sided sign-test p-values are reported for honesty, not as the decision rule.
- Anything about requirements the visible test cannot express. Task 22's `None` vs `-1` is invisible to every execution-based arm, so flec is blind exactly where requirements are under-specified by the example.
- Generality beyond Qwen2.5-1.5B-Instruct, greedy base decoding, short single-function tasks, and one visible test.
- Novelty of the mechanism. Uncertainty-located lookahead (AdaDec), example contrast (USCD) and execution-guided search (PG-TD) all exist. A positive result would be an empirical finding about which signal matters at matched budget, not a new algorithm.
- Wall-clock claims. Positions are the budget; seconds are logged but contended on a shared Mac.

## 9. Reproduction

```sh
zsh experiments/first_answer/reproduce.sh dryrun                # tests + frozen check + scripted-model run of every stage
APPROVE=1 DRY=<dir> zsh experiments/first_answer/reproduce.sh realdry   # real model, all arms, 2 development items
APPROVE=1 zsh experiments/first_answer/reproduce.sh stage0      # fork replay, 16 development items
APPROVE=1 zsh experiments/first_answer/reproduce.sh stage1      # sampler tuning (development)
APPROVE=1 zsh experiments/first_answer/reproduce.sh stage2      # pilot40, base + reworded
APPROVE=1 zsh experiments/first_answer/reproduce.sh stage3      # humaneval40
```

Real-model stages wait for at most one other local model job and at least 10 GB of reclaimable memory.

**Weights.** The only local copy of the Instruct weights is in a macOS temp folder (`/var/folders/.../qwen15-official`, sha256 `dd924a11…`); `~/paper_project/models/Qwen2.5-1.5B-Instruct` has no weights. Copying the file into place would protect reproduction.

**Files.**

| File | Role |
|---|---|
| `common.py` | Tasks, prompts, masking, rewording, leakage guard, ledger, engine, top_no sampler port, graders |
| `policies.py` | The seven arms |
| `fork_replay.py` | Stage 0 replay, exact simulation, gate |
| `run_arms.py` | Stages 1–3 |
| `analyze_arms.py` | Accuracy, paired tables, compute, plan cells, rewording, decision |
| `build_panels.py` | Panel construction |
| `trace_headroom.py` | Observational analysis of existing traces |
| `fake.py` | Scripted model for dry runs |
| `tests/` | 15 tests |
| `configs/` | Frozen configs and `FROZEN.sha256` |
| `panels/` | Frozen panels |
