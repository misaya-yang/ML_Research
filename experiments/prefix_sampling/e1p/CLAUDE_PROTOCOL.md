# E1-P: predict-then-test for correlated sampling (handoff to Codex, 2026-11-25)

**Goal.** Validate one predictor that is computed only from ordinary i.i.d. samples. It should predict the matched-sample gain of lattice correlated sampling over i.i.d. in three regimes with expected gains of large, intermediate and roughly zero. The predictions are frozen before any lattice sample is drawn.

**Machine rules.**
- Run `ps aux | grep -E "envs/.*/bin/python|mlx_env"` before every model job.
- One model job at a time, `nice -n 19`, ≤2 CPU threads, keep ≥6 GB free, MLX width 2.
- Width 4 is allowed only after a 10-minute benchmark shows ≥1.5× throughput with ≥6 GB free.
- Outputs go to `results/local_runs/e1p-20261125/`.

## Fixed design (do not tune)

**Model and sampling**
- Model: `ancestry-budget-20261125/models/qwen3-1.7b-4bit` (MLX), via `ancestry-budget-20261125/mlx_env`.
- Sampling: T = 0.6, top-p 0.95 in every format and arm. The inverse CDF runs over tokens sorted by descending probability after top-p.

**Problems**
- 64 MATH-500 problems with integer gold answers, levels 2–5.
- Choose them by sorting `sha256("e1p-v1/"+unique_id)` and taking the first 64.
- Freeze the list in `problems.json` before generating anything.

**Formats** (same problems, grader and sampling for all three)

| Format | Mode | Prompt suffix | Cap |
|---|---|---|---|
| F1 answer-only | thinking off | "Give only the final answer in \boxed{}." | 24 new tokens |
| F2 brief CoT | thinking off | "Reason briefly, then put the final answer in \boxed{}." | 1,024 |
| F3 thinking | thinking on | "Please reason step by step, and put your final answer within \boxed{}." | 8,192 |

- F3 problems: the 32 problems with the highest F2 i.i.d. accuracy below 1.0. Freeze this list before any F3 generation.
- Grader: integer match on the last `\boxed{}`. A trace that hits the cap without a box counts as wrong; also record `capped = true`.

**Arms**
- `iid`: n = 32 per problem per format for F1/F2; n = 24 for F3.
- `lattice`: groups of k = 8. Point i uses U_i = (bitrev3(i)/8 + Δ) mod 1, with one Δ ~ U[0,1) per group.
  - 4 groups per problem for F1/F2 (32 samples); 3 groups for F3 (24 samples).
  - Bit-reversed order makes points {0,1}, {0..3} and {0..7} exact 2-, 4- and 8-lattices, so pass@2, @4 and @8 come from each group.

**Arithmetic decoding (lattice)**
- Keep a float64 code u per sample.
- At each step, pick j with cdf[j−1] ≤ u < cdf[j], then set u ← (u − cdf[j−1]) / p_j and clamp u below 1.
- Accumulate bits += −log2 p_j. Once bits > 40, replace u with an independent fresh uniform each step, and log `refresh_step`.

**Per-sample logging (all arms)**
- Generated token ids.
- For each step: chosen-token probability, its CDF interval [lo, hi), and the entropy of the truncated distribution.
- Also: finish reason, answer, correct, capped, group id, Δ, point index, seed, and seconds.
- Store per-step arrays in `.npz`, keyed by sample id.

## Predictor (computed from i.i.d. rows only)

1. Build the token prefix tree of the n i.i.d. samples for one problem and format.
2. Truncate it at nodes with fewer than m = 4 samples. A truncated node becomes a leaf with value equal to the success rate of its samples, corrected leave-one-out so that squared values are unbiased.
3. Order children by empirical frequency, with ties broken by token id.
4. Compute exact pass@k (k = 2, 4, 8) on this truncated tree for:
   - lattice coupling, averaged over Δ on a 4,096-point grid;
   - i.i.d. sampling.
   Leaves are independent Bernoulli draws given the leaf reached.
5. Prediction = the difference between the two.
   - This counts only outcome variance settled while lattice points still share a prefix. It assumes no coupling after the samples split. That assumption is the theory's claim under test.
   - Report the normalized gain G/G_max, where G_max = min(1, kμ̂) − [1 − (1 − μ̂)^k]. Use problems with G_max ≥ 0.02.
6. **Validate before use.** Run it on 200 random synthetic trees (depth 4–8, branching 2–6, Dirichlet probabilities) at n = 32, against exact lattice and i.i.d. pass@k computed by enumeration. Report bias and RMSE. If |bias| > 0.01 in pass@8, fix the estimator before freezing anything.

## Order of work today

**P0 — start now, in parallel**
- **A. Model job 1:** i.i.d. banks for F1 and F2, all 64 problems. Estimated ≈1.5–2 h; F1 takes minutes.
- **B. CPU:** implement the lattice decoder and unit-test it on a toy 3-step categorical chain. 10^5 draws per point index; chi-square p > 0.01 for every point index, and empirical pass@k must match the exact value.
- **C. CPU:** implement the predictor and run the synthetic validation (step 6).

**P1 — after A, B and C**
- **D. Real-model exactness check** (model job, ~10 min, F1): 2,000 lattice vs 2,000 i.i.d. draws on 5 problems. Two-sample chi-square on the first token and on the answer; p > 0.01 is required.
- **E.** Compute F1 and F2 predictions. Write `predictions_F1F2.json` and record its SHA-256 in `PROTOCOL_LOG.md` **before** step F.
- **F. Model job 2:** lattice arms for F1 and F2.
- **G. Score F1/F2.**
  - **Gate:** F1 normalized pass@8 gain ≥ 0.30 with a problem-bootstrap 95% CI excluding 0.
  - Predictor within max(30% of predicted, 1 point) of realized, per format and k.
  - Spearman correlation between predicted and realized per-problem gain ≥ 0.5, pooled over F1 and F2.

**P2 — only if the F1 gate passes; overnight**
- **H. Model job 3:** F3 i.i.d. bank, 32 problems × 24 samples (≈3.5–4 h).
- **I.** Compute F3 predictions; write `predictions_F3.json` with its SHA-256 first.
- **J. Model job 4:** F3 lattice arm, 3 groups × 8 (≈3.5–4 h).
- Chain H → I → J in one script that refuses to start J if the predictions file is missing.

**Parallel CPU/network tasks (any time, no model)**
- **K. Thought Anchors offline prediction** for R1-Distill-Qwen-14B and Llama-8B.
  - Download the 162 `*/chunk_0/solutions.json` files from `uzaymacar/math-rollouts` at revision `a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2`. These are i.i.d. rollouts from the bare prompt, about 100+100 per problem.
  - Tokenize with each model's own tokenizer and run the predictor at k = 2, 4, 8, 16.
  - Write the predictions before looking at anything else in these files.
- **L. Archive breadth.** Fetch the remaining Prefix Consistency cells (5 models × 4 benchmarks; `restart-live-20261125/fetch_ranges.py` shows how). Rerun the scripts in `claude-cutoff-restart-20261125/` unchanged and produce one table.

## Metrics to save (`report.json`, plus per-problem CSV)

For each format, arm and k:
- pass@k with a paired problem bootstrap;
- normalized gain;
- predicted vs realized gain;
- realized pairwise co-success excess E[S_i S_j] − μ̂²;
- share of 8-groups with mixed outcomes (relevant to RL);
- split depth of lattice pairs, in tokens and in cumulative surprisal;
- `refresh_step` distribution;
- cap rate.

## Decision rules (pre-registered)

| Outcome | Meaning | Next |
|---|---|---|
| F1 gate fails after D passes | Implementation or theory is wrong in the easy regime | Stop. Debug with the synthetic trees, and replicate QuasiMoTTo's Countdown setup. No F3 run. |
| F1/F2 calibrated; F3 realized normalized gain ≤ 0.10 (CI upper ≤ 0.15) and predicted ≤ 0.05 | The predictive limit holds on this model | Continue: theory write-up, then request GPUs for predict-then-test on 2 larger thinking models |
| F2 or F3 realized gain exceeds prediction beyond tolerance | Coupling after the split (noise rank aligned with value) is not negligible | Revise the claim to be conditional on alignment. Measure alignment with a prefill-only pass over i.i.d. traces before any GPU request. |
| F3 saves ≥ 15% of samples at matched pass@8 (CI excludes 0) | The limit is wrong where it matters | Pivot to a long-CoT correlated-sampling method paper |
| Spearman < 0.3 across F1+F2 problems | No usable predictor | The limits paper loses its main contribution; reassess the direction |
