# Execution-localized forks on fresh development tasks

This is the separate Section 7 mechanism, not a rerun of FLEC. Frozen dev24 is
sampled by a published SHA256 ranking from the 105 fresh MBPP+ IDs left after
excluding old uses, reserved IDs and the pilot roster. The existing pilot40 and
HumanEval40 are not evaluated. No efficacy threshold from the old FLEC stage is
inherited.

Every policy sees the same greedy completion and first public assertion only.
Spend at most 192 extra actual positions per policy using KV reuse. Apply the
new router only after public failure and only on code lines visited by the
failing test, nearest in line number to the last visited line, then lower margin.
A line record does not establish where the semantic bug is. Operator-map syntax
failures have no eligible localization. Compare unlocalized operator-forks,
localized confidence-only, and cached whole-program T1/top-p .99 retries.
Missing verification/isolation aborts; lacking a candidate falls back to the
same draft. All 24 are counted, including fallbacks and budget exhaustion.
Offline hidden grading never chooses or prompts a candidate.

Sandbox tracing reuses the trusted verifier worker's test namespace and
candidate filename, with pinned local image, no network, read-only, no new
privileges, no capabilities, and resource limits identical to the existing
grader. Candidate stdout is not a trace or label.

The trusted Docker preflight runs a hand-authored two-line fixture, no model.
Run independently only after one model slot is clear and enough memory remains:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
nice -n 19 ~/miniconda3/envs/aidemo/bin/python experiments/first_answer_localized/run_dev.py \
 --approve --out <fresh-private-run-directory>
```

The original frozen `dev24.json` stays on the experiment device; restore that
exact file before using the checksum or runner in a clean checkout. This is
the concluded development run, not authorization to rerun it or open a holdout.
