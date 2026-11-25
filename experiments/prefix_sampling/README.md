# Reproduce the prefix-sampling studies

The checkpoints and receipts are described in [the evidence table](../../results/prefix_sampling/README.md). The [original Claude E1-P protocol](e1p/CLAUDE_PROTOCOL.md) retains the historical experimental rules, with machine-local process IDs omitted. These scripts are research snapshots, not a production decoder.

## CPU only, no checkpoint or download

From the repository root, use Python 3.10+ with NumPy and SciPy:
```sh
WORK=$(mktemp -d)
cp experiments/prefix_sampling/e1p/*.py "$WORK/"
cp results/prefix_sampling/e1p/countdown_control* "$WORK/"
(
  cd "$WORK"
  python3 verify_countdown_control.py
  python3 e1p_lattice.py
  python3 e1p_predictor.py validate
  python3 countdown_predictor_diagnostic.py
)
python3 experiments/prefix_sampling/transfer/transfer_report.py \
  results/prefix_sampling/transfer/live_compact.jsonl
```
The verifier regrades the small local arithmetic control without MLX. The transfer script recomputes endpoints/costs from compact receipts, not the model generations. The CPU toy validates approximate numerical implementation at its tested tolerance, not machine-exact marginals or predictor performance on reasoning.

## Reproduce archive replays

- **Thought Anchors**: stage the labeled prefix metadata for the two model families from [math-rollouts](https://huggingface.co/datasets/uzaymacar/math-rollouts) at the revision in [dataset_revision.txt](../../results/prefix_sampling/archive/dataset_revision.txt). Preserve its model / temperature / correct-or-incorrect base solution / problem structure. The [input hash list](../../results/prefix_sampling/archive/input_sha256.txt) names the exact files. Run `archive/restart_headroom.py DATA`, `archive/robust_oracle.py DATA`, `archive/budget_replay.py DATA` and `archive/early_variance.py DATA`.
- **Prefix Consistency**: obtain the specific members of the [authors' archive](https://doi.org/10.5281/zenodo.20082164); manifest [here](../../results/prefix_sampling/archive/prefix_consistency_input_hashes.json). Match the folder layout in the manifest: the three original gpt-oss-20b AIME cut files and its init file are under `external_aime/`; the other AIME cells are under `external_multi/<archive cell>/`; BRUMO/HMMT under `external_ministral_transfer/<archive cell>/`. From the checkout, pass this staged root to `archive/cutoff_replay.py`, `archive/exclusion_bound.py`, `archive/inheritance.py`, and `archive/inheritance_by_failure_type.py`. Run `archive/sanity_and_within.py` from its script directory with an absolute data path (it loads the adjacent cutoff script). Run `transfer/capped_restart_strong.py`, `transfer/decompose_route.py` and `transfer/ministral_strong_cross_task.py` from the staged archive root. Archived generation records are not redistributed here.
- `archive/check_coupling_identity.py` numerically checks the pairwise result under shared-prefix re-stratification. The toy is not testing the sequence-level lattice.

## Model runs (Mac / MLX)

E1-P uses the cached Qwen3-1.7B 4-bit revision pinned in `e1p_sampling.py`. Run its scripts from an isolated ignored directory, for example `results/local_runs/e1p-repro/`, after copying the `e1p/*.py` files there. Provide the original MATH500 JSONL and existing checkpoint at the relative `SRC` and `MODEL` paths in the script (or adapt those paths in the isolated copy). Run `python3 e1p_sampling.py freeze` before model collection and compare the frozen manifest hash and selected IDs against [the receipt](../../results/prefix_sampling/e1p/problem_ids.json). On the original machine those paths are already staged. Use the isolated MLX environment and one model process only. Follow the protocol's ordering; do not bypass prediction seals or reuse incomplete F1 as successful calibration. Keep all fresh raw banks, runtime logs and per-token arrays under the ignored run directory. The [F1 audit](e1p/audit_f1_execution.py) and [native comparator](e1p/diagnose_f1_native.py) require the original F1 bank.
