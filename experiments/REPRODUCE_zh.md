# CPU Reproduction

Run from the project root. Use Python 3; the finite-state checks require NumPy and SciPy. EWT also requires pandas and GCC or Clang (the scripts default to `gcc`). No GPU or API key is needed. The finite-state checks require no data. Before reproducing the EWT results, install both corpus files as described in [the source archive notes](../paper/sources/SOURCE_LINKS.md).

```sh
mkdir -p results/local_runs
run_dir="$(mktemp -d "$PWD/results/local_runs/finite_state-XXXXXX")"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/half_disclosure.py > "$run_dir/half_disclosure.csv"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/half_gate.py > "$run_dir/half_gate.log"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/p_only.py > "$run_dir/p_only.log"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/scale_floor.py > "$run_dir/scale_floor.log"
```

Each EWT invocation creates a separate output directory, under `results/local_runs/` by default. Set `IMH_RUN_DIR` to use a new subdirectory there or a new destination outside the project. The script prints its output location at startup and leaves the archived CSVs and logs intact. The independent reference runs 4,096 chains of length 1,024 per task and is substantially slower than the finite-state checks. For a quicker start, run the existing strong baseline, which can certify budgets using only target draws:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/ewt/baseline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/ewt/sources/pilot.py
```

The [per-task high-mark table](../results/ewt/high_mark/pilot.csv) includes an independent risk reference. The [strong baseline with matched information](../results/ewt/stationary_coupling/baseline.csv) covers the same six EWT documents. Each quota is a separate fixed experiment; comparisons selected after seeing results do not imply simultaneous coverage. Floating-point arithmetic and pseudorandomness are not formalized. For the original Pro scripts, corpora, and frozen protocols, unpack [the source archive](../paper/sources/SOURCE_LINKS.md) in a separate directory. The [small-quota diagnostic](../results/diagnostics/RESULTS.md) is not a main-track win.

The legacy spectral helper `ewt/curve/risk_profile.py` supplies the strong baseline's `filt` dependency. The six EWT tasks do not require running it separately. If the legacy `p.npz` and `q.npz` files are available, pass their directory explicitly with `--input-dir` to write the curve CSV in a separate run directory.
