# Independent extraction check

The archive was extracted to a fresh project directory. The historical score bank was absent at extraction. Compiled libraries were recreated from C source either in the fresh experiment directory or a new TMPDIR. Commands ran with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, no project PYTHONPATH, and no EWT_DATA override. The offline Python environment included python-flint in the user's installed package directory. Stored files in the provisional ZIP were distinct from rerun outputs. Each match in this table is a byte comparison; for gzip the compared payload is the decompressed CSV.

| Rebuilt from the extracted archive | Checks against archived outputs |
|---|---|
| Exact proof-construction check and rational fixed-gate smoke, math/verify.py and math/rational_gate.py | Both outputs matched, three cases in each |
| Exact finite controls, finite/run.py | 8 exact-risk rows, 320 summary rows and all 15,360 replicate rows matched |
| Least-information internal check, finite/band.py | All 3 CSV outputs matched, including all 13 design rows |
| Frozen GUM selection and score capture, freeze_ids.py and prepare.py | Frozen JSON and all 24 source NPZ files matched |
| GUM primary audit, audit.py, evaluate.py and score.py | issues.csv, truth.csv, runs.csv and summary.csv matched |
| GUM oracle and fixed-time diagnostics, diagnose.py, component_check.py and terminal_check.py | 32 gating rows, 64 component rows and 128 terminal rows matched |
| Independent Arb numerical reference, reference_interval.py | All 16 outward interval pairs and cutoff verdicts matched |
| Historical bank inputs, price_round/banks/make_banks.py | 32 pairs of P-identity/Q-stream digests matched |
| Historical failure ablation, panel/old_ablation.py | All 128 oracle-gift rows matched |

The GUM panel contains eight new documents and 16 safe windows. The extraction check is a software reproducibility check on the same data. The historical 8,192-path Monte Carlo reference was reused from the archive. Regenerating the historical bank is a different operation and was tested. The final manuscript and the reference index can be edited without changing these experiment results.

A separate fresh extraction also reran the post-panel blind in-band experiment. Its safe/unsafe per-budget summary, all decompressed per-replication rows, and the generated main-table.csv matched the working records byte for byte.
