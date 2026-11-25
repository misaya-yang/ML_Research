# Experiment entry points

Start at [the evidence index](RESEARCH_INDEX.md). The following cold-start commands are historical reproductions, not instructions to start a new run.

**MBPP56 stored receipt:** [stdlib-only replay](results/first_passage/mbpp56_audit/README.md).

`ICML_2027 repository root` · git `https://github.com/misaya-yang/ML_Research.git` (`main`)
Plan: `paper/research_notes/roadmap_zh.md` (also `research_brief_zh.md`, `pro_result_zh.md`, `README.md`). Do not claim method advantage/Oral.
**Experiment tasks (pseudocode, frozen criteria, phases, GPU budget): `experiments/EXPERIMENT_PLAN_zh.md`.** Local status (not in a fresh clone): `results/local_runs/STATUS_zh.md`.
Proofs: `theory/proofs/{three_channels,cold_risk_bound,variance_scaled_floor,small_overlap_joint_frontier,tail_profile_frontier,fixed_target_extensions,fixed_target_power,long_horizon_two_source_frontier,normalization_disclosure_half_variance,normalizer_estimation_black_box,pro_moment_frontier_original}.md`
Code: `experiments/REPRODUCE_zh.md`; `experiments/finite_state/{half_disclosure,half_gate,p_only,scale_floor,cap_equality,soak_floor,jump_cost,risk_bound,risk_bound_refined,variance_floor,small_overlap_floor}.py`; `experiments/ewt/baseline.py`, `experiments/ewt/sources/{pilot,capture,model,runtime_paths}.py`, `draw.c`
Frozen results (never overwrite): `results/{disclosure,ewt,finite_state,diagnostics,pro}/` · new runs: `results/local_runs/` only · stale PDF: `paper/manuscripts/cold_start_audit_proposal_start_draft.pdf`
Missing: `experiments/ewt/data/en_ewt-ud-{train,test}.conllu` (archive: `paper/sources/SOURCE_LINKS.md`); Pro/GUM `dual_round/` not here (`paper/sources/REPRODUCE.md`)
Kaggle T4 only if asked: `~/kaggle_t4_template/` (private, smoke first, `/kaggle/working`, 9h/session, 30h/week, cancel on web)
Run from the repo root with an environment that supplies NumPy, SciPy, pandas and mpmath as required, plus a C compiler for EWT. Verify the interpreter first: the M4 default Homebrew Python and the Jupyter kernel do not currently provide all these dependencies. Do not overwrite existing local environments or frozen results. For the read-only finite-state commands on this M4, the existing `aidemo` interpreter was verified (Python 3.13, NumPy 2.4.2, SciPy 1.17.1, pandas 3.0.1, mpmath 1.3.0).
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY="${PY:-python3}" # on this Mac: PY="$HOME/miniconda3/envs/aidemo/bin/python"
mkdir -p results/local_runs && run_dir="$(mktemp -d "$PWD/results/local_runs/finite_state-XXXXXX")"
for s in half_disclosure.csv half_gate.log p_only.log scale_floor.log; do "$PY" experiments/finite_state/${s%.*}.py > "$run_dir/$s"; done
"$PY" experiments/finite_state/cap_equality.py
# EWT after the two .conllu files are in experiments/ewt/data/:
"$PY" experiments/ewt/baseline.py; "$PY" experiments/ewt/sources/pilot.py
```
One job at a time; OPENBLAS/OMP/MKL/torch threads ≤2; `nice -n 19`; keep ≥6GB RAM free. Unpublished: no public posting/uploads. No commit/branch/push unless the user asks.
