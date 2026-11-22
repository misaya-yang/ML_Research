# EXPERIMENTS_README.md — Cold-Start IMH Audit
`ICML_2027 repository root` · git `https://github.com/misaya-yang/ML_Research.git` (`main`)
Plan: `paper/research_notes/roadmap_zh.md` (also `research_brief_zh.md`, `pro_result_zh.md`, `README.md`). Do not claim method advantage/Oral.
**Experiment tasks (pseudocode, frozen criteria, phases, GPU budget): `experiments/EXPERIMENT_PLAN_zh.md`.** Status file: `results/local_runs/STATUS_zh.md`.
Proofs: `theory/proofs/{three_channels,cold_risk_bound,variance_scaled_floor,small_overlap_joint_frontier,tail_profile_frontier,fixed_target_extensions,fixed_target_power,long_horizon_two_source_frontier,normalization_disclosure_half_variance,normalizer_estimation_black_box,pro_moment_frontier_original}.md`
Code: `experiments/REPRODUCE_zh.md`; `experiments/finite_state/{half_disclosure,half_gate,p_only,scale_floor,cap_equality,soak_floor,jump_cost,risk_bound,risk_bound_refined,variance_floor,small_overlap_floor}.py`; `experiments/ewt/baseline.py`, `experiments/ewt/sources/{pilot,capture,model,runtime_paths}.py`, `draw.c`
Frozen results (never overwrite): `results/{disclosure,ewt,finite_state,diagnostics,pro}/` · new runs: `results/local_runs/` only · stale PDF: `paper/manuscripts/cold_start_audit_proposal_start_draft.pdf`
Missing: `experiments/ewt/data/en_ewt-ud-{train,test}.conllu` (archive: `paper/sources/SOURCE_LINKS.md`); Pro/GUM `dual_round/` not here (`paper/sources/REPRODUCE.md`)
Kaggle T4 only if asked: `~/kaggle_t4_template/` (private, smoke first, `/kaggle/working`, 9h/session, 30h/week, cancel on web)
Run from repo root (python3 3.9.6 + numpy/scipy/pandas/mpmath + gcc already here; no installs unless asked):
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p results/local_runs && run_dir="$(mktemp -d "$PWD/results/local_runs/finite_state-XXXXXX")"
for s in half_disclosure.csv half_gate.log p_only.log scale_floor.log; do python3 experiments/finite_state/${s%.*}.py > "$run_dir/$s"; done
python3 experiments/finite_state/cap_equality.py
# EWT after the two .conllu files are in experiments/ewt/data/:
python3 experiments/ewt/baseline.py; python3 experiments/ewt/sources/pilot.py
```
One job at a time; OPENBLAS/OMP/MKL/torch threads ≤2; `nice -n 19`; keep ≥6GB RAM free. Unpublished: no public posting/uploads. No commit/branch/push unless the user asks.
