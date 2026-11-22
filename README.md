# Cold-Start IMH Audit

Living research workspace for error guarantees on a fixed proposal-start IMH chain. Not a submission snapshot.

## Start here

- [Current research brief (中文)](paper/research_notes/research_brief_zh.md)
- [ICML 2027 roadmap (中文)](paper/research_notes/roadmap_zh.md)
- [Three-channel synthesis and paper-level theorems](theory/proofs/three_channels.md)
- [Experiment plan for the experiment agent (中文)](experiments/EXPERIMENT_PLAN_zh.md)
- [Fixed-target pointwise power and the equality boundary](theory/proofs/fixed_target_power.md)
- [Long-window finite-budget frontier](theory/proofs/long_horizon_two_source_frontier.md)
- [Same-chain disclosure results and white-box controls](results/disclosure/README.md)
- [CPU reproduction](experiments/REPRODUCE_zh.md) and [cap-equality finite check](experiments/finite_state/cap_equality.py) (mpmath)
- [Manuscript versions](paper/manuscripts/README.md)

## Layout

- theory/: proofs and related work.
- experiments/: CPU code, input data references and run instructions.
- results/: paired ledgers and diagnostics by topic. Replays must not overwrite frozen output.
- paper/: manuscripts, current research notes and source links.

Keep one current note and evidence table per topic. Do not make copies of the whole project or parallel date-stamped versions.
