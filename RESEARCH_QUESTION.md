# ICML 2027: When should a frozen decoder branch or restart?

**Decision date: 2026-11-25.** Follow the Claude audit's new main line. The [previous first-hit acceleration question](paper/research_notes/first_hit_research_question_20261123.md) is retained for provenance, not an active research track. Keep its verified receipts; do not extend native SMC acceleration or CompletionDebt.

## Main question

For a frozen reasoning model and a task with an explicitly specified success readout, when does reusing a partial trace, choosing a different branch depth, or coupling parallel samples buy more correct answers than independent fresh restarts at the same generated-token budget?

Claude's archived Thought Anchors analysis measured median midpoint-resume efficiency of 0.75× (DeepSeek-R1-Distill-Qwen-14B) and 0.54× (DeepSeek-R1-Distill-Llama-8B) on the positive-baseline subset of failed base traces. Those numbers are the measured result for the archived panel and remaining-character cost proxy. A quarter-length restart showed more opportunity. The direction now follows that effect, not an assumed accelerator for a weak mother policy.

## Contract

- Separate selection after a checked failure from blind branching without online outcome feedback. Specify whether an offline grader measures a delivered answer or whether a deployed verifier actually decides acceptance. Never substitute oracle coverage for the performance of a reader without that verifier.
- Compare fresh IID (and a tuned low-temperature baseline), fixed-depth branching/restarts, and marginal-preserving correlated sampling as distinct arms. Use [QuasiMoTTo](https://arxiv.org/abs/2607.01179) as a direct related baseline, not as a claim of novelty for QMC itself.
- Estimate prefix properties on development data only. Record actual completion lengths and failures. Charge prefix preparation, discarded attempts and any verifier. Match budgets using real generated tokens and report a quality–cost curve; do not call a character-cost replay a model-token result.
- For independent retries of deterministic cost, the exact quantity is the failure exponent per unit cost, not simply success probability divided by cost. For coupled retries the relevant joint failure probabilities must be measured or modeled separately.
- Use the M4 for the first decision. Keep the old run evidence and do not interrupt a running old job; do not queue new old-line work. Only scale to GPUs after a local, held-out quality–cost gain against strong alternatives.

Local source: `results/local_runs/restart-headroom-20261124/` (Claude scripts, hashes, revision and original output, intentionally not in Git). [Evidence index](RESEARCH_INDEX.md).
