# Scale disclosure and telemetry

A near-threshold stress set: 16 prespecified learned HMM sequence tasks from eight documents, 8 reference-safe and 8 reference-unsafe. Same fixed chain and physical caps in the primary arms.

| Access / comparator | Safe issues (of 8) | Unsafe issues (of 8) |
|---|---:|---:|
| Direct RB8 replay | 3 | 0 |
| Opaque accepted P, class bound plus replay fallback | 6 | 0 |
| Attempt counts | 6 | 0 |
| Rejected labels/scores | 7 | 0 |
| Supplied scale interval (factor 1.05) | 8 | 0 |
| Supplied exact scale | 8 | 0 |
| Model-aware raw P+Q, secondary | 8 | 0 |
| Public-structure HMM dynamic program, secondary | 8 | 0 |

The stronger raw comparator restores all eight safe issues with 78,592 additional Q calls on those eight. The external interval is unpriced. [Paired decision ledger](paired_decisions.csv) reports bought prefixes, not full pre-generation cost or wall-clock. The white-box method decides all 16 correctly without full-sequence generation or scoring calls; it uses exposed factors and a public low-order reward. This does not show that practical learned models require independent target samples. Zero false issues among eight unsafe tasks is not enough to estimate a 5% rate.

Pro's additional reproduction archive includes 49 GUM tasks solved with structured named scoring and no independent target samples. The only verified TinyStories issue was matched by a p=0 baseline with public normalization. These are scope controls, not deployment wins.


## High-order learned sequence stress

The later CPU stress used a learned 8-way proposal and a trained full-sequence nonlinear reward on seven tags (8^7 possible sequences), with genuinely nonzero third-order interactions in the actual score tiers. Six frozen short-window cells: three safe and three unsafe. Under the scored-sample interface, raw-score ratio calibration and the exact-normalizer arm both signed the same three safe cells at the same bought prefixes (75,868 calls total across those cells, including target generation). A Q-only raw-score variant with no target samples also signed all three, at 123,648 proposal calls. None signed an unsafe cell.

A stronger interface exposes the proposal factors and permits named scores. Enumerating only high-probability sequences, bounding their known complementary mass and using the quadratic half-variance gate, a zero-target method decided all six after 64–512 named scores per cell (1,472 stopped-prefix calls total). The run physically precomputed 16,384 names per cell for replay; these counts are not wall-clock speedups. This is a disclosed-structure control on the same small panel, not another independent validation.

Interpretation: a large nominal space and a high-order reward did not make the proposal hard to cover. The public pointwise score cap S≤1 is stronger than an upper bound on the total normalizer Z alone, so this is not a counterexample to the opaque fixed-target-power theorem. Any next comparison must first measure how much probability is in an affordable proposal head.

## Real-model controls after the initial pilots

- [Raw tilted TinyStories reanalysis](#raw-tilted-tinystories) screened more diffuse proposals and did not demonstrate a benefit from paying to disclose the scale.
- On Pro's Qwen long-prefix panel, prior short top-2 depth-10 tasks at 256 prefix queries yielded 1/9/12 signatures (replay / same-cache structure / active), all from two of four contexts. On four top-10 depth-32 contexts, frozen replays sign 0/48; width and depth both changed, and windows are not independent. Two opposite-risk completions per context preserve disclosed Q, Z, mean and variance, showing ambiguity for those transcripts, not that the actual Qwen instances are unsafe.
- A public nucleus retained-mass floor 0.95 gives an analytic IMH MSE ratio upper bound 0.45755 for 16 steps and length 32. It is not a deployment win over alternate estimators: at the same envelope, capped independent rejection with the mean of all accepted draws, or the mean of all rejected draws if none are accepted, attains a worst normalized MSE of 0.48787 in 12 scored proposals. Rejected fallback is not an exact target draw.
- Pro's latest score-information artifact freezes 200 safe and 200 unsafe four-choice cases not already covered by the zero-data bound. At 512 proposal draws 120 safe are signed and there are zero unsafe signatures; a single forward with all four scores available correctly decides all 400. In open-vocabulary GSM8K, both 32 accepted targets plus 128 Q rows and recording all paid acquisition candidates signed 114/127 strictly safe windows, with two unique signatures per arm and zero among 84 reference-unsafe windows. When the monitor is separate from the verifier, the raw log added eight unique signatures and accepted-only added none on the 138 safe windows with quota met. No within-prompt information advantage crossing was verified (0/8).

### Raw tilted TinyStories

Six frozen short-window cells with disjoint archived neural proposal and reference rows, nonlinear bounded reward and undisclosed normalizer: two reference-safe, three reference-unsafe and one unresolved. At cap 10,240 clipped scored-Q only signed neither safe cell. Opaque and revealed-scale target arms, attempt-count sharing, a paid scale interval and independent pre-tilt draws each signed the same one safe cell and neither the unsafe nor unresolved cells. No power gain from an oracle or paid interval on this panel. These are comparisons of bought prefixes of archived samples, not new generations or wall-clock.

### External source economics

An external target row and a rejection-derived target row are different contracts. A local design replay on the frozen GSM8K format monitor, conditional on sufficient observed reward-tier support, has 140 correlated windows from five prompts. At the expected rejection acquisition price every window allocates all information budget to raw Q. A separately delivered iid target row at a hypothetical lower price can favor a mixed design. These are local variance predictions, not finite certificate power and not evidence that such a production source exists. Capturing raw rejected rows is the executable control when the auditor pays to acquire P itself.
