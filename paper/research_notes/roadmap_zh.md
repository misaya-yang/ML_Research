# ICML 2027 Roadmap

## 2026-11-23: Main line and execution order

Follow the [research question and main line](../../RESEARCH_QUESTION.md). This round focuses on exact first-passage acceleration for the existing verification-guided search: (1) establish what the same real policy does uniquely well and what headroom remains after optimization; (2) turn the carrier, its probability, conditional sampling, and earlier-hit checks into priced native interfaces, with a boundary under equal access; (3) evaluate strong baselines on the same task with independent holdout data and the same physical budget. Preserve the 30-task 14.90% result and the original 15% gate. Do not expand the narrow task, launch OProver 8B, or promote one-shot decoding or older audit work to the main line. The earlier plan and status below are historical and should not be run automatically if they conflict with this section.

Drafted 2026-11-22. Based on prior years, the deadline is expected in late January; confirm on the official site. This is the only roadmap and is updated in place, not as dated copies. See the [current research brief](research_brief_zh.md).

**Target:** solid accept. Oral is an upside case, contingent on E3 producing a figure in which experiments match the theoretical prediction.

**Responsibilities:** Claude leads theory; see [three_channels.md](../../theory/proofs/three_channels.md). The experimental agent follows [EXPERIMENT_PLAN_zh.md](../../experiments/EXPERIMENT_PLAN_zh.md), starting at [EXPERIMENTS_README.md](../../EXPERIMENTS_README.md).

## 1. Positioning

**Candidate claim:** Auditing a sampler and running it require different information. Cold-start IMH certification cost follows a two-source frontier between target samples and proposal scores; disclosing a single scalar, the normalizer, changes it.

**Reframing** (to answer “Why not just sample from P?”):

- Replace “MSE < Var/2” with “cold-start effective sample size ESS ≥ k”; Var/2 corresponds to k=2. State the main result for general k.
- Add marginal bias |E h(X_T) − Ph| to cover generative sampling that returns a single draw.
- Amortization proposition: certify once offline, then deploy without access to P. Turn the pricing analysis in §7 of the older draft into a break-even criterion for when auditing is cheaper than drawing directly from P.

**ML connection:** Certify inference-time compute budget T. Use a reward-tilted target P ∝ π_ref·exp(r/β), where Z is genuinely intractable and white-box dynamic programming does not apply.

**Not claimed:** Method advantage at practical budgets or general deployment speedup. The auditor is positioned as valid and rate-optimal.

## 2. Main paper structure (8 pages)

| Section | Contents | Existing source |
|---|---|---|
| Main theorem | Three information channels: upper W tail → centering (pay for P), overlap/holding time → cold-chain dynamics (pay T for a Q path or 1/η₀ per virtual P jump), lower W tail → scale (pay for Q when ω is small) | [Long-window two-source frontier](../../theory/proofs/long_horizon_two_source_frontier.md), [moment frontier](../../theory/proofs/pro_moment_frontier_original.md) |
| Exact theorem | Fixed-p ceiling Π_p = min{1, δ/(1−κ)^p} and scale-free e-value using elementary symmetric polynomial e_p; at p=0, power ≤ δ | [Fixed-target power](../../theory/proofs/fixed_target_power.md) Part I |
| Corollary | Two-source minimax lower bound for log Z, addressing the open question in Micaletto (arXiv 2610.17403) | [Normalizer estimation](../../theory/proofs/normalizer_estimation_black_box.md) |
| Experiments | E1, E2, E3; E4 in the appendix | See §4 |
| Appendix | Full proofs, equality counterexample, and finite-state checks | Part II, [cap_equality.py](../../experiments/finite_state/cap_equality.py) |

Figures: Fig 1 three channels and phase diagram; Fig 2 E1 frontier slope; Fig 3 E2 power curves; Fig 4 E3 β sweep.

## 3. Theory tasks

| ID | Task | Priority | Completion |
|---|---|---|---|
| T0 | Bring proofs into the repository. Import files currently available only through ChatGPT Library: verify_local.py, verify_interval.py, weighted_half_gate.py, ratio_floor_fixed_omega.py, fixed_gate.py, rational_gate.py, LOCAL-POWER.md, NORMALIZER-MINIMAX.md, half_gate.md. Write self-contained appendix proofs and have a collaborator check them line by line | Blocking | Each proof records checker and date; all referenced scripts run locally |
| T1 | Unify the main theorem: replace the moment class with a tail-profile class P(W>D) ≤ Ψ(D), optimizing the frontier over cutoff D. The upper bound already clips at D≈q/p; use a three-layer lower bound at the critical level; recover p^{1−r}q^r in the power-law case | Required | One theorem covers all three channels; if unfinished at the end of W3, state separate moment and overlap theorems and unify them in a phase diagram |
| T2 | Add a finite-sample rate uniform in q to the exact fixed-p theorem, at least in the moment class | Required | Address the objection that the result is only a pointwise limit |
| T2b | Check the corollary: κ ∝ \|G\| near a simple root, so pointwise as q→∞ about log(1/δ)/ε target samples suffice, whereas the uniform minimax requirement is ε⁻²log(1/δ) | Required | State a proposition or remove the claim if false |
| T3 | Extend to general Var/k and marginal bias; calibrate the lower bound at an arbitrary root instead of J(c)=1/2 | Desirable | The theorem does not depend on k=2 |
| T4 | Fill gaps: weighted lower bound for η₀∈(1/4096,1/64], short windows Tη₀<C/ε, and ω>cη₀ | Desirable | Resolve at least ω>cη₀, corresponding to the E3 β sweep |

**Out of scope:** Classifying all equality cases (keep the counterexample in the appendix); putting the Tauberian/Wiener material in the main text. Remove the sentence about the Riemann hypothesis from the older draft.

**Progress (11/22; all derived by Claude and pending human verification):**
- T1 drafted in [tail_profile_frontier.md](../../theory/proofs/tail_profile_frontier.md). Under constant overlap the frontier is p≍A·Ψ(q/p). Moment, regularly varying, and piecewise linear tails follow; k is general, with constants degrading polynomially in k. Joint small overlap and tail profiles remain open.
- T2 drafted in [fixed_target_extensions.md](../../theory/proofs/fixed_target_extensions.md) §4, with finite-q rate (M/q^r)^{1/(1+r)} for approaching the ceiling in the moment class.
- T2b proved (same file §2): κ≥|G_k|/(1+1/k), giving pointwise Θ(ε⁻¹log(1/δ)) target samples, versus a uniform ε⁻² lower bound.
- T3 partly done (same file §1): the same ceiling applies to any one-dimensional centering audit (ESS, output bias, window bias). The centering lower bound extends to general k. The weighted dynamics family covers only k∈(1.2,2.34).
- T4 partly done: the scale constraint is redundant when ω≥cη₀ ([three_channels.md](../../theory/proofs/three_channels.md), Proposition C1), so all three inequalities hold for every ω. Short horizons remain open.
- New disclosure dichotomy: with known Z, pointwise p=0 suffices; with unknown Z, the precise cap is δ/(1−κ)^p. This can anchor Theorem 2.
- Joint small-overlap frontier partly resolved in [small_overlap_joint_frontier.md](../../theory/proofs/small_overlap_joint_frontier.md).
  - The centering channel applies at arbitrary overlap using an intersection-over-grid certificate, with an extra log log q in the upper bound.
  - The unified frontier for everywhere-valid auditors is p≍ε⁻²log(1/δ)·max{Ψ(q/p),ε}. As q→∞ it connects to pointwise ε⁻¹, linking Theorem A and Theorem 2.
  - The joint three-channel frontier holds when the tail profile does not constrain weights within the window scale. The absorbing layer in the dynamics lower bound moves from T² to roughly 3T, verified by exact numerical checks.
  - Tail-dependent P jump cost is C·min{1/η₀, 1+D_{1/2}/(η₀t)}, checked by simulation.
  - Cases where the tail decays within the window are open; the conjecture and next step are written down.
- Cold-start risk upper bound proved in [cold_risk_bound.md](../../theory/proofs/cold_risk_bound.md).
  - For any equivalent pair, any h∈[0,1], and any (b,t), R_Q(b,t)≤44Γ_t with Γ_t=E_P min(W,t)/t. No tail assumption or spectral gap is required; W can be unbounded. The largest exact numerical ratio is about 1.4.
  - Refined R≤C·[E_P min(W/t,1,L/W)+τ_P(L)²]: cold-start contributions for weights beyond L are discounted by L/W; the bias term is quadratic in τ.
  - Near-threshold instances necessarily have Var_P(h)≤k(44Γ_t+ε), so hard cases must have low-variance h.
  - The Γ certificate is everywhere valid without running the chain, with p≍kℓ/Var and q≍d_{1/4}ℓ. Certification with a constant-factor margin is cheap; the three-channel cost arises only near the threshold.
  - The upper-bound half of the open-regime conjecture is established up to a factor k.
- Variance-scaled lower bound proved for constant overlap in [variance_scaled_floor.md](../../theory/proofs/variance_scaled_floor.md).
  - Two sticky levels at t/c₁ and t/c₂ each have target mass ρ/2. Moving only h=1 labels across them preserves W laws under P and Q, both h marginals, μ, and Var, meeting Pro's conditions.
  - The result is p+q/t≳A·min{Ψ(t),c/k}: the light-window dynamics cost is A·ρ rather than A. Combined with R4, the constant-overlap dynamics channel is p+q/t≍AΨ(t) for Ψ(t)≥Cε; the upper bound is still a draft.
  - The label process is independent of the level process, so G_k is exactly quadratic in the tilt parameter, with zero first-order curvature. Exact numerical check passed (8 states, residual 1e-16).
  - The very light regime Ψ(t)<Cε remains open. The established result can be Theorem 1(c).
- Small-overlap lower bound derived (same “Small overlap” section). The gate, public information, and sampling access match the constant-overlap regime; human audit pending, experiments X11/P14.
  - Impossibility: if the class requires Γ_t≤γ<½, no instance exists at ηt<1/(4γ). A five-layer base realizes the other regime, with ρ_B=d/t≍1/(ηt).
  - V_η: sticky-label lower bound p+q/t≳ρA holds at any η. The coefficient of P remains 1, not ηp.
  - Proposition N: after b≳2d·log(Γρ_B/ε²) burn-in, or at b=0 and ηt≳Γ/ε², any witness pair preserving public marginals satisfies KL_P≥ε²/(330Γ_t). It cannot witness λ<1 here. If interpolation attains this rate (unproved), E4's λ≍1/c_P conjecture fails in this regime.
  - Theorem S: at b=0, tilting labels in the startup level (weight 1, P mass 1/d, cold-start occupancy about ρ_B/2) can cross the gate. Its cost is p+q≳d a_U²A/64 with a_U≍ρ_B·max(ρ,ρ_B). For a P-heavy auditor this is ρ_B⁴A/η, far larger than ρA when η≪ρ_B⁴/ρ. Exact numerical check: gain K grows linearly with t, reaching 468 at t=2^18.
  - Still open: ε²/Γ_t≲ρ_B≲ε/max(ρ,ρ_B); k≤3.6 when startup lies within the window; a Γ/ρ_B³ gap between Theorem S and E4's P route.

## 4. Experimental tasks

See [EXPERIMENT_PLAN_zh.md](../../experiments/EXPERIMENT_PLAN_zh.md) for executable pseudocode, frozen criteria, and Kaggle budgets. Mapping: E0→X0, E1→X1, E2→X2, E3→X4/X5, E4→X6, E5→X7, plus X3 (general-k verification).

Continue existing accounting: controls get equal information; target draws are charged at actual generation cost, including rejection attempts; label white-box controls as oracles and exclude them from method wins/losses.

| ID | Experiment | Success criterion |
|---|---|---|
| E0 | **Pre-screen.** The higher-order reward control shows that when the proposal head is cheap to cover, pointwise scores with zero target samples decide every case ([per-case results](../../results/disclosure/README.md)). Measure how much probability an affordable proposal head covers before scaling up | Only tasks with insufficient head coverage enter E3/E4 |
| E1 | Use lossless multinomial counts on a logarithmic (n_P, n_Q) grid to compare a known-world likelihood ratio test (information lower envelope) with the auditor. r∈{.25,.5,.75}; run known and unknown scale | Slope confidence intervals cover −(1−r)/r; the unknown-scale frontier shifts in the predicted direction |
| E2 | p∈{1,2,4,8,16}; power convergence with q and target quota dependence on κ | Difference from δ/(1−κ)^p within Monte Carlo error; required p increases as 1/κ |
| E3 | LLM reward tilt P ∝ π_ref·exp(r/β), Q is π_ref or a smaller model; exact P via rejection sampling with envelope e^{r_max/β}. β determines both ω=e^{−Δr/β} and η₀. Candidate observables: GSM8K verifier correctness, safety marker, or format. Certify inference budget T on held-out prompts | For at least one β interval, the failure point of P-only scale estimation matches theory; report real generation cost |
| E4 | Preregister at least 200 safe and 200 unsafe near-threshold tasks (synthetic and LLM) | 95% upper confidence bound on false certification ≤ δ |
| E5 | Improve constants: betting CIs (Waudby-Smith & Ramdas) instead of Maurer–Pontil empirical Bernstein, estimate θ via conditional binomial intervals on accepted Q draws (16/16 pass if exact θ is given in the ablation), and sequential e-process early stopping | At moderate budgets, substantially close the budget ratio relative to the likelihood ratio envelope |

## 5. Timeline

| Week | Dates | Theory | Experiments | Checkpoint |
|---|---|---|---|---|
| W1 | 11/23–11/29 | Finalize narrative and theorems; import missing T0 files | E0 screen; E3 feasibility (rejection acceptance, model/reward selection) | Finalize narrative; E0 finds at least one insufficient-head setting |
| W2 | 11/30–12/06 | Appendix-level T1/T2 proofs | E1 | |
| W3 | 12/07–12/13 | Collaborator verifies T1/T2; T2b | E2 | If T1 remains open, use the fallback |
| W4 | 12/14–12/20 | T3 | Main E3 run | If acceptance is too low, shorten sequences or narrow β |
| W5 | 12/21–12/27 | T4 | Finish E3; E5 | Holiday buffer |
| W6 | 12/28–01/03 | Unify notation | E4 | Freeze primary figures |
| W7 | 01/04–01/10 | Write paper and appendix | Add ablations | |
| W8 | 01/11–01/17 | Write | Reproduction bundle consistency checks | Full first draft |
| W9 | 01/18–01/24 | Mock review focused on proof gaps | | Submit |

## 6. Fallbacks

- If E0 cannot find an LLM with insufficient proposal head coverage, or E3 has no result by the end of W5, submit theory plus synthetic validation to ICML. Remove inference-time budgeting from the narrative and retain ESS certification.
- If T1 remains open, state separate moment and overlap theorems and unify them with a phase diagram.
- Consider submitting the log Z lower bound separately to AISTATS or COLT to avoid overloading the main text.

## 7. Risk register

| Risk | Issue | Countermeasure |
|---|---|---|
| R1 Motivation | Older draft §7 itself proves 13 independent labels suffice for a cached scalar with guaranteed MSE; Var/2 looks arbitrary | Reframe in §1 and do T3 |
| R2 Experiments | GUM clip+RB8 certifies 0/16; finite cases need 2²⁴ target samples and 2⁴⁰ scores before beating replay; white-box DP decides 16/16; zero-target sampling decides 6/6 on the higher-order reward control | Do not claim practical advantage; run E0 and move E3 to settings with intractable Z |
| R3 Fragmentation | 4–5 theorems use different oracles, classes and regimes, with constant restrictions and acknowledged gaps | T1 and T4 |
| R4 Verification | Most proofs are compressed Pro derivations, some labeled “candidate”; referenced files are missing from the repository | T0 |
| R5 Venue fit | Presentation reads like COLT/AoS | ML connection in §1 and E3 |

## 8. To verify

- The T2b comparison of pointwise ε⁻¹ and uniform ε⁻².
- Exact findings of prior LLM-MCMC work (QUEST, QAlign, Karan & Du power sampling, and RSO rejection sampling). Most use prefix-preserving MH; IMH is the special case that redraws the full sequence.
- Exact ICML 2027 deadline and page limits.
