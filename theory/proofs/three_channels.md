# Three information channels: synthesis and paper-level statements

Status: synthesis by Claude, 2026-11-22. It collects which results enter the paper, adds one closure (Proposition C1), and lists the open problems. The proofs live in the linked files.

## The picture

A certificate for a cold IMH window needs three kinds of information. Each is paid for by a different source.

| Channel | What is unknown | Who pays | Governing quantity | Result |
|---|---|---|---|---|
| Target center | μ=Ph, through target mass at weights the proposal rarely visits | Target draws above the balanced level q/p; proposal scores below it | Upper tail τ_P(D)=P(W>D) | [Theorem A](tail_profile_frontier.md) and [E1′](small_overlap_joint_frontier.md): p≍A·max{Ψ(q/p),ε} |
| Cold dynamics | Holding times of the deployed chain | Q paths cost T each; P-based lag jumps cost 1/η₀ each | Overlap η | [Long-horizon frontier](long_horizon_two_source_frontier.md): η₀p+q/T≳A |
| Scale | The common normalizer Z | P harmonic estimation if ω is large, otherwise bridges that need q | Lower tail ω=inf W | Same file plus Proposition C1: ωp+η₀q≳A |

These combine with the pointwise and uniform lenses of [fixed_target_extensions.md](fixed_target_extensions.md) §3:

| | Pointwise (q→∞) | Uniform (finite q) |
|---|---|---|
| Z known | p=0 suffices | center channel + dynamics channel |
| Z unknown | exact ceiling δ/(1−κ)^p, i.e. p≍ε⁻¹log(1/δ) | all three channels |

## Proposition C1: the scale constraint is redundant once ω≥cη₀

Take the long-horizon setting of the frontier note (Tη₀≥C/ε). Since W≥ω everywhere implies η≥ω, always 0≤ω≤η.

**Claim.** For any fixed c∈(0,1] and ω≥cη₀, the feasible set with unknown scale equals the feasible set with known scale, up to constants depending on c. The three-inequality characterization therefore holds for all ω∈[0,η₀]; the original proof needed ω≤cη₀ only for its hard family.

*Necessity.* An unknown-scale auditor is also a known-scale auditor (it can ignore Z), so the first two inequalities are necessary. They imply the third:

- if η₀n_P≥A/2, then ωn_P≥cη₀n_P≥cA/2;
- otherwise n_Q/T≥A/2, and η₀n_Q≥(η₀T)A/2≥A/2, because η₀T≥C/ε≥1.

*Sufficiency.*

- In the Q-rich branch (n_Q≥C′TA), direct cold paths use score ratios only, and the clipped center estimator is scale-free.
- In the target-rich branch, n_P≥C″A/η₀≥C″cA/ω. The harmonic estimator (E_P(1/S)=1/Z, E_PW⁻²≤1/ω) gives a valid O(ε) log-bracket from O(A/ω) target draws and no Q. Choosing C″ larger than the harmonic constant divided by c makes this affordable.

∎

## Observation C2: small overlap forces a flat tail profile

For D≥1, W1{W≤D}≤D·min(1,W). Hence P(W≤D)≤Dη, that is, τ_P(D)≥1−Dη. A class with overlap η₀ therefore has Ψ(D)≥1−Dη₀, so Ψ≈1 up to about 1/(2η₀).

Consequences:

- In the long-horizon class the center channel reads p≥A, which matches family 1 of the frontier note.
- Theorem A's original pilot needed overlap at least 1/8, and the dynamics floor put its soak at weight T². Both restrictions are now removed: a grid-intersection certificate handles any overlap, and a soak at about 3T suffices. This gives the joint frontier whenever Ψ≡1 up to the window scale ([small_overlap_joint_frontier.md](small_overlap_joint_frontier.md)). Profiles that decay inside the window remain open.

## Draft statements for the paper

**Theorem 1 (three-channel frontier).** Let A=ε⁻²log(1/δ) and k>1.

- (a) *Center.* On C(Ψ) at **any** overlap, for auditors that are valid everywhere, feasibility is equivalent to p≍ε⁻²log(1/δ)·max{Ψ(q/p),ε}. One scale-free auditor attains it for every Ψ simultaneously. As q→∞ this meets the pointwise rate of Theorem 2. See [small_overlap_joint_frontier.md](small_overlap_joint_frontier.md) E1 and E1′.
- (b) *All three channels.* On the long-window class with Tη₀≥C/ε, W≥ω (any ω∈[0,η₀]) and a tail profile that constrains only weights beyond the window scale (Ψ≡1 on [1,128kT]), feasibility is equivalent to

$$
n_P\gtrsim A\,\Psi\big(\max(n_Q/n_P,128kT)\big),\qquad \eta_0n_P+n_Q/T\gtrsim A,\qquad \omega n_P+\eta_0n_Q\gtrsim A\ \ (\text{only when }Z\text{ is unknown}).
$$

Each inequality is witnessed by its own family inside the class (E2). With Ψ≡1 this is the original long-horizon theorem.

- (c) *Light windows, constant overlap (lower bound proved, upper bound a sketch).* If Ψ(t)≥Cε, the dynamics channel becomes p+q/t≍ε⁻²log(1/δ)·Ψ(t), up to constants depending on (k,β,η₀,C_Ψ,C_d) and the factor 1+β on the direct-path side. Necessity holds even with both W laws, both h marginals, μ and Var disclosed ([variance_scaled_floor.md](variance_scaled_floor.md)). Sufficiency uses R≤44Γ_t ([cold_risk_bound.md](cold_risk_bound.md) R4).
  - *Small overlap, same disclosure (lower bounds proved).*
    - The class is empty when ηt<1/(4γ).
    - p+q/t≳AΨ(t) holds with P-coefficient 1 at every η.
    - After burn-in b≳2d·log(Γ_tρ_B/ε²), no disclosed-marginal witness forces less than about Γ_tA target draws.
    - At b=0, startup labels force p+q≳d a_U²A, which is about ρ_B⁴A/η for P-heavy auditors.
    - See the "Small overlap" section of variance_scaled_floor.md.

**Theorem 2 (exact pointwise ceiling).** For any interval audit (ESS gate for any k>1, output bias, window bias), every honest issuer has power at most min{1,δ/(1−κ)^p} for every q, and a single e-value procedure built from elementary symmetric polynomials attains it as q→∞. The target requirement is Θ(ε⁻¹log(1/δ)) pointwise, versus Θ(ε⁻²log(1/δ)) uniformly. With Z disclosed, p=0 suffices pointwise.

**Corollary 3 (normalizer).** Estimating log Z to ε needs n_P,n_Q≳η₀⁻¹ε⁻²log(1/δ) each, and the bridge estimator attains this ([normalizer_estimation_black_box.md](normalizer_estimation_black_box.md)).

## Status ledger

| Claim | File | Status |
|---|---|---|
| Moment frontier p^{1−r}q^r≍MA | pro_moment_frontier_original.md | Pro proof; rational smoke checks in the external archive |
| Tail-profile generalization, general k | tail_profile_frontier.md | Claude derivation; needs audit (X1, X3) |
| Long-horizon three inequalities | long_horizon_two_source_frontier.md | Pro proofs; several marked "candidate"; finite checks in repo |
| ω≥cη₀ closure | this file, C1 | Claude derivation; elementary |
| Exact ceiling (k=2, binary) | fixed_target_power.md Part I | Pro proof, checked by the team |
| Interval audits, general k, bias audit | fixed_target_extensions.md §1 | Claude derivation; elementary extension |
| Pointwise ε⁻¹ vs uniform ε⁻² | fixed_target_extensions.md §2 | Claude derivation; elementary |
| Disclosure dichotomy | fixed_target_extensions.md §3 | Claude derivation; (a) elementary |
| Finite-q rate | fixed_target_extensions.md §4 | Claude derivation; needs audit (X2c) |
| log Z two-source minimax | normalizer_estimation_black_box.md | Pro proof |
| Center channel at any overlap; saturation at ε⁻¹ | small_overlap_joint_frontier.md E1, E1′ | Claude derivation; needs audit |
| Joint frontier under window-scale flatness | small_overlap_joint_frontier.md E2 | Claude derivation; built from audited components plus E3 |
| Soak lowered to about 3T | small_overlap_joint_frontier.md E3 | Proof sketch; exact numerics (soak_floor.py) |
| Tail-aware lag-jump cost | small_overlap_joint_frontier.md E4 | Claude proof; simulation (jump_cost.py) |
| Clipped bridge, instance-dependent scale cost | small_overlap_joint_frontier.md E5 | Elementary; similar bridge functions exist in the literature |
| Cold-start risk bound R≤44Γ_t; beyond-window refinement R≤C·Ξ_{b,t}; cold-law domination | cold_risk_bound.md Lemmas R, R′, D | Claude proof; exact numerics (risk_bound.py, risk_bound_refined.py), observed sup ≈1.4 |
| Γ-certificate (valid everywhere, no chain paths) | cold_risk_bound.md R2 | Claude derivation; needs audit (X9) |
| Light-window dynamics upper bound ∝A·max{Γ_t,ε} | cold_risk_bound.md R4 | Sketch |
| Variance-scaled light-window floor p+q/t≳A·min{Ψ(t),c/k} (constant overlap; marginals disclosed) | variance_scaled_floor.md Theorem V | Claude proof; exact numerics (variance_floor.py); Step 2 remainders follow the flat family |
| Small overlap: impossible interval ηt<1/(4γ); V_η (coefficient 1 at any η); no λ<1 witness after burn-in (Prop. N); startup-label witness at b=0 (Thm S) | variance_scaled_floor.md, section "Small overlap" | Claude derivation; needs audit (X11); exact numerics (small_overlap_floor.py); Thm S Step 2 moments are leading order |

## Open problems, in priority order

1. **Joint small-overlap + tail-profile frontier.** *Partly resolved* in [small_overlap_joint_frontier.md](small_overlap_joint_frontier.md). The center channel now holds at every overlap (grid-intersection certificate), and the complete joint frontier is proved whenever Ψ≡1 up to the window scale. Still open: profiles that already constrain weights inside the window. There, conjecturally, the dynamics channel becomes λp+q/T≳A·max{ρ_T,ε} with λ≍max{η₀,min(1,η₀t/D_{1/2})} and ρ_T≍Γ_T=E_P min(W,T)/T. The risk bound R≤44Γ_t is proved ([cold_risk_bound.md](cold_risk_bound.md)), and so is the upper half of the conjecture up to a factor k. The variance-scaled lower family is built at constant overlap ([variance_scaled_floor.md](variance_scaled_floor.md)), giving p+q/t≳A·min{Ψ(t),c/k}. With R4, the constant-overlap light-window dynamics channel is p+q/t≍AΨ(t) when Ψ(t)≥Cε; the upper half is still a sketch. Small overlap is now settled except for one band:
   - λ=1 (V_η) whenever the startup is burned off or the horizon is long (Proposition N);
   - λ<1 only through startup labels at b=0 (Theorem S).

   Open: the band ε²/Γ_t≲ρ_B≲ε/max(ρ,ρ_B), a matching estimator for Proposition N's regime, and Ψ(t)<Cε.
2. **Short horizon Tη₀<C/ε.** Replace the high-only endpoint estimator with the exact lag sampler (Theorem 1 of the stale PDF draft), which removes the O(1/(Tη₀)) bias. Its low branch consumes about k(1−η)^{k−1} Q draws, so the expected frontier is the earlier pooled rate Θ(min(T,1/η₀)A) split between the two sources. The cost accounting is not done.
3. **General k for the weighted dynamics floor.** The CTMC functional F(c,0) ranges only over about [0.427, 0.83] (numerical minimum 0.4267 at c≈2.28), so it crosses 1/k only for k∈(1.2, 2.34). For a threshold 1/k outside this range, a time-rescaled family (about k switches per window) is needed.
4. **The gap η₀∈(1/4096,1/64]** in the center-known weighted floor. This is minor: the necessary inequality already holds with a worse constant.
