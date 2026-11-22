# Joint frontier at small overlap

Status: derivation by Claude, 2026-11-22. It attacks open problem 1 of [three_channels.md](three_channels.md): small overlap combined with a tail profile. Needs an independent audit.

Numerical checks:
- [soak_floor.py](../../experiments/finite_state/soak_floor.py): exact computation.
- [jump_cost.py](../../experiments/finite_state/jump_cost.py): simulation.
- Logs are in `results/local_runs/theory_checks-20261122/`.

Notation follows [tail_profile_frontier.md](tail_profile_frontier.md). A=ε⁻²log(1/δ) and ℓ=log(1/δ). D_{1/2}:=inf{D: Ψ(D)≤1/2} is the bulk scale of the profile. "Valid everywhere" means false-issue probability at most δ on every equivalent pair; "powerful on a class" means issue probability at least 1−δ on its members with G_k≤−ε. Validity should not lean on an unverifiable tail assumption (Pro's empty-tail obstruction), so this is the natural convention. Only one statement below (E1′) needs it.

## Results

| | Statement | Status |
|---|---|---|
| E1 | The center channel p≍AΨ(q/p) holds at **every** overlap η₀>0, using a grid-intersection certificate. The upper bound costs an extra log log q in ℓ. | Proof below |
| E1′ | For auditors valid everywhere: p≍ε⁻²log(1/δ)·max{Ψ(q/p), ε}. As q→∞ the uniform cost falls to the pointwise rate ε⁻¹log(1/δ) of Theorem B2. | Proof below |
| E2 | **Complete joint frontier** whenever Ψ≡1 on [1,128kT] (Theorem A's standing assumption in the long-window setting): (C)∧(D)∧(S) | Proof below; uses E1 and E3 |
| E3 | The dynamics floor family needs its soak level only at about 3T, not T² | Proof sketch plus exact numerics |
| E4 | Tail-aware cost of the P-based lag jump: c_P≤C·min{1/η₀, 1+D_{1/2}/(η₀t)} target draws per replicate | Proof plus simulation |
| E5 | Clipped bridge: the scale costs p≳A/M(D) and q≳A/N(D) for the best clip level D (instance-dependent) | Proof below |
| Open | Profiles that already constrain weights inside the window (Ψ(cT)<1) | Risk bound R≤44Γ_t proved ([cold_risk_bound.md](cold_risk_bound.md)); upper half of the conjecture (sketch); lower bound p+q/t≳AΨ(t) at constant overlap ([variance_scaled_floor.md](variance_scaled_floor.md)); small overlap: P-coefficient 1 survives (V_η), λ<1 impossible to witness at long horizon or after burn-in (Prop. N), witnessed at b=0 via startup labels (Thm S); band ε²/Γ_t≲ρ_B≲ε/max(ρ,ρ_B) open, §6 |

## 1. E1: the center channel at any overlap

The only step of Theorem A that used η≥1/8 is Pro's pilot (Z/16≤A₀≤64Z). Replace it as follows.

**Anchor.** From a separate bank of n₀=C·ℓ target rows, take the empirical 1/4-quantile a of S. Since W1{W≤D}≤D·min(1,W), we have P(W≤D)≤Dη. So the true 1/8-quantile of W under P is at least 1/(8η)≥1/8. In the band (Ψ(K/C)≤1/2) the true 3/8-quantile is at most inf{D: Ψ(D)≤5/8}≤K. By the DKW inequality, a/Z∈[1/8,K] with probability at least 1−δ/4.

**Grid.** Set H_j=a·K_c·2^{−j} for j=−3,…,⌈log₂K⌉+1, so J≤log₂q+6. With j*=⌊log₂(a/Z)⌋, the clip level D_{j*}=H_{j*}/Z lies in (K_c,2K_c].

**Certificate.** Run Pro's clipped interval for (μ,ν) at every H_j, each with error δ/(4J). Intersect the intervals, then apply the unchanged path test.

- *Validity.* Pro's identity is exact for every positive clip level fixed independently of the certificate rows, which the anchor bank ensures. A union bound over J levels and an intersection of valid intervals give validity on every equivalent pair.
- *Power.* Analyze only j*. The head denominator obeys u₁+u₀=E_Q min(W,D)/D≥P(W≤D)/D≥(1−Ψ(D))/D≥1/(2D) in the band, with no dependence on η. With (U2)–(U3) of tail_profile_frontier.md, the widths are O(√(Ψ(K)ℓ'/p)+ℓ'/p), where ℓ'=log(4J/δ). The path bank needs q≳LA, which holds because KΨ(K)≥D₀≥L.

**Lower bound.** Unchanged. Theorem A's family has overlap at least 3/16, which is at least η₀, so it lies in C(Ψ;η₀).

The log log q costs nothing whenever δ≤1/log q. Removing it in general would need a pilot that brackets Z to a constant factor; at small overlap neither the proposal quantile nor the target quantile does that.

## 2. E1′: saturation at the pointwise rate

**Necessity.** Take the base instance of Theorem A's family without the hidden level, with G_k≈−ε. It lies in C(Ψ) and sits near a simple root, so κ≍ε. Theorem B1 allows the unsafe contamination to lie outside the class, which is legitimate because validity is required everywhere. Power 1−δ at the base therefore needs p≥log((1−δ)/δ)/(−log(1−κ))≳ℓ/ε.

**Sufficiency.** The certificate error is O(√(Ψ(K)ℓ/p)+ℓ/p). This is at most ε once p≥C·max{Ψ(K)A, ℓ/ε}. If Ψ(K)≤ε, the path bank is still covered: KΨ(K)≥D₀ gives K≥D₀/ε≥L/ε, so q=pK≥Cℓ·L/ε²=CLA.

Combining both directions with Theorem A gives **p≍A·max{Ψ(q/p),ε}**. This one formula links the two lenses:

- with no tail control (Ψ≡1), the uniform cost stays at ε⁻²;
- with tail control, once proposal scoring resolves all tail mass above ε (Ψ(q/p)≤ε), the uniform cost equals the pointwise ε⁻¹ rate.

An auditor that is valid only on C(Ψ) and knows Ψ might go below ℓ/ε. That convention is not pursued here.

## 3. E2: the complete joint frontier under the standing flatness

**Setting.** Long window T=L, Tη₀≥C/ε, η₀≤1/4096, 0≤ω≤η₀, δ≤1/10.

**Class.** C(Ψ,η₀,ω) consists of pairs with η≥η₀, W≥ω and τ_P≤Ψ, where Ψ satisfies (Ψ1)–(Ψ3) and Ψ≡1 on [1,D₀] with D₀=128kT. In words, the profile constrains only weights beyond the window scale.

**Theorem E2.** Up to constants depending on (k,C_Ψ,C_d), (p,q) is feasible if and only if

- **(C)** p≳AΨ(max(q/p,D₀)), within Theorem A's band; auditors valid everywhere also need p≳ℓ/ε;
- **(D)** η₀p+q/T≳A;
- **(S)** ωp+η₀q≳A, only when Z is unknown.

For Ψ≡1 this reduces to the long-horizon theorem, where (C) reads p≳A. In general the center floor relaxes to AΨ(q/p) in the Q-rich branch, where q/p≫T.

*Necessity.* Each inequality has a family inside the class.

- **(C):** Theorem A's family at hidden level K=max(q/p,D₀). Its weights are at most K, its tail τ_P is at most Ψ (checked in tail_profile_frontier.md), and min W≈.19≥ω.
  - For q/p<D₀, use K=D₀ and m=1/16. Then p+q/D₀≳A/16, so p≳A/32.
  - Long-horizon family 1 is no longer needed. Its unbounded weights would violate Ψ in any case.
- **(D):** the weighted dynamics family with its soak moved to H=C_sT, where 3≤C_s≤128k (E3). Every weight is at most D₀, so τ_P≤1=Ψ there, and min W=2κ≥ω.
- **(S):** the scale family of the long-horizon note. Its largest normalized weight is about T/c*≤T≤D₀ (c*≈2.557), and min W≥ω by construction.

*Sufficiency.*

- **Q-rich branch, q≥C′TA.**
  - Path moments come from ordinary cold paths and are scale-free.
  - Center: if p≥CA, use direct target anchors.
  - Otherwise use the grid certificate of E1. K=q/p≥C′T/C≥D₀ lies in the band, and (C) supplies p.
- **Target-rich branch, q<C′TA.**
  - (D) forces p≳A/η₀≥A, so direct anchors handle the center.
  - The high-only lag-jump estimator uses at most 2/η₀ target draws per replicate (E4 or the original bound).
  - Scale: the harmonic estimator if ωp≳A; otherwise (S) gives q≳A/η₀, and the standard bridge works because p≳A/η₀ already.

∎

## 4. E3: the soak can sit at about 3T

The original weighted family puts the soak at H=T² so that the cold chain almost never visits it (probability Tq_H=O(1/T)). Move it to H=C_sT and keep every other definition. The quantity (H−l)q_H=1−l−(m−l)q_M is unchanged, so Ph=0, Var_P h=1, and the formulas for u and v stay exactly as they were.

Let τ_H be the first proposal of a soak state. Every soak proposal is accepted (H is the top weight), and the chain then stays for about C_sT/(400cκ)≫T steps. Hence

$$
R(c)=\mathbb E\,\Phi(c,\tau_H)+O(\kappa/C_s),\qquad P(\tau_H<T)\le 1/C_s ,
$$

where Φ(c,τ) is the risk given entry at τ. Before τ_H the path is the H-free process; conditioning on no soak proposal changes the proposal law only at relative order q_H. The c-derivative of Φ(·,τ) is bounded by the Poisson-mark argument of the original note. Therefore R′(c)=F′(c)+O(1/C_s) and R=F+O(1/C_s).

Calibrate c to the **actual** root of R, as Pro does with β, instead of using the limit. The crossing survives once the O(1/C_s) terms fall below the margins (.017 in level, .09 in slope). The KL orders do not change: the soak's P-mass, which is Θ(1), moves by O(κΔ), and its Q-mass Θ(1/(C_sT)) moves at relative order O(κΔ).

Exact cell-chain computation (T=2¹⁴, κ=1/4096, soak_floor.py):

| Soak H | R(.9) | R(1.2) | dR/dc at 1.05 |
|---|---|---|---|
| T² | .5223 | .4784 | −.1454 |
| 300T | .5223 | .4786 | −.1451 |
| 10T | .5235 | .4823 | −.1367 |
| 3T | .5227 | .4871 | −.1184 |
| T | .5015 | .4779 | −.0786 |

The root and a nonzero slope persist down to H≈T. The standing assumption Ψ≡1 on [1,128kT] covers H=3T with room to spare.

## 5. E4 and E5: what tails buy in the dynamics and scale channels

**Lemma E4 (tail-aware jump cost).** Use the high-only lag jump of the long-horizon note: descending virtual heights V_(1)≥V_(2)≥…, made from k iid Pareto(1) heights, each tested with a fresh target draw that succeeds when W≤V. For lag k, let β_k=exp(−k/(2D_{1/2})). Then the number N_k of target draws satisfies

$$
\mathbb E N_k\le \min\Big\{\tfrac1\eta,\ 2+\tfrac{2\beta_k\log(e/\beta_k)}{\eta}\Big\}.
$$

*Proof.* Let G be the number of heights at least D_{1/2}, so G~Bin(k,1/D_{1/2}). Each of the first G tests succeeds with probability at least 1−Ψ(D_{1/2})=1/2, independently. Hence P(N>j)≤min{(1−η)^j, 2^{−j}+E2^{−G}}; the first term is the draft's unsorted-trial bound. Also E2^{−G}≤β_k. Summing over j gives the bound. ∎

**Averaging over lags.** Σ_kβ_klog(e/β_k)≈4D_{1/2}. With i,j uniform on a window of length t, both lags have density at most 2/t, so the expected number of target draws per two-time replicate is

$$
c_P\le C\min\Big\{\tfrac1{\eta_0},\ 1+\tfrac{D_{1/2}}{\eta_0 t}\Big\}.
$$

In words, the window amplifies the effective overlap from η₀ to λ:=1/c_P≍max{η₀, min(1,η₀t/D_{1/2})}. When the target bulk lies inside the window's reach (D_{1/2}≤η₀t), P-based simulation costs O(1) target draws per replicate.

Simulation (jump_cost.py, 1500 replicates per cell, number of tests at lag k):

| η₀, r, D_{1/2} | k=D₁/4 | k≈D_{1/2} | 4D_{1/2} | 16D_{1/2} |
|---|---|---|---|---|
| 1/256, .5, 1024 | 52 | 111 | 2.2 | 1.15 |
| 1/256, .25, 4096 | 54 | 23 | 1.6 | 1.3 |
| 1/1024, .5, 4096 | 219 | 500 | 16.7 | 1.13 |

The drop occurs near k≈D_{1/2}log(1/η₀), as the β_k term predicts. That logarithm is real, but it disappears after averaging over lags.

**Proposition E5 (clipped bridge).** For every c>0, Z=E_Q min(S,c)/E_P min(1,c/S). Indeed, with D=c/Z, E_P min(1,D/W)=E_Q min(W,D). Put N(D)=E_Q min(1,W/D) and M(D)=E_P min(1,D/W)=D·N(D).

- Both are means of [0,1]-valued variables whose variance is at most their mean. Bernstein's inequality then gives a log-bracket of width O(ε) from q≳ℓ/(ε²N(D)) and p≳ℓ/(ε²M(D)).
- Intersect over a geometric grid of c values, anchored by a separate pilot. The best D is then attained at a cost of log log in ℓ.
- In the class:
  - D=1 gives N=M=η, the standard bridge (A/η₀, A/η₀);
  - D=D_{1/2} gives M≥1/2 and N≥1/(2D_{1/2}), that is (2A, 2D_{1/2}A).
- Against the scale floor ωp+η₀q≳A, the high-clip choice matches when D_{1/2}≤C/η₀, which means the tail starts to decay at the overlap scale.

The two-sided normalizer lower bound in [normalizer_estimation_black_box.md](normalizer_estimation_black_box.md) uses unbounded weights. In a tail class its target floor therefore weakens; E5 shows how far.

## 6. Open regime: profiles that constrain weights inside the window

If Ψ(cT)<1 for some c≤128k, three ingredients fail together:

- Theorem A's bulk level at J=128kL;
- the dynamics floor's soak, which needs a nearly full target mass at weights of at least 3T;
- the flatness used by the scale family.

What remains:

- **Upper bounds.** The grid center certificate, provided q≳LA. P-based dynamics at c_P (E4). The clipped bridge (E5).
- **Lower bounds.** (C) where Theorem A's family fits. The saturation p≳ℓ/ε. The scale floor, provided Ψ≡1 up to about 1/(8η₀).

**Risk bound (proved, [cold_risk_bound.md](cold_risk_bound.md)).** For every equivalent pair, every h∈[0,1] and every (b,t), R_Q(b,t)≤44Γ_t, where Γ_t=E_P min(W,t)/t=(1/t)∫_0^tτ_P. No tail assumption and no variance factor is needed. Under (Ψ2), Γ_t≤1/t+C_ΨΨ(t), and d≥1/(2η) puts the startup term 1/(ηT) inside Γ_t. The refinement R≤1520·[E_P min(W/t,1,L/W)+τ_P(L)²] discounts weights beyond L. Consequently, instances near the threshold have Var_P(h)≤k(44Γ_t+ε), so the absolute-margin minimax is governed by low-variance instances. Certifying with a constant-factor margin needs no chain paths (the Γ-certificate, R2 there).

**Conjecture.** In this regime the dynamics channel reads λp+q/T≳A·max{ρ_T,ε}, with ρ_T≍Γ_T and λ as in E4. Here Γ_T replaces the earlier guess Ψ(cT)+1/(η₀T), which is comparable under (Ψ2). In the flat regime ρ_T≍1 and λ≍η₀, which is (D). The upper half, with an extra factor k on the P route, is Corollary R4 of cold_risk_bound.md. The lower half at constant overlap (λ≍1) is Theorem V of [variance_scaled_floor.md](variance_scaled_floor.md): p+q/t≳A·min{Ψ(t),c/k} whenever this is at least Cε, with both W laws, both h marginals, μ and Var disclosed. Probably a relative-margin formulation (G_k/Var_P h≤−ε) is the cleaner way to state it.

**Small overlap (variance_scaled_floor.md, section "Small overlap").** The same gate and disclosure, with η→0 and both worlds in one W law that decays inside the window:

- *Lemma F.* Small overlap and a light window coexist only if ηt≥1/(4γ), where Γ_t≤γ<½. Below this the regime is empty. Above it a five-level base realises it with ρ_B=d/t≍1/(ηt).
- *Theorem V_η.* The sticky floor p+q/t≳ρA holds at every η. Its P-coefficient stays 1; it does not become ηp+q/t.
- *Proposition N.* If ρ_Be^{−b/(2d)}≤ε²/(660Γ_t), every disclosed-marginal pair that moves the gate by 2ε has KL_P≥ε²/(330Γ_t). So λ<1 cannot be witnessed after burn-in b≳2d·log(Γ_tρ_B/ε²), nor at b=0 once ηt≳Γ_t/ε². In that regime the conjectured λ≍1/c_P is false if a plug-in label estimator reaches the Cauchy–Schwarz rate (not proved).
- *Theorem S.* At b=0 the over-occupied startup level (weight 1, P-mass 1/d) carries a label tilt that crosses the gate. It costs p+q≳d a_U²A/64, with a_U≍ρ_B·max(ρ,ρ_B) away from a cancellation. A P-heavy auditor then pays about ρ_B⁴/η·A instead of ρA.
- *Open.* The band ε²/Γ_t≲ρ_B≲ε/max(ρ,ρ_B); k≤3.6 with the startup inside the window; and the factor Γ/ρ_B³ between Theorem S and E4's P route.

**Next steps.**
1. ~~Prove the risk bound.~~ Done: [cold_risk_bound.md](cold_risk_bound.md), Lemma R (C₀=44) and Lemma R′.
2. ~~Build a variance-scaled dynamics family.~~ Done at constant overlap: [variance_scaled_floor.md](variance_scaled_floor.md). Two sticky levels at weights t/c₁, t/c₂ carry target mass ρ/2 each, and the worlds move only the h=1 labels between them. G_k is exactly quadratic in the tilt. Small overlap is done except for one band; see the paragraph above and variance_scaled_floor.md. Still open: that band, and Ψ(t)<Cε.
3. Find a matching estimator. A plug-in route: learn the Q-law of the bulk from proposals and the Q-law of heavy states from target draws via Q(dx)=P(dx)/W(x) (Z known or bracketed), then evaluate the exact IMH path functional. The finite-support recursion of the experiment plan, §2.1, is that evaluator.
