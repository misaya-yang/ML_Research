# Variance-scaled dynamics floor for light windows

Status: Claude derivation, 2026-11-22; needs a human audit. The small-overlap section (Lemma F, V_η, Proposition N, Theorem S) was added the same day, with numerics in `small_overlap_floor.py`. Exact finite-t numerics for Theorem V are in `experiments/finite_state/variance_floor.py`, with the log in `results/local_runs/theory_checks-20261122/variance_floor.log`. This note does step 2 of the open-regime programme in [small_overlap_joint_frontier.md](small_overlap_joint_frontier.md) §6: the lower bound for the case where the tail profile already constrains weights inside the window. It also meets the terms the Pro–Claude synthesis in [cold_risk_bound.md](cold_risk_bound.md) set for such a construction:

- the same overlap and tail class in both worlds;
- the same relative gate;
- the same advertised statistics on both sides;
- only the signed observable/weight pairing differs.

## What is new

The flat weighted floor (long-horizon note, family 2) and Pro's signed 3T-layer witness both rely on target mass that the cold chain never reaches. That mass sits at weight T² or 3T, so Γ_T≍1. In a light window (Γ_t=ρ≪1) this mechanism disappears. By Lemma R′, mass beyond L contributes only τ(L)². By R1, near-threshold instances have Var_P(h)≤k(44Γ_t+ε). The hard instances must therefore be variance-scaled.

The family below has R≈Γ_t/2 and Var≈kR. It lives in every class whose profile allows mass ρ up to weight t. It shows that the dynamics channel costs A·ρ, not A:

$$
p+\frac qt\ \gtrsim\ \rho\,\varepsilon^{-2}\log(1/\delta),\qquad \rho\asymp\Gamma_t .
$$

## Oracle

The oracle is the same as in the long-horizon note:

- Q is public and uniform over N names.
- Each name has a color: its weight level and its h-label. The colors are randomly permuted among the names.
- A target or proposal draw returns the name, its score and its h-label. Normalized scores are allowed, so Z is effectively known.
- Named queries are allowed in the fresh-label/predeclared named-row sense. As N→∞, the response to a named query is within O(K·maxW/N) TV of a fresh proposal draw, where K=p+q.

The lower bound does not cover arbitrary named access to a fixed public finite two-point family. It does not depend on Z being unknown, and it still holds if μ, Var_P(h) and both W laws are disclosed.

## The family

**Parameters.**

- k>1 and β=b/t≥0.
- Two sojourn constants c₁≠c₂, a light weight w_D∈(0,1) and a light label rate s_D⁰∈(0,1). The numerics use w_D=s_D⁰=0.1.
- The first-order sojourn coefficient

$$
\Phi(c,\beta)=\frac{2(c-1+e^{-c})}{c^2}-e^{-c\beta}\,\frac{2}{c^2}\big(1-(1+c)e^{-c}\big),
$$

  with Φ_i=Φ(c_i,β), Φ̄=(Φ₁+Φ₂)/2 and ΔΦ=Φ₁−Φ₂.
- The standing condition

$$
\sigma:=\frac{k\bar\Phi-1}{2}-\frac{s_D^0w_D}{1-w_D}>0 .\tag{K}
$$

  Here Φ(c,β) is the expected squared window occupation, per unit of target mass, of one level with holding scale t/c, starting from a cold chain β windows before the window.
  - At β=∞ it is the stationary two-state value 2(c−1+e^{−c})/c², which tends to 1 as c→0.
  - At β=0 it is at most 0.278, attained near c≈3.
  - So (K) is satisfiable for every k>1 with enough burn-in, and with no burn-in for k>3.6.

**Levels.** Each level is a block of names with one weight:

| Level | Weight | Q-mass | P-mass | h-label rate (fraction of the level with h=1) |
|---|---|---|---|---|
| B (bulk) | 1 | q_B=1−q_D−q_{M₁}−q_{M₂} | q_B | s_B(θ) |
| D (light) | w_D | q_D=(ρ−q_{M₁}−q_{M₂})/(1−w_D) | w_Dq_D≈w_Dρ/(1−w_D) | s_D(θ) |
| M₁ (sticky) | t/c₁ | ρc₁/(2t) | ρ/2 | θ |
| M₂ (sticky) | t/c₂ | ρc₂/(2t) | ρ/2 | 1−θ |

The bulk and light label rates solve two linear equations. These keep P(h=1) and Q(h=1) at their values for θ=½, s_B=σρ, s_D=s_D⁰:

$$
q_Bs_B+q_Ds_D=Y_Q-\tfrac\rho2(\theta-\tfrac12)\frac{c_1-c_2}t,\qquad q_Bs_B+w_Dq_Ds_D=Y_P .
$$

The determinant is (w_D−1)q_Bq_D≠0. The solution moves by O(1/t) in θ, so the label rates stay in (0,1) once t≥C/(σρ).

**Invariants.** The following hold in every world θ:

- (I1) The law of W is the same under P and under Q, since the level masses do not depend on θ.
- (I2) P(h=1) and Q(h=1) are the same. Hence μ=Ph and Var_P(h)=μ(1−μ) are the same.
- (I3) The pair is in the light-window class:
  - E_QW=1 and η=E_Q min(1,W)≥1−ρ;
  - min W=w_D, and the P-median of W is d=1;
  - τ_P(v)≤ρ for v≥1, and τ_P(v)=0 for v≥t/min(c₁,c₂);
  - Γ_t=(ρ/2)[min(1,1/c₁)+min(1,1/c₂)]+O(1/t).

So the worlds differ **only** in which sticky level carries the h=1 labels: the signed pairing of h with W. The numerics confirm (I1) and (I2) to 1e-16.

## Theorem V

Fix (k,β,c₁,c₂,w_D,s_D⁰) satisfying (K). There are constants C, ρ₀>0, depending only on these, such that the following holds for every ε≤1/C, every ρ∈[C|ΔΦ|⁻¹ε, ρ₀] and every t≥C/ε, with b=βt.

**(a) Separated worlds.** There is θ* with |θ*−½|≤Cρ such that θ_±=θ*±Δ with Δ=4ε/(ρ|ΔΦ|)≤1/4 satisfy

$$
G_k(\theta_{-\mathrm{sgn}\Delta\Phi})\le-\varepsilon,\qquad G_k(\theta_{+\mathrm{sgn}\Delta\Phi})\ge+\varepsilon .
$$

Here G_k=R_Q(b,t)−Var_P(h)/k. At θ* the instance has R=(ρ/2)Φ̄(1+O(ρ)) and Var_P(h)=kR.

**(b) Per-draw information.** The worlds θ_± satisfy

$$
\mathrm{KL}_P\le\frac{512\,\varepsilon^2}{\rho\,\Delta\Phi^2}(1+o(1)),\qquad
\mathrm{KL}_Q\le\frac{256(c_1+c_2)\,\varepsilon^2}{\rho\,t\,\Delta\Phi^2}(1+o(1)),
$$

where o(1)→0 as ρ→0 and ρt→∞. The KL_Q bound also covers named queries in the oracle above.

**(c) Necessity.** Suppose an auditor issues with probability at least 1−δ on the safe world and at most δ on the unsafe world. Then

$$
p+\frac{(c_1+c_2)}{2}\cdot\frac qt\ \ge\ \frac{\Delta\Phi^2}{512}\,\rho\,\varepsilon^{-2}\log\frac1{4\delta}\,(1-o(1)).
$$

## Proof

**Step 1: θ-free levels, iid labels, exact quadratic risk.** The weights depend only on the level, so the level of the IMH state is itself a Markov chain. Its transitions depend only on level weights and level Q-masses, which do not depend on θ by (I1).

When the chain moves to a new name, the name's label is drawn independently. The new name may come from a move to another level, or from an accepted proposal within the same level, which refreshes the name. A proposal of the current name counts as a refresh with an unchanged label, and the distribution is the same. The label is Bernoulli with the level's rate: θ, 1−θ, s_B(θ) or s_D(θ). The initial name X₀~Q is labelled the same way.

Call the maximal runs on a single name *segments*, and let o_g be segment g's window occupation as a fraction of t. Conditioning on the level path,

$$
R(\theta)=E\Big[\Big(\sum_g\vartheta_g o_g-\mu\Big)^2\Big]+E\Big[\sum_g\vartheta_g(1-\vartheta_g)o_g^2\Big],
$$

where ϑ_g is the label rate of g's level. The level path, and hence the joint law of {o_g}, does not depend on θ. Every ϑ_g is affine in θ, and μ does not depend on θ. So **R(θ), and therefore G_k(θ)=R(θ)−μ(1−μ)/k, is exactly a quadratic polynomial in θ at every finite t.** The numerics confirm this: the residual from a quadratic fit is 1e-16.

**Step 2: slope and curvature to first order.** Let X_i be the window occupation of level M_i, let S_i=Σ_{g⊂M_i}o_g², and let N be the number of M-segments that meet [−β,1] in window time units.

- *Bounding N.* Entries into M happen at total rate at most ρ(c₁+c₂)/2 per unit time. Refreshes inside M_i happen at rate ρc_i/2. Exits happen at rate (1−ρ)c_i. So P(N≥2)≤C₂ρ², with C₂ depending on (c,β). The start X₀∈M has probability O(ρ/t).
- *Derivatives.* Differentiate Step 1, using that s_B′ and s_D′ are O(1/t):

$$
R'(\theta)=2E\big[m\,(X_1-X_2)\big]+(1-2\theta)E[S_1+S_2]+O(1/t),\qquad
R''=2E\big[(X_1-X_2)^2-S_1-S_2\big]+O(1/t),
$$

  where m=Σ_gϑ_go_g−μ.
- *On {N≤1}.* Here X₁X₂=0 and S_i=X_i². So the curvature integrand vanishes, and |R″|≤4P(N≥2)+O(1/t)≤Cρ²+C/t.
- *The slope.* On {N≤1}, m=ϑX_i+m_bulk with |m_bulk|≤|s_B−μ|+O(ρ)=O(ρ)+fluctuation, so

$$
R'(\theta)=E X_1^2-E X_2^2+O(\rho^2)+O(1/t).
$$

- *The single-segment second moment.* The level process converges to a CTMC in time units of t, with error O(1/t) by the coupling of the long-horizon note: bounded rates and O(1) expected jumps. In the limit:
  - Entries into M_i come at rate (ρ/2)c_i. The sojourn is Exp(c_i(1−ρ)), cut by refreshes at rate O(ρ).
  - A single sojourn entered at window time u contributes |[u,u+E]∩[0,1]|².
  - Integrating u over [−β,1] gives c∫E[|[u,u+E_c]∩[0,1]|²]du=Φ(c,β), using E min(E,s)²=(2/c²)(1−e^{−cs}(1+cs)) and memorylessness for u<0.

  Hence E X_i²=(ρ/2)Φ_i+O(ρ²)+O(1/t), and

$$
G_k'(\theta)=\tfrac\rho2\,\Delta\Phi\,(1+O(\rho))+O(1/t),\qquad |G_k''|\le C\rho^2+C/t .
$$

**Step 3: calibration and separation.**

- At θ=½, R=(ρ/2)Φ̄+O(ρ²)+O(1/t). Here the bulk-label term Σϑ(1−ϑ)o² over bulk segments is O(1/t), and the mean part on {N=0} is O(ρ²).
- Also μ=ρ/2+σρ+s_D⁰w_Dρ/(1−w_D)+O(ρ²)=ρkΦ̄/2+O(ρ²), by (K). So G_k(½)=O(ρ²)+O(1/t).
- The slope is of order ρ|ΔΦ| and the curvature O(ρ²), so G_k has a root θ* with |θ*−½|≤Cρ/|ΔΦ|. This lies in [3/8,5/8] for ρ≤ρ₀.
- Since G_k is quadratic,

$$
G_k(\theta^*\pm\Delta)=\pm\Delta\,G_k'(\theta^*)+\tfrac12G_k''\Delta^2 .
$$

  With Δ=4ε/(ρ|ΔΦ|), the first term has absolute value at least 2ε(1−Cρ) and the second at most Cε²/ΔΦ². So |G_k|≥ε with the stated signs, once ρ≤ρ₀ and ε≤1/C. The requirement Δ≤1/4 is ρ≥16ε/|ΔΦ|.
- At θ*, Var_P(h)=kR exactly, and R=(ρ/2)Φ̄(1+O(ρ)).

**Step 4: information.**

- *Target draws.* A target draw observes a cell (level, label). The four sticky cells (M_i, h) have P-masses θρ/2, (1−θ)ρ/2 and their complements within ρ/2. Each moves by ρΔ between θ_±, and each has mass at least ρ/8 when θ∈[1/4,3/4]. So the sticky cells contribute χ²≤4(ρΔ)²/(ρ/8)=32ρΔ²=512ε²/(ρΔΦ²).
- *Bulk and light cells.* Their label rates move by O(Δ/t). Since s_B≥σρ/2 and s_D≥s_D⁰/2, they add O(Δ²/(σρt²)), which is o(ρΔ²) once ρt→∞.
- *Proposal draws.* The sticky Q-cells are the P-cells scaled by c_i/t. They give χ²≤Σ_i2(ρΔc_i/t)²/(ρc_i/(8t))=16ρΔ²(c₁+c₂)/t. The bulk and light Q-cells move by O(ρΔ/t) over masses of order σρ and ρ. They add O(Δ²/(σt²)), which is o(ρΔ²/t).

Then KL≤χ² gives (b).

Part (c) follows as in the long-horizon note:

1. The name-permutation coupling puts every adaptive transcript, including named queries and caches, within TV δ/4 of the fresh-color transcript.
2. The KL chain rule charges p·KL_P+q·KL_Q.
3. By Bretagnolle–Huber, two-sided error δ forces total KL≥log(1/(4δ)), up to the TV slack. ∎

**Quadratic-regime constants.** In the quadratic regime, KL≈χ²/2 with the θ=½ masses ρ/4. The exact constants are then 128 for P and 64(c₁+c₂) for Q. The numerics give 141–146 and 352–355 for (c₁,c₂)=(1,4), and 919–951 for (3,10), against theory values 320 and 832. The excess is the nonlinearity of a tilt of Δ=1/4.

## Corollaries

**V1 (light-window dynamics floor in profile form).**

*Setting.* C(Ψ,η₀) is a tail-profile class with η₀≤0.9. Choose c₁,c₂≥1, so every weight is at most t. For example:

- (c₁,c₂)=(1,4) with β=2 satisfies (K) for every k≥2;
- (c₁,c₂)=(3,10) with β=0 satisfies (K) for k≥4.7.

*Statement.* With ρ:=min{Ψ(t), ρ₀} and ρ≥Cε, every auditor that is valid and powerful on C(Ψ,η₀) at window (βt,t) satisfies

$$
p+\frac qt\ \ge\ c\,\varepsilon^{-2}\log(1/\delta)\cdot\min\{\Psi(t),\rho_0\} .
$$

Here c and ρ₀ depend on (k,β,c₁,c₂). With σ fixed, ρ₀ is of order 1/k, because the bulk label rate σρ must stay bounded. So the bound reads A·min{Ψ(t), c′/k}. The family uses overlap at least 1−ρ, min W=0.1 and τ_P≤ρ on [1,t]. So it lies in the class for every Ψ with Ψ(t)≥ρ, at any η₀≤0.9 and any ω≤0.1.

**V2 (constant-overlap light windows; the upper half is a sketch).** Fix η₀ and assume (Ψ2). Then sup_{C(Ψ,η₀)}Γ_t≤1/t+C_ΨΨ(t).

- *Sufficiency.* [cold_risk_bound.md](cold_risk_bound.md) R4 with full-window replicates and E4 jumps (c_P=O(1/η₀)), together with E1, gives

$$
p\gtrsim A\max\{\Psi(q/p),\varepsilon\},\qquad \frac{p}{k\,c_P}+\frac qL\gtrsim A\max\{\Gamma_t,\varepsilon\}.
$$

  A full-window replicate Ā² has variance at most R≤44Γ_t, so the Q route carries no factor k.
- *Necessity.* V1 together with E1/E1′.
- *Together.* When Ψ(t)≥Cε, the dynamics channel of the constant-overlap light-window class is

$$
p+\frac qt\ \asymp\ \varepsilon^{-2}\log(1/\delta)\cdot\Psi(t)
$$

  up to constants depending on (k,β,η₀,C_Ψ,C_d), and the factor L/t=1+β on the direct-path side.
- *Caveats.* This characterization is only as solid as R4, which is still a sketch. E4's high-only jump is an approximate replication with bias O(1/(tη₀)); it must be budgeted, not treated as exact. In the flat regime Ψ≡1, it reduces to (D) with η₀ constant.

## Numerical checks (`variance_floor.py`)

*Setup.* The chain is the exact 8-cell chain (B±, D±, M₁±, M₂±), computed by backward recursion for the risk. The root θ* is found by bisection. The tilt is Δ=1/4 and ε=ρ|ΔΦ|/16. The grid is t∈{2^10, 2^12, 2^14} × ρ∈{0.01, 0.05}.

| Case | G(θ*∓Δ)/ε | R/ρ vs Φ̄/2 | Var/(kR) at θ* | Γ_t/ρ | η | Invariants (W law; P(h=1); Q(h=1)) | KL_P·ρΔΦ²/ε² | KL_Q·ρtΔΦ²/ε² |
|---|---|---|---|---|---|---|---|---|
| k=2, β=2, c=(1,4) | −1.90…−1.98 / +1.93…+1.99 | 0.253–0.259 vs 0.260 | 1.000 | 0.63–0.72 | 0.95–0.99 | ≤2e-16 | 141 | 352–354 |
| k=4, β=2, c=(1,4) | −1.80…−1.96 / +1.83…+1.97 | 0.240–0.256 vs 0.260 | 1.000 | 0.63–0.72 | 0.95–0.99 | ≤1e-16 | 141–143 | 352–357 |
| k=6, β=0, c=(3,10) | −1.90…−1.99 / +1.95…+2.00 | 0.105–0.108 vs 0.109 | 1.000 | 0.22–0.31 | 0.95–0.99 | ≤1e-16 | 141–146 | 919–951 |

*Quadratic structure* (separate check at t=4096, ρ∈{0.01, 0.02, 0.04}):

- G(θ) is quadratic to a residual of 1e-16.
- The slope over (ρ/2)ΔΦ is 0.97–0.995.
- The curvature over ρ² is 0.12–0.18.

All constants are flat in t, and they agree across ρ up to the predicted O(ρ) corrections.

## Relation to the earlier witnesses

| | Flat weighted floor (long-horizon family 2) | Pro's signed 3T-layer witness | This note |
|---|---|---|---|
| Where the target mass sits | weight T², never visited | weight 3T, never visited | weights 1 and t/c_i, all visited |
| Γ_t | ≍1 | ≍1 | ≍ρ |
| What differs between worlds | Q-mass of the middle level | signed mark pairing | signed label pairing on the sticky levels |
| Invariant marginals | Ph, Var | W laws, both h marginals | W laws (P and Q), both h marginals, μ, Var |
| Floor | η₀p+q/T≳A | η₀p+q/T≳A | p+q/t≳ρA |

## Small overlap: what can and cannot be witnessed

Status: Claude derivation, 2026-11-22; needs a human audit. Exact numerics are in `experiments/finite_state/small_overlap_floor.py`, with the log in `results/local_runs/theory_checks-20261122/small_overlap_floor.log`. This section keeps Theorem V's gate, public information (both W laws, μ, Var disclosed) and sampling permissions (P draws, Q draws, fresh-label named rows). It drives the actual overlap η to zero, with both worlds sharing one W law in a tail class that decays inside the window.

Two earlier inputs are used without redoing them:

- Pro's first-mark split (cold_risk_bound.md, Pro–Claude synthesis) isolates the pre-mark segment as the only place where cold and stationary risk differ. That segment is where the mechanism of Theorem S lives.
- GPT's replay `experiments/finite_state/pro_cold_core.py` reproduces `results/pro/pro_cold_core.csv` byte for byte. Pro's 3T-layer witness is a flat-class (Γ≍1) construction, so it is not used here.

**Notation.** d is the P-median of W and ρ_B:=d/t. Then (d) of cold_risk_bound.md gives ρ_B/2≤Γ_t when d≤t. The quantities A₁, φ₀, Φ̄₀ and β_B are defined in Theorem S.

**Outcome.**

| Regime | Result |
|---|---|
| ηt<1/(4γ) and the class asks Γ_t≤γ<½ | **impossible**: small overlap and a light window cannot coexist (Lemma F) |
| any η, with ρ≥Cε, dρ≥C, ρ+ρ_B≤c₀ | Theorem V_η: p+q/t≳ρA. The sticky-label floor keeps P-coefficient 1 at every overlap; it does **not** become ηp+q/t |
| ρ_Be^{−b/(2d)}≤ε²/(660Γ_t) (long horizon, or burn-in b≥2d·log(660Γ_tρ_B/ε²)) | **impossible to witness λ<1**: every disclosed-marginal pair has KL_P≥ε²/(330Γ_t), so it certifies at most p≳330Γ_tε⁻²log(1/(4δ)) (Proposition N) |
| b=0, ρ_B·\|β_Bρ_B−(1−s_B)A₁ρ\|≥8ε, kΦ̄₀>1, 2kρ_B<1 | Theorem S: p+q≥(d a_U²/64)·ε⁻²log(1/(4δ)). With ρ_B≳ρ, P-heavy auditors need about ρ_B⁴/η·A, which exceeds ρA once η≪ρ_B⁴/ρ |
| ε²/Γ_t≲ρ_B≲ε/max(ρ,ρ_B) | open |
| k≤1/max_cΦ(c,0)≈3.6 with the startup inside the window (b≲d) | not covered |

### Lemma F (feasibility and the impossible interval)

For every equivalent pair with overlap η and window (b,t):

1. P(W≤D)≤Dη (Observation C2), so d≥1/(2η).
2. min(t,d)≤2tΓ_t (fact (d) of cold_risk_bound.md).
3. Hence, if Γ_t≤γ<½, then d≤2γt (otherwise Γ_t≥½). Combined with step 1, ηt≥1/(4γ).

So no pair with ηt<1/(4γ) has Γ_t≤γ. Near the threshold, R1 adds only Var_P(h)≤k(44Γ_t+ε), which a low-variance h always satisfies. So **the only parameter obstruction to "small overlap + light window + near threshold" is ηt<1/(4γ)**. Above it, the base family below realises the regime with ρ_B≍1/(ηt), that is, at the edge of step 3. ∎

### The small-overlap base B(t,d,ρ;c₁,c₂)

One W law, five levels. Every level is a block of names; labels are per-level Bernoulli rates, as in Theorem V.

| Level | Weight | Q-mass | P-mass | Label rate |
|---|---|---|---|---|
| dark D | 1/d² | ≈1−2/d | ≈1/d² | ϑ_D (compensator) |
| startup U | 1 | 1/d | 1/d | ϑ_U |
| bulk B | d | ≈(1−ρ)/d | ≈1−ρ | s_B (calibrator and compensator) |
| sticky M_i | t/c_i | ρc_i/(2t) | ρ/2 | θ_i |

The dark and bulk Q-masses solve Σq=1 and Σqw=1 exactly.

**Properties.**

- Overlap: η=(2−ρ)/d+O(d⁻²+ρ/t). Any η is reached by choosing d=ρ_Bt.
- Median and tail: the P-median is d, and τ_P(v)=1−O(1/d) for v<d. Also τ_P(v)≤ρ on [d,t/c₁) and τ_P(v)=0 beyond.
- Window quantity: Γ_t=(1−ρ)ρ_B+(ρ/2)Σ_i min(1,1/c_i)+O(1/t).
- Class membership: B lies in C(Ψ,η₀) for every η₀≤η and every Ψ with Ψ≥1−2/d on [1,d) and Ψ≥ρ on [d,t/min c_i]. One example satisfying (Ψ1)–(Ψ3) is Ψ(D)=max{ρ, min(1,(d/D)^r)} with r<1.
- Calibration: putting G_k=0 with the bulk label needs the bulk to absorb variance faster than it adds risk. The bulk alone has R/Var≈2ρ_B, so this requires 2kρ_B<1 together with dρ≫1. Where this fails, the numerics report n/a.

**Structural fact.** As in Theorem V Step 1, the level chain does not depend on the labels, so for every finite t

$$
R(\vartheta)=\vartheta^{\!\top}C\vartheta+\sum_\ell\vartheta_\ell(1-\vartheta_\ell)S_\ell,\qquad
C_{\ell\ell'}=E[(O_\ell-p_\ell)(O_{\ell'}-p_{\ell'})],\quad S_\ell=E\textstyle\sum_{g\subset\ell}o_g^2 .
$$

The script computes C and S by an exact backward recursion on the 5-level chain. The result agrees with an explicit 10-cell chain to 1e-15.

### Theorem V_η (the sticky-label floor survives small overlap)

On B with labels θ on M₁ and 1−θ on M₂, ϑ_U=½, and dark and bulk labels compensating μ and Q(h=1), the conclusions of Theorem V hold with error terms O(ρ²+ρρ_B+ρ_B²) in place of O(ρ²):

- G_k(θ*∓Δ)≶∓ε;
- KL_P≤C_Vε²/ρ and KL_Q≤C_V′ε²/(ρt);
- p+(c₁+c₂)q/(2t)≥c_Vρε⁻²log(1/(4δ)).

The constants are uniform in η.

*Proof delta.* Steps 1 and 4 are verbatim. In Step 2 the startup adds occupation O(ρ_B) before the bulk phase. Its cross terms with X₁−X₂ are O(ρρ_B), and its square is O(ρ_B²). The bulk label noise is O(s_Bρ_B). None of these touches the leading slope (ρ/2)ΔΦ₀. ∎

### Proposition N (where P draws cannot be blinded)

**Setting.** Two instances share the W law under P and Q, hence the level chain, and share μ. They differ only in per-level label rates ϑ⁰ and ϑ¹, with |G_k¹−G_k⁰|≥2ε. Let

$$
e_b:=\frac1t\sum_{i=b}^{b+t-1}\lambda(d)^i\le\min(1,2d/t)\,e^{-b/(2d)} .
$$

**Statement.**

$$
\mathrm{KL}_P\ \ge\ \frac{\varepsilon^2}{165\,\Gamma_t}-\frac{e_b}{2}.
$$

In particular, if e_b≤ε²/(330Γ_t), then KL_P≥ε²/(330Γ_t). Such a pair forces at most p≥330Γ_tε⁻²log(1/(4δ)). So no disclosed-marginal witness can show a P-coefficient below an absolute constant in that regime.

*Proof.*

1. *Path.* Put Δ=ϑ¹−ϑ⁰ and ϑ^s=ϑ⁰+sΔ. Then μ is constant along the segment, so Σ_ℓp_ℓΔ_ℓ=0.
2. *Derivative.* By the structural fact, dR/ds=2E[m_sΣ_ℓΔ_ℓO_ℓ]+E[Σ_g(1−2ϑ^s_g)Δ_{ℓ_g}o_g²], with m_s=Σ_ℓϑ^s_ℓ(O_ℓ−p_ℓ).
3. *First term.* By Cauchy–Schwarz and Jensen, using Σ_ℓO_ℓ=1,

$$
|E[m_s\textstyle\sum\Delta_\ell O_\ell]|\le\sqrt{E m_s^2}\sqrt{\sum_\ell\Delta_\ell^2EO_\ell}.
$$

   Also Em_s²≤R(ϑ^s)≤44Γ_t by Lemma R.
4. *Second term.* By Cauchy–Schwarz, it is at most √(Σ_ℓΔ_ℓ²EO_ℓ)·√(EΣ_go_g²). Here EΣ_go_g²≤24Γ_t, because Σ_g len_g²≤2Σ_iF_i, where F_i is the residual run length from time i. Its conditional mean is at most s(w(X_i)), and term C of Lemma R gives the bound.
5. *Combining.* 2ε≤|R¹−R⁰|≤(2√44+√24)√(Γ_tΣ_ℓΔ_ℓ²EO_ℓ)=18.17√(Γ_tΣ_ℓΔ_ℓ²EO_ℓ).
6. *Occupation.* EO_ℓ≤4p_ℓ for w_ℓ≥d (Lemma D). For w_ℓ<d, EO_ℓ≤2p_ℓ+e_ℓ with Σ_{w_ℓ<d}e_ℓ≤e_b. To see this, use Lemma H:
   - X_i lands below d with more than twice its P-mass only if it is still at X₀ (then V_i<w(X₀)<d) or was refreshed from P(·|W≤V_i) with V_i<d.
   - Both events lie in {V_i<d}, which has probability λ(d)^i.
   - On {V_i≥d}, the refresh density is at most 2.

   Hence Σ_ℓΔ_ℓ²EO_ℓ≤4χ̃+e_b, where χ̃:=Σ_ℓp_ℓΔ_ℓ² (using |Δ_ℓ|≤1).
7. *Information.* A target draw observes (level, label), so Pinsker gives KL_P=Σ_ℓp_ℓ kl(ϑ⁰_ℓ‖ϑ¹_ℓ)≥2χ̃.
8. *Conclusion.* From steps 5 and 6, 4ε²≤330Γ_t(4χ̃+e_b), so χ̃≥ε²/(330Γ_t)−e_b/4. Step 7 finishes the proof. ∎

**Reading.** Burn-in beyond the overlap scale, b≳2d·log(Γρ_B/ε²), is enough. So is a long horizon at b=0, ρ_B≲ε²/(660Γ_t), which by Lemma F needs ηt≳Γ_t/ε². In either case P draws see every label difference that moves the gate. The conjectured coefficient λ≍1/c_P (E4) therefore cannot be witnessed there under the disclosed-marginal rules. If a plug-in label estimator attains the Cauchy–Schwarz rate (plausible, not proved), the conjecture is false in that regime.

The numerics check steps 5 and 6 directly, on 200 random compensated label pairs per row:

- the ratio |R¹−R⁰|/(18.17√(Γ_tΣΔ²EO)) is at most 0.023;
- the low-level excess divided by e_b is at most 0.998.

### Theorem S (startup-label witness)

**Setting.** Take b=0, the base B with sticky labels θ₁=θ₂=1, and reference labels ϑ_D=ϑ_U=½. The bulk label s_B is calibrated so that G_k=0. The worlds are ϑ_U=½±Δ, with dark and bulk labels solving the two linear equations that keep μ and Q(h=1) fixed. Both corrections are −Δ/d+O(Δ/d²). Define

$$
\varphi_0(c)=1-\frac{1-e^{-c}}{c},\qquad A_1=\sum_i\tfrac12\big(1-\varphi_0(c_i)\big),\qquad \bar\Phi_0=\tfrac12\sum_i\Phi(c_i,0),\qquad \beta_B=\tfrac{\vartheta_D}2+\vartheta_U+\tfrac12-\tfrac52s_B .
$$

**Statement.** Assume kΦ̄₀>1, 2kρ_B<1, dρ≥C, ρ+ρ_B≤c₀ and t≥C/ε.

- **(a) Invariants.** Both worlds share the W laws under P and Q, the overlap η=(2−ρ)/d, the tail class, μ, Var_P(h) and Q(h=1).
- **(b) Slope and curvature.** G_k is exactly quadratic in ϑ_U, with

$$
a_U=\rho_B\big[\beta_B\rho_B-(1-s_B)A_1\rho\big]+\frac{2(1-s_B)\rho\bar\Phi_0}{d}\;+\;O\big(\rho_B(\rho+\rho_B)^2+t^{-1}\big),\qquad |G_k''|\le2\rho_B^2 .
$$

- **(c) Separation.** For every ε≤min{|a_U|/8, a_U²/(4ρ_B²)}, the tilt Δ=2ε/|a_U| gives G_k≤−ε in one world and G_k≥+ε in the other.
- **(d) Information.** KL_P and KL_Q are each at most 64ε²/(d a_U²)·(1+O(Δ²+1/(dρ))). At Δ=¼ the exact value is kl(¼‖¾)/d=0.549/d for both.
- **(e) Necessity.**

$$
p+q\ \ge\ \frac{d\,a_U^2}{64}\,\varepsilon^{-2}\log\frac1{4\delta}\,\big(1-o(1)\big).
$$

  If ρ_B≥2(1−s_B)A₁ρ/β_B, then a_U≥β_Bρ_B²/2 and the right side is at least (β_B²/256)·ρ_B⁵t·ε⁻²log(1/(4δ)), which is about (β_B²/128)ρ_B⁴η⁻¹·ε⁻²log(1/(4δ)).

*Proof.*

**Step 1 (exact quadratic).** Use the structural fact. The compensators are affine in ϑ_U.

**Step 2 (startup structure, in window units, b=0).**

- *Dark phase.* X₀ is dark with probability 1−O(1/d). The dark phase lasts until a U, bulk or sticky proposal, which is an exponential with mean ρ_B/2.
- *Startup level.* With probability ½ the next level is U. Its sojourn is exponential with mean ρ_B and ends at a bulk or sticky proposal. Refreshes inside U cut it into segments, with on average 2 segments of mean ρ_B/2.
- *Sticky entries.* A sticky name is accepted from every lower level. So the sticky process is blind to the startup, except that a sticky entry ends the startup, which is an O(ρρ_B²) effect.
- *Moments.* EO_D=EO_U=ρ_B/2, EO_U²=ρ_B², Cov(O_D,O_U)=0, S_U=ρ_B²/2, and EO_{M_i}−ρ/2=−(ρ/2)(1−φ₀(c_i))+O(ρρ_B), all up to O(ρ_B(ρ+ρ_B)) relative corrections.

**Step 3 (slope).**

- Write m=Σ_ℓ(ϑ_ℓ−s_B)(O_ℓ−p_ℓ). Then the bias is

$$
b_0=Em=(\vartheta_D-s_B)\tfrac{\rho_B}2+(\vartheta_U-s_B)\tfrac{\rho_B}2-(1-s_B)A_1\rho .
$$

- The compensated direction is dm=(O_U−p_U)−d⁻¹[(O_D−p_D)+(O_B−p_B)].
- From the moments of Step 2,

$$
a_U=2E[m\,dm]+(1-2\vartheta_U)S_U+\dots=b_0\rho_B+\tfrac32(\vartheta_U-s_B)\rho_B^2+(1-2\vartheta_U)\tfrac{\rho_B^2}2+\tfrac2d(1-s_B)\rho\bar\Phi_0 .
$$

  The last term comes from E[m(O_B−p_B)]≈−(1−s_B)E(O_M−ρ)²=−(1−s_B)ρΦ̄₀.
- Collecting the ρ_B² terms gives β_B.

**Step 4 (curvature).** G_k″=2E(dm)²−2S_U+O(1/d)=2ρ_B²−ρ_B²+…, so |G_k″|≤2ρ_B² for small ρ+ρ_B. Numerically the ratio is about 1.2ρ_B².

**Step 5 (separation).**

$$
G_k(\tfrac12\pm\Delta)=\pm a_U\Delta+\tfrac12G_k''\Delta^2 .
$$

With Δ=2ε/|a_U| the first term has absolute value 2ε and the second at most 4ε²ρ_B²/a_U²≤ε.

**Step 6 (information).**

- The U cells have equal P- and Q-masses 1/d, and their label rates differ by 2Δ, so χ²≤16Δ²/(d(1−4Δ²)).
- The dark corrections act on Q-mass ≈1 with label-rate changes O(Δ/d). They add O(Δ²/d²) to χ²_Q.
- The bulk corrections act on P-mass ≈1−ρ with label rate s_B≍ρ. They add O(Δ²/(d²ρ)) to χ²_P.
- Then KL≤χ².

**Step 7 (necessity).** This is as in Theorem V (c): name-permutation coupling, the KL chain rule and Bretagnolle–Huber. ∎

**What Theorem S adds to V_η.**

- Together the two give the necessary region

$$
\{p+q/t\gtrsim\rho A\}\cap\Big\{p+q\gtrsim\tfrac{d a_U^2}{64}A\Big\}.
$$

- Theorem S binds only while q<d a_U²A/64. That is a small Q budget, since d a_U²/64≪ρt. There it raises the P-only cost from ρA to max{ρ, d a_U²/64}·A.
- In λ language, a P-heavy auditor has λ_eff≲min{1, Cρη/ρ_B⁴}.
- E4's route gives λ≳η/ρ_B (c_P≈1+d/(ηt)). The remaining gap is a factor Γ/ρ_B³.
- The gain comes from the over-occupied startup level U, not from a factor η in front of p.

### Numerics (`small_overlap_floor.py`)

*Setup.* k=6, b=0, (c₁,c₂)=(3,10), u=1 and w_U=1. Then A₁=0.2084 and Φ(c_i,0)=(0.2775, 0.1600). The tilt is Δ=¼, and ε:=min|G_k(±)| is the honest margin. The column K is KL_P^V/ε² divided by KL_P^S/ε². The invariant column is the largest change in μ and Q(h=1) between the two worlds.

| t | ρ | ρ_B | d | η | Γ_t | a_U/pred | S: G∓/ε | S: KL·d/Δ² (P, Q) | V_η: G∓/ε | K | Invariants |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^14 | 0.05 | 0.050 | 819 | 2.40e-3 | 0.058 | 0.999 | −1.00 / +1.36 | 8.84 / 8.80 | −1.00 / +1.03 | 28.5 | 5e-17 |
| 2^16 | 0.05 | 0.050 | 3277 | 6.00e-4 | 0.058 | 1.010 | −1.00 / +1.35 | 8.80 / 8.79 | −1.00 / +1.03 | 116 | 5e-17 |
| 2^18 | 0.05 | 0.050 | 13107 | 1.50e-4 | 0.058 | 1.013 | −1.00 / +1.35 | 8.79 / 8.79 | −1.00 / +1.03 | 468 | 1e-17 |
| 2^18 | 0.10 | 0.050 | 13107 | 1.47e-4 | 0.067 | 0.995 | −1.00 / +1.62 | 8.79 / 8.79 | −1.00 / +1.07 | 110 | 1e-17 |
| 2^18 | 0.05 | 0.025 | 6554 | 2.99e-4 | 0.035 | 1.027 | −1.00 / +1.43 | 8.80 / 8.79 | −1.00 / +1.03 | 8.2 | 1e-16 |
| 2^14–2^18 | 0.05, 0.10 | ρ/16 | 102–1638 | — | — | −4.0 … +1.7 (a_U near its cancellation) | crosses (sign of a_U flips) | 8.8–9.5 | −1.00 / +1.03…+1.06 | 0.0 | ≤6e-17 |

*Notes.*

- At Δ=¼ the exact Bernoulli value of KL·d/Δ² is 8.79.
- Four rows are n/a because no bulk label calibrates the gate: the three rows with ρ_B=ρ=0.1 violate 2kρ_B<1 (2kρ_B=1.2), and (t,ρ,ρ_B)=(2^14, 0.05, ρ/16) has dρ≈2.6.
- In the ρ_B=ρ/16 rows, a_U sits near the cancellation β_Bρ_B≈(1−s_B)A₁ρ, so Theorem S gives nothing (K≈0) while V_η still crosses. This is the open band.
- K grows linearly in t at fixed (ρ,ρ_B), as (e) predicts (d a_U²∝ρ_B⁵t).
- The quadratic-form risk equals the explicit cell chain to 1e-15.

## Open

1. *Small overlap.* Settled except for one band, in the section above:
   - The sticky floor keeps P-coefficient 1 at every η (V_η).
   - λ<1 cannot be witnessed once ρ_Be^{−b/(2d)}≤ε²/(660Γ_t) (Proposition N).
   - λ<1 is witnessed by startup labels when ρ_B·max(ρ,ρ_B)≳ε (Theorem S).

   Still open: the band ε²/Γ_t≲ρ_B≲ε/max(ρ,ρ_B), where Proposition N's Cauchy–Schwarz step is probably loose. The bias |EĀ|≤CΓ_t is linear, not √Γ_t, which suggests the true threshold is ρ_B≍ε/Γ_t. Also open: k≤3.6 with the startup inside the window, and the gap Γ/ρ_B³ between Theorem S and E4's P route.
2. *Very light windows (Ψ(t)<Cε).* The family needs ρ≥16ε/|ΔΦ|. Whether q/t≳ℓ/ε still holds there is open. The upper bound there is about LAε for q.
3. *Dependence on k.* The lower constant is uniform in k≥k₀(β,c), but ρ₀≍1/k. On the upper side, the P route carries a factor k. Closing this needs either a two-time replicate without the Var term or a family with ρ≍1/k-free calibration.
4. *Rigour of Step 2.* The O(ρ²) remainders and the O(1/t) CTMC coupling follow the flat family's arguments. Explicit constants are not written out. Exactness of the quadratic structure (Step 1) makes this a one-variable check. The same holds for Theorem S Step 2: the startup moments are leading order with O(ρ_B(ρ+ρ_B)) relative corrections. The exact recursion gives a_U/pred∈[0.93,1.03] wherever ρ_B≥ρ/2.
