# Fixed-target power: general audits, rates, and the disclosure dichotomy

Status: derivation by Claude, 2026-11-22, extending Part I of [fixed_target_power.md](fixed_target_power.md). Needs an independent audit. A Monte Carlo check of the identity E B₊=μ^p for real-valued h is described in §1; the numerical checks are X2 and X3 in [the experiment plan](../../experiments/EXPERIMENT_PLAN_zh.md).

Notation follows Part I: p scored target draws, q proposal draws or named scores, unknown raw scale Z, and B₊=e_p(Sh)/e_p(S), B₋=e_p(S(1−h))/e_p(S) over the pooled rows.

## 1. Any one-dimensional center audit has the same exact ceiling

Let γ be a vector of bounded functionals of the law of (h(X₀),…,h(X_{L−1})) on the cold path. Examples are (a,d)=(EA,EA²) and b=E h(X_{L−1}). An **interval audit** issues only if μ=Ph lies in an open interval I(γ)=(ℓ(γ),u(γ)), where ℓ and u are continuous. Instances with μ∉I(γ) are unsafe.

| Audit | γ | I(γ) |
|---|---|---|
| ESS gate G_k<0 (binary h) | (a,d) | roots of F_k(m)=d−(2a+1/k)m+(1+1/k)m² |
| Output-marginal bias \|b−μ\|<τ | b | (b−τ,b+τ) |
| Window-average bias \|a−μ\|<τ | a | (a−τ,a+τ) |

For the MSE gate, F_k(0)=d>0 and F_k(1)=E(1−A)²>0, so the safe set is exactly the open root interval, as in Part I.

At a base instance with γ₀ and μ₀∈I(γ₀), define

$$
\kappa_0=1-\ell(\gamma_0)/\mu_0,\qquad \kappa_1=1-(1-u(\gamma_0))/(1-\mu_0),\qquad \kappa=\min(\kappa_0,\kappa_1),
$$

with κ_j=1 when that side lies outside [0,1].

**Theorem B1.** Assume the nonatomic-refinement condition of Part I on a minimizing label. Then every honest issuer has power at most Π_p=min{1,δ/(1−κ)^p} at the base, for every finite q, even with adaptive named probes. The procedure below is honest for every finite q, and its power tends to Π_p as q→∞ at every such base.

*Necessity.* This is Part I verbatim. Contaminate by P₁=(1−α)P₀+αP₀(·|D) with D⊂{h=j}, α∈(κ_j,1) and Q(D)→0. The mean m₁ lies strictly outside I(γ₀). Coupling common proposals and uniforms changes the path law by at most L·Q(D) in total variation, so γ₁→γ₀. By continuity, m₁∉I(γ₁) for small Q(D). The likelihood ratio on no-hit transcripts is (1−α)^{n_P}≥(1−α)^p.

*Attainability.*

- Q-only cold paths give a confidence set Γ̂∋γ₀ with probability at least 1−γ_q.
- Put l=sup_{Γ̂}ℓ and v=inf_{Γ̂}u. On coverage, an unsafe instance has μ≤ℓ(γ)≤l or μ≥u(γ)≥v.
- B₊ and B₋ are exactly unbiased for μ^p and (1−μ)^p. This holds for real h∈[0,1] as well as binary h. Conditional on the pooled values, the P positions form a p-subset with probability ∝∏S_i, so E[∏_{j≤p}h(Y_j) | pooled]=e_p(Sh)/e_p(S), while E∏h(Y_j)=μ^p. A 2×10⁵-replicate Monte Carlo on a 5-state real-h example gave .15968 against μ^p=.15948, within about one standard error.
- So E=min(B₊/l^p, B₋/(1−v)^p) is an e-value on the unsafe set. Issue with probability min(1,(δ−γ_q)E).
- Continuity and shrinking Γ̂ give l→ℓ(γ₀) and v→u(γ₀). Part I's U-statistic limit gives B₊→μ₀^p and B₋→(1−μ₀)^p.

**Consequences.**

- (i) Part I holds for every k>1 with (ℓ_k,u_k), the roots of F_k.
- (ii) The bias audit, which is the natural criterion for a generative sampler that returns one output after L−1 transitions, has the same exact ceiling with (ℓ,u)=(b−τ,b+τ).
- (iii) With p=0, no honest issuer exceeds power δ for any audit of this form and any q.

## 2. Pointwise ε⁻¹ versus uniform ε⁻²

Take the ESS gate with binary h. Because F_k is quadratic with leading coefficient 1+1/k,

$$
|G_k|=(1+1/k)(\mu_0-\ell)(u-\mu_0).
$$

Also κ₀=(μ₀−ℓ)/μ₀≥μ₀−ℓ and κ₁=(u−μ₀)/(1−μ₀)≥u−μ₀. Since both factors are at most 1,

$$
\kappa\ \ge\ \min(\mu_0-\ell,\ u-\mu_0)\ \ge\ |G_k|/(1+1/k).
$$

**Proposition B2.**

- (a) On every instance with G_k≤−ε, the procedure of §1 reaches limiting power ≥1−β as soon as p≥(1+1/k)ε⁻¹log((1−β)/δ). This uses −log(1−κ)≥κ.
- (b) Near a simple root, with μ₀ and the root gap bounded away from 0 and 1, κ₀=ε/[(1+1/k)μ₀(u−μ₀)]. Every honest issuer then needs p≥(1−κ)(1+1/k)μ₀(u−μ₀)·ε⁻¹log((1−β)/δ). This uses −log(1−κ)≤κ/(1−κ).
- (c) For every fixed finite q, the uniform lower bounds still require p≥cε⁻²log(1/δ). See the center floor in [long_horizon_two_source_frontier.md](long_horizon_two_source_frontier.md); for general k the root becomes μ_*=(k+1)^{−1/2}.

The three statements are compatible. The order of quantifiers differs: per instance q→∞, versus the worst case over the class at fixed q. **Reading:** unlimited proposal scoring can at best turn two-sided uncertainty about the target center (ε⁻²) into one-sided uncertainty about hidden contamination (ε⁻¹). It can never remove the need for target draws.

## 3. Disclosure dichotomy

**Proposition B3.** Fix an interval audit and a strictly safe base with 0<ℓ(γ₀)<μ₀<u(γ₀)<1.

- (a) **Z disclosed exactly.** Some issuer that is honest over all equivalent pairs uses **p=0** and has power →1 as q→∞.
- (b) **Z unknown.** Every honest issuer has power at most min{1,δ/(1−κ)^p} for every q, with equality in the limit (Theorem B1). In particular, power is at most δ when p=0.

*Proof of (a).*

1. With W=S/Z known, put Y=Wh≥0 and Y'=W(1−h)≥0, so μ=E_QY and 1−μ=E_QY'.
2. For a truncation level c, Hoeffding on Y∧c∈[0,c] gives a lower confidence bound mean(Y∧c)−c√(log(4/γ)/(2q)). It is valid for every distribution, because E[Y∧c]≤EY.
3. Do the same for Y', and set μ_L=LCB(Y) and μ_U=1−LCB(Y').
4. Take c_q=q^{1/4}. The width c_q√(log q/q)→0, and the truncation bias →0 by monotone convergence (E_QW=1).
5. Issue exactly when [μ_L,μ_U]⊂(l,v), where (l,v) comes from the Q-path confidence set of §1.
6. On joint coverage (probability at least 1−δ), issuing implies μ∈I(γ). So the issuer is honest, and its power →1 because both intervals shrink.

**Not a contradiction with the uniform lower bounds.** Hidden heavy names with Q-mass much smaller than 1/q carry target mass that truncated importance sampling misses at any fixed q. That is why the center floor survives when Z is granted.

| | Pointwise (q→∞, fixed instance) | Uniform (finite q, worst case over the class) |
|---|---|---|
| Z known | p=0 suffices | p≳Ψ(q/p)A, plus the dynamics constraint |
| Z unknown | exact ceiling; p≍ε⁻¹log(1/δ) | also ωp+η₀q≳A (scale channel) |

This table is the organizing picture for the paper. Disclosing one scalar removes the pointwise target requirement entirely; in the uniform setting it removes only the scale constraint. Part II of fixed_target_power.md covers the intermediate disclosure of a public cap B≥Z: the ceiling survives when Z₀/B≤1−κ, and the boundary case is the equality counterexample.

## 4. A finite-q rate for the approach to the ceiling

Part I gives no rate. In the moment class we get one. Fix k>1, binary h, p≥1 and r∈(0,1]. Consider instances with E_QW^{1+r}≤M (M≥1), λ:=min(ℓ,1−u)≥λ₀ and root gap g:=u−ℓ≥g₀. Split q into n=q/2 pooled proposal rows and q/(2L) independent cold paths, and take γ_q=1/q.

**Theorem B4.** For q≥q₀(L,g₀,k),

$$
\Pi_p-\mathrm{Power}_q\ \le\ C\,\delta\,p^2\lambda^{-p-1}\Big[\tfrac1r\big(M/q^{r}\big)^{1/(1+r)}+pMq^{-r}+\tfrac{1}{(1+1/k)g}\sqrt{L\log q/q}\Big]+\tfrac{2}{q}\lambda^{-p}.
$$

For r=1 this is O(q^{−1/2}). The q≳M^{1/r} scale matches Pro's proposal-only endpoint Ω(M^{1/r}ε^{−(1+r)/r}). No matching lower bound on the rate is claimed.

*Proof.*

1. **Reduction.** Let X* be the limiting e-value and X̂=min(B₊/l^p,B₋/(1−v)^p). Since min(1,δ'x) is 1-Lipschitz in δ'x and X*≤λ^{−p}, the deficit is at most γ_qλ^{−p}+γ_q+δE[(X*−X̂)₊1_𝒢], where 𝒢 is path coverage.
2. **Each side.** On 𝒢, ℓ≤l≤ℓ+Δ, so (μ₀/ℓ)^p−B₊/l^p≤ℓ^{−p}(μ₀^p−B₊)₊+pΔℓ^{−p−1}.
3. **Root error.** With path-interval width w=√(L log(4q)/q), the upper polynomial U satisfies F_k≤U≤F_k+3w on [0,1]. Since |F_k'|≥(1+1/k)g/2 on [ℓ,ℓ+g/4], we get Δ≤6w/((1+1/k)g) once 3w≤(1+1/k)g²/8.
4. **Symmetric polynomials.** With s=ΣW_i, s₁=ΣW_ih_i, ρ=s₁/s and m=max W_i over the pooled rows, p!e_p∈[s^p−C(p,2)m s^{p−1}, s^p]. Hence B₊≥ρ^p−C(p,2)m/s, and (μ₀^p−B₊)₊≤p·min(1,|ρ−μ₀|)+C(p,2)min(1,m/s). The same holds for B₋.
5. **Moments.** Let E₁={Σ_QW_i≥n/2}. The von Bahr–Esseen inequality gives P(E₁ᶜ)≤2^{3+2r}Mn^{−r}.
   - On E₁: E|Σ_QW_i(h_i−μ₀)|≤(2nM)^{1/(1+r)}, a mean-zero sum. The target rows contribute E min(1,2Σ_PW_j/n)≤p2^rMn^{−r}, because E_PW^r=E_QW^{1+r}.
   - P(max_QW>xn)≤Mx^{−(1+r)}n^{−r}, so E min(1,2max_QW/n)≤(1+1/r)(2^{1+r}Mn^{−r})^{1/(1+r)}.
6. Collecting terms proves the bound.

## 5. Open items in this topic

- A matching lower bound for the rate in §4.
- Real-valued h for the ESS gate. The gate then depends on ν=Ph², so it is not an interval audit in μ alone.
- The equality-boundary classification. Part II gives a counterexample only.
- Population audits over a prompt distribution: certifying a quantile of G_k(x) over x from per-prompt banks. This is relevant to the LLM experiments; there is no theory yet.
