# Tail-profile frontier for the target-center channel

Status: derivation by Claude, 2026-11-22. It extends [Pro's moment frontier](pro_moment_frontier_original.md) from the power-moment class to general tail profiles and from the half-variance gate to a general ESS gate. Only the steps that differ from Pro's proof are written out; every other step is used verbatim. Needs an independent line-by-line audit before entering the manuscript. Numerical checks: X1 and X3 in [the experiment plan](../../experiments/EXPERIMENT_PLAN_zh.md).

## Setting

Oracle, chain and window as in Pro's theorem: equivalent P, Q, W=dP/dQ, raw scores S=ZW with unknown Z, observable h∈[0,1], cold window b,t with L=b+t, ordinary average A. For a fixed k>1 put

$$
G_k=R_Q(b,t)-\operatorname{Var}_P(h)/k ,
$$

so G_k<0 means the cold window is worth more than k independent target draws (cold-start ESS above k). Pro's gate is k=2. An auditor must issue with probability at most δ whenever G_k≥0 and at least 1−δ whenever G_k≤−ε. Require t≥2k.

**Tail profile.** Ψ:[1,∞)→(0,1] nonincreasing with Ψ(D)=1 for D≤D₀:=128kL, and

- (Ψ1) DΨ(D) is nondecreasing;
- (Ψ2) ∫₁^D Ψ(x)dx ≤ C_Ψ·DΨ(D) for all D≥1;
- (Ψ3) Ψ(D/2) ≤ C_d·Ψ(D) for all D≥2.

The class C(Ψ) contains all equivalent pairs with overlap η=E_Q min(1,W)≥1/8 and target tail τ_P(D):=P(W>D)≤Ψ(D) for every D≥1.

**Examples.**

- Moment class. E_Q W^{1+r}≤M gives τ_P(D)=E_Q[W1{W>D}]≤M D^{−r}. Take Ψ=min(1,MD^{−r}) with M≥(D₀)^r. Then (Ψ1) holds for r≤1, (Ψ2) with C_Ψ=(2−r)/(1−r), (Ψ3) with C_d=2^r. Pro's pointwise bound min(W,D)²≤D^{1−r}W^{1+r} gives an r-uniform constant in (U2) below and should be used for this class.
- Regularly varying Ψ(D)=D^{−r}ℓ(D), 0<r<1, ℓ slowly varying. (Ψ1)–(Ψ3) hold after modification on a compact set (Karamata's theorem and Potter bounds).
- Broken power law: exponent r₁ below a break D_b and r₂ above it, 0<r₁,r₂<1.

## Theorem A (target-center frontier)

Let A=ε⁻²log(1/δ), K=q/p, δ≤1/10. Assume the band K≥D₀, Ψ(K)≤1/2 and ε≤c₀Ψ(K), where c₀>0 depends only on k. There are constants 0<c<C depending only on (k, C_Ψ, C_d) such that:

1. **Upper.** If p≥CΨ(K)A, Pro's scale-free clipped certificate (unchanged, run with the k-gate polynomial) is honest for every equivalent pair with finite positive scores and has power at least 1−δ on all of C(Ψ). It never reads Ψ, r, M, η or Z.
2. **Lower.** Every auditor that is honest and powerful on C(Ψ) uses p≥cΨ(K)A.

By (Ψ1), p↦p/Ψ(q/p)=q/(KΨ(K)) is increasing, so the feasible set is an up-set in each coordinate and its boundary is the unique solution of

$$
p=A\,\Psi(q/p)\qquad\text{(up to constant factors).}
$$

**Reading.** The level K=q/p is where the two sources see a state equally often. A state with normalized weight w and target mass π appears about pπ times among p target draws and about qπ/w times among q proposal draws, and these counts agree at w=q/p. Proposal scoring resolves the weight landscape below q/p, so the target stream must pay for the target mass Ψ(q/p) above it. The moment class gives Pro's p^{1−r}q^r≍MA.

## Proof of the upper bound: the three envelope inequalities change

Validity is distribution-free and does not change. Pro's power analysis uses three envelope facts at the random clip level D=H/Z. Replace them as follows.

- (U1) u₁+u₀=E_Q min(W,D)/D≥η/D. This is unchanged; it is also at least (1−τ_P(D))/D.
- (U2) E_Q v²=E_Q min(W,D)²/D²≤(1+2C_Ψ)Ψ(D)/D. To see this, write min(W,D)²=∫₀^D 2x·1{W>x}dx. For x≥1, Q(W>x)≤x⁻¹P(W>x)≤x⁻¹Ψ(x), because P(W>x)=E_Q[W1{W>x}]≥xQ(W>x); for x<1, bound Q(W>x)≤1. Hence E_Q min(W,D)²≤1+2∫₁^DΨ≤1+2C_Ψ DΨ(D)≤(1+2C_Ψ)DΨ(D), since DΨ(D)≥Ψ(1)=1 by (Ψ1).
- (U3) E_P(1−g)²≤P(W>D)≤Ψ(D). Same form as before.

**Pilot.** Pro's pilot gives Z/16≤A₀≤64Z using only η≥1/8 and E_QW=1. So D∈[K_c/16, 64K_c] with K_c≍K, that is D∈[K/c₂, c₁K] for absolute c₁,c₂≥1. Then:

- (Ψ1) and monotonicity give DΨ(D)≤c₁KΨ(c₁K)≤c₁KΨ(K);
- (Ψ3) gives Ψ(D)≤Ψ(K/c₂)≤C_d^{⌈log₂c₂⌉}Ψ(K).

**Widths.** The head ratio error is at most 4e_u/(u₁+u₀), where e_u=O(√(E_Qv²·ℓ/q)+ℓ/q) and ℓ=log(C₀/δ). By (U1)–(U2) this is

$$
\frac{4D}{\eta}\,O\!\Big(\sqrt{\tfrac{\Psi(D)\ell}{Dq}}+\tfrac{\ell}{q}\Big)=O\!\Big(\sqrt{\tfrac{D\Psi(D)\ell}{q}}+\tfrac{D\ell}{q}\Big)=O\!\Big(\sqrt{\tfrac{\Psi(K)\ell}{p}}+\tfrac{\ell}{p}\Big),
$$

using DΨ(D)≤c₁KΨ(K), D≤c₁K and K/q=1/p. By (U3) the residual interval width is O(√(Ψ(K)ℓ/p)+ℓ/p).

**Lower-order terms.**

- Since p≥CΨ(K)A and ε≤c₀Ψ(K), we get ℓ/p≤ε²/(CΨ(K))≤c₀ε/C.
- Path bank: (Ψ1) and K≥D₀ give KΨ(K)≥D₀Ψ(D₀)=D₀≥L. Hence q=pK≥CAKΨ(K)≥CLA, and the Hoeffding slack √(Lℓ/q) is at most ε/√C.
- The denominator condition 2e_u≤(u₁+u₀)/2 holds for C large.

**Gate.** With real h, G_k=d−2μa+(1+1/k)μ²−ν/k, with |∂_μG_k|≤6, |∂_νG_k|=1/k and |∂_aG_k|≤2. The tested polynomial stays convex in m because its leading coefficient 1+1/k is positive, so the endpoint rule still upper-bounds G_k. As in Pro, the certificate exceeds G_k by less than ε on the good event.

## Proof of the lower bound: changes to Pro's family

Use Pro's three-level family with the following parameters.

- Bulk level J=D₀=128kL.
- Hidden level K'=K=q/p.
- Hidden target mass m∈[Ψ(K)/16, Ψ(K)/8]; the band condition Ψ(K)≤1/2 gives m≤1/16.
- Everything else as in Pro: B₀=3/4, q_B=B₀/J, q_H=m/K, q_L=1−q_B−q_H, l=(1−B₀−m)/q_L.

**Class membership** (this replaces Pro's moment check).

- For D<J: τ_P(D)≤1=Ψ(D), because D<D₀.
- For J≤D<K: τ_P(D)=m≤Ψ(K)/8≤Ψ(D).
- For D≥K: τ_P(D)=0.
- Overlap is at least 1−B₀−m≥3/16. Tilting labels never changes W.

**General-k calibration.** In the all-low coupling the window average is a mean of t iid Bernoulli(1/2) labels, so G_k is within the coupling error e_c≤L(q_B+q_H)≤3/(512k)+1/(2048k)=13/(2048k) of

$$
\varphi^{(k)}_t(\mu)=(\mu-\tfrac12)^2+\tfrac1{4t}-\tfrac{\mu(1-\mu)}{k}.
$$

- At β=1/2, θ=1/2: φ=1/(4t)−1/(4k)≤−1/(8k), which beats e_c.
- At β=1: μ≥.872, so φ≥.1384−.1117/k+1/(4t)>.026 for every k>1, which also beats e_c.
- Continuity gives a root β_*. Writing μ(1−μ)=1/4−(μ−1/2)², the root satisfies (1+1/k)(μ_*−1/2)²≥1/(8k)−e_c, so μ_*−1/2≥c/√k.

Pro's bracket becomes −2a₀+(1+1/k)(μ₁+μ₀)−1/k=(1+1/k)(2μ_*−1)−(2a₀−1)+O(ε). This is at least c'/√k because |2a₀−1|≤2e_c. Then:

- choose the tilt s∈[C_kε/m, (C_k+1)ε/m] with C_k≍√k;
- the band condition s≤.072 becomes ε≤c₀Ψ(K) with c₀≍1/√k;
- the path-moment perturbation is still at most 6Lq_H s=(6L/K)ms≤6ms/(128k), small next to the signed center term c'ms/√k.

**Information.** Unchanged: KL≤C(pm+qm/K)s²=C·C_k²ε²(p+q/K)/m. The randomized public names, adaptive named probes and chain rule are verbatim. Hence p+q/K≥c m A/C_k², and K=q/p gives p≥cΨ(K)A/(32C_k²).

Scanning K' over the band gives the necessary condition p+q/K'≥cΨ(K')A for every K'. The choice K'=q/p binds up to constants, and the upper bound shows nothing stronger is needed.

## Corollaries

1. **Moment class:** p^{1−r}q^r≍MA (Pro).
2. **Regularly varying tails:** p^{1−r}q^r≍ℓ(q/p)A. Slowly varying factors in the tail appear inside the frontier.
3. **Broken power law:** on log–log axes the frontier has slope −r₁/(1−r₁) for q/p<D_b and slope −r₂/(1−r₂) beyond it. The frontier bends exactly where the tail profile bends. This is the most visual test (experiment X1c).
4. **Adaptivity:** one estimator, with no tail input, is rate-optimal over every admissible profile at once. By Pro's empty-tail obstruction, validity cannot rely on a profile inferred from the data.

## Not covered

- r→1 or light tails, where (Ψ2) fails. There E_QW²<∞, and proposal-only importance sampling already gives q≍A·E_QW².
- Small overlap combined with a tail profile. The pilot step needs η≥1/8. Moreover P(W≤D)≤Dη, so a class with overlap η₀ forces Ψ(D)≥1−Dη₀; that is, Ψ≈1 up to about 1/(2η₀). See [three_channels.md](three_channels.md).
- Instance-dependent lower bounds and the k-uniform dependence of the constants. The constants above degrade polynomially in k.
