# Cold-start risk bound

Status: Claude derivation, 2026-11-22. It needs a human audit. Exact numerical checks are in `experiments/finite_state/risk_bound.py` and `risk_bound_refined.py`, with logs in `results/local_runs/theory_checks-20261122/`. This note proves step 1 of the open-regime programme in [small_overlap_joint_frontier.md](small_overlap_joint_frontier.md) §6. Under the standing assumptions it also proves that note's heuristic risk bound, in a sharper form with no variance factor and no tail assumption.

## Setting and notation

- *Pair.* P and Q are mutually absolutely continuous, with W=dP/dQ, so E_QW=1 and E_PW⁻¹=1. Write τ(v)=P(W>v).
- *P-median.* Let d=inf{v: τ(v)≤1/2}, the P-median of W.
- *Chain and window.* IMH has kernel K and starts cold at X₀~Q. The window is i∈[b,L), with L=b+t and t≥1.
- *Risk.* For h∈[0,1], put μ=Ph and f=h−μ, so |f|≤1 and the oscillation of f is at most 1. Let Ā=(1/t)Σ_{i=b}^{L−1}f(X_i). The cold risk is R=R_Q(b,t)=EĀ².
- *Window quantities.*

$$
\Gamma_t=\frac{E_P\min(W,t)}{t}=\frac1t\int_0^t\tau(v)\,dv,\qquad
\Xi_{b,t}=\underbrace{E_P\min\!\Big(\frac Wt,\,1,\,\frac LW\Big)}_{\Xi_1}+\tau(L)^2 .
$$

- *Acceptance.* α(v)=E_Q min(1,W/v)=E_Q min(W,v)/v is the acceptance probability from a state of weight v, and λ=1−α. α is nonincreasing, so λ is nondecreasing.

**Elementary facts.**

- (a) d≥1/2. Indeed P(W≤v)=E_Q[W;W≤v]≤v, so τ(v)>1/2 for v<1/2.
- (b) d≥1/(4η), where η=E_Q min(1,W). If d≥1, the stronger d≥1/(2η) follows from P(W≤D)≤Dη for D≥1. If d<1, η≥P(W≤d)≥1/2 and (a) gives d≥1/2≥1/(4η).
- (c) P(W≥d)≥1/2 and P(W≤d)≥1/2.
- (d) min(t,d)≤2tΓ_t, because tΓ_t=E_P min(W,t)≥min(t,d)P(W≥d).
- (e) Γ_t≥1/(4t), because E_P min(W,t)≥(1/2)P(W≥1/2)≥1/4.
- (f) 1/α(v)≤2max(v,d). For v≥d, α(v)≥E_Q[W;W≤v]/v=P(W≤v)/v≥1/(2v). For v<d, α(v)≥α(d).
- (g) Ξ_{b,t}≤2Γ_t, since Ξ₁≤Γ_t and τ(L)²≤τ(t)≤Γ_t. Conversely Γ_t≤Ξ₁+τ(L), because min(W/t,1)=1 on {W>L}.

## Results

| Label | Statement | Constant (proved) | Observed sup |
|---|---|---|---|
| Lemma H | Height representation K^n(x,·)=E[L_{V_n}(x,·)] | exact | — |
| Lemma D | For w(x)≥d: P(X_i∈dx)≤4·min(1,(i+1)/w(x))·P(dx) | 4 | 1.93 |
| **Lemma R** | R_Q(b,t)≤C₀Γ_t for every h∈[0,1], every (b,t), every equivalent pair | C₀=44 | 1.45 |
| Lemma R′ | R_Q(b,t)≤C₁Ξ_{b,t}, so weights beyond L are discounted by L/W | C₁=1520 | 1.36 |
| Cor. R1 | G_k≥−ε implies Var_P(h)≤k(44Γ_t+ε) | — | — |
| Cor. R2 | Γ-certificate: valid everywhere, uses no chain paths; p≍kℓ/Var, q≍d_{1/4}ℓ | — | — |
| Cor. R3 | Time-averaged cold law: E_νf²≤2Var_P(h)+16Γ_t | — | — |
| Cor. R4 | Light-window upper bound for the dynamics channel, ∝A·max{Γ_t,ε} | sketch | — |

Here ℓ=log(1/δ) and A=ε⁻²ℓ.

## Lemma H (height representation)

For v>0, define the kernel L_v(x,·) as follows: refresh from P(·|W≤v) if w(x)≤v, and stay at x otherwise. Let H have density F(h)/h², with F(h)=P(W≤h). This is a probability density because ∫F(h)h⁻²dh=E_P∫_W^∞h⁻²dh=E_PW⁻¹=1. Then:

- (i) K(x,·)=E[L_H(x,·)].
- (ii) L_uL_v=L_{u∨v}.
- (iii) Hence K^n(x,·)=E[L_{V_n}(x,·)], where V_n is the maximum of n iid copies of H, and P(V_n<v)=λ(v)^n. Set V₀=0.

*Proof.*

(i) For y≠x,

$$
E[L_H(x,dy)]=P(dy)\,E\big[\mathbf 1\{H\ge w(x)\vee w(y)\}/F(H)\big]=P(dy)\int_{w(x)\vee w(y)}^\infty h^{-2}dh=\frac{P(dy)}{w(x)\vee w(y)}.
$$

The right side equals Q(dy)min(1,w(y)/w(x)), which is the IMH off-diagonal kernel. Also

$$
P(H<v)=E_P[(W^{-1}-v^{-1})_+]=E_Q[(1-W/v);W<v]=1-\alpha(v)=\lambda(v).
$$

(ii) Case v≥u. If w(x)≤u, the result is P(·|W≤u) refreshed again from P(·|W≤v), which is P(·|W≤v). If u<w(x)≤v, x stays under L_u and is refreshed under L_v. If w(x)>v, x stays under both.

Case v<u. If w(x)≤u, the result is P(W≤v|W≤u)·P(·|W≤v)+P(·∩{W>v}|W≤u)=P(·|W≤u). Otherwise x stays under both.

(iii) Follows from (i) and (ii). ∎

Only the kernel identity is used below, together with the Markov property E[f(X_j)|X_i]=K^{j−i}f(X_i). This is the exact n-step structure of IMH; Smith & Tierney (1996) give explicit n-step IMH kernels, and the connection should be checked when writing related work.

From (iii), for any x and n≥0:

$$
K^nf(x)=\lambda(w(x))^n f(x)+g_n(w(x)),\qquad g_n(w)=E[m(V_n);V_n\ge w],\qquad m(v)=E_P[f\mid W\le v].
$$

Define

$$
\varphi(v)=\begin{cases}1,&v<d,\\ \min(1,2\tau(v)),&v\ge d.\end{cases}
$$

Then |m(v)|≤φ(v). For v≥d, E_P[f;W≤v]=−E_P[f;W>v] and P(W≤v)≥1/2. The function φ is nonincreasing and right-continuous, and φ(∞)=0.

**Two summation bounds.** Put s(v)=min(t,2max(v,d)). In (S) and its uses below, an integral from d to an upper endpoint below d denotes the integral over the empty interval, hence is zero. By (f), for n ranging over at most t consecutive integers,

$$
\sum_n\lambda(v)^n\le\min(t,1/\alpha(v))\le s(v).
$$

By the layer-cake formula (the set {φ>u} is an interval [0,v_u), so P(φ(V_n)>u)=λ(v_u)^n), for every w∈(0,∞]:

$$
\sum_n E[\varphi(V_n);V_n<w]\ \le\ s(0)\varphi(0)+\int_0^{w}s'(v)\varphi(v)\,dv\ =\ \min(t,2d)+2\int_d^{w\wedge t/2}\varphi(v)\,dv=:\Phi(w).\tag{S}
$$

## Lemma D (cold-law domination)

For every i≥0 and every x with w(x)≥d,

$$
P(X_i\in dx)\le\Big(\frac1{w(x)}+2\min\big(1,\tfrac{i}{w(x)}\big)\Big)P(dx)\le 4\min\Big(1,\frac{i+1}{w(x)}\Big)P(dx).
$$

*Proof.* By Lemma H(iii),

$$
QK^i(dx)=\int Q(dx_0)\,\big(P(V_i<w(x_0))\,\delta_{x_0}(dx)+E[\mathbf 1\{w(x_0)\le V_i\}P(dx\mid W\le V_i)]\big).
$$

This is at most

$$
Q(dx)+P(dx)\,E\big[\mathbf 1\{V_i\ge w(x)\}/F(V_i)\big].
$$

Now:

- Q(dx)=P(dx)/w(x).
- On {V_i≥w(x)≥d}, F(V_i)≥1/2.
- P(V_i≥w)=1−λ(w)^i≤min(1,iα(w)), and α(w)≤E_QW/w=1/w.

For the last step use w(x)≥d≥1/2. If w≤i+1, the bound is at most 2+2=4. If w>i+1, it is at most (2i+1)/w≤2(i+1)/w. ∎

**Consequences.** For v≥d, P(w(X_i)>v)≤4τ(v). For any F≥0,

$$
E[F(w(X_i));w(X_i)\ge d]\le4E_P\big[F(W)\min(1,(i+1)/W);W\ge d\big].
$$

The cold chain under-samples weights beyond its age: it can reach weight w only through about i·Q(dw)=iP(dw)/w proposals.

## Lemma R: R_Q(b,t)≤44Γ_t

*Proof.* Expanding the square,

$$
t^2R=\sum_iEf(X_i)^2+2\sum_{i<j}Ef(X_i)f(X_j).
$$

With n=j−i, E f(X_i)f(X_j)=E[f(X_i)K^nf(X_i)]. Since |f|≤1 and |g_n(w)|≤Eφ(V_n)=:e_n, this is at most Eλ(w(X_i))^n+e_n. For fixed i, n ranges over at most t−1 values. So

$$
R\le\underbrace{\frac1t}_{\text{diag}}+\underbrace{\frac2{t^2}\sum_i\sum_n e_n}_{B}+\underbrace{\frac2{t^2}\sum_iE\,s(w(X_i))}_{C}.
$$

- *Diagonal.* 1/t≤4Γ_t by (e).
- *B.* By (S) with w=∞,

$$
\sum_ne_n\le\min(t,2d)+4\int_d^{t/2}\tau\le4t\Gamma_t+4t\Gamma_t,
$$

using (d) and ∫_0^tτ=tΓ_t. Hence B≤(2/t)·8tΓ_t=16Γ_t.
- *C.* Split on w(X_i)<d, where s≤min(t,2d)≤4tΓ_t, and w(X_i)≥d, which Lemma D controls. This gives

$$
E\,s(w(X_i))\le4t\Gamma_t+4E_P\min(t,2W)\le12t\Gamma_t.
$$

Hence C≤24Γ_t.

Total: R≤(4+16+24)Γ_t=44Γ_t. ∎

**Reading.** In Γ_t=(1/t)∫_0^tτ, each weight contributes its window-truncated holding time min(W,t)/t, as in the heuristic of the frontier note. The heuristic's startup term 1/(ηT) is already inside Γ_t, since d≥1/(4η) gives Γ_t≥min(1,1/(4ηt))/2. This corrects the general startup comparison only; the factor 2 in its denominator is valid when d≥1, and the other stated constants are unchanged. The log(1/η) factor one might expect from the mixing time does not appear: (f) bounds every holding time by 2max(w,d) directly. No spectral gap is used, and W may be unbounded. Under (Ψ2), Γ_t≤1/t+C_ΨΨ(t), so R≲Ψ(T)+1/T. This is step 1 of the frontier note §6, without the variance factor.

## Lemma R′: beyond-window discount

$$
R_Q(b,t)\le144\,\Xi_1+96\,\tau(L)^2+72\min(1,d/t)+\frac{17}t\ \le\ 1520\,\Xi_{b,t}.
$$

*Proof.* Write R=VarĀ+(EĀ)².

*Variance.* Since f(X_i) has oscillation at most 1,

$$
t^2\mathrm{Var}\bar A\le t/4+2\sum_{i<j}|\mathrm{Cov}(f(X_i),f(X_j))|.
$$

Also Cov(f(X_i),f(X_j))=Cov(f(X_i),K^nf(X_i)). Use Cov(U,V)=E[(U−EU)(V−c)] for any constant c, with |U−EU|≤1:

- |Cov(U,λ^nU)|≤Eλ(w(X_i))^n;
- |Cov(U,g_n(w(X_i)))|≤E|g_n(w(X_i))−g_n(0)|.

For the second, note |g_n(w)−g_n(0)|=|E[m(V_n);V_n<w]|≤E[φ(V_n);V_n<w].

Summing over n, the two summation bounds give

$$
\sum_{n}|\mathrm{Cov}|\le E\,s(w(X_i))+E\,\Phi(w(X_i)).
$$

- *The s term.* Use Lemma D with the factor min(1,(i+1)/W)≤min(1,L/W), together with min(t,2W)min(1,L/W)≤2t·min(W/t,1,L/W). This gives

$$
E\,s(w(X_i))\le\min(t,2d)+8t\,\Xi_1.
$$

- *The Φ term.* Use φ≤2τ and P(w(X_i)>v)≤4τ(v):

$$
E\,\Phi(w(X_i))=\min(t,2d)+2\int_d^{t/2}\varphi(v)P(w(X_i)>v)\,dv\le\min(t,2d)+16\int_0^t\tau^2 .
$$

  This is the step where the target tail enters *quadratically*.
- *The τ² integral.* For v≤L, τ(v)=P(v<W≤L)+τ(L). So τ(v)²≤P(v<W≤L)+τ(v)τ(L), and

$$
\int_0^t\tau^2\le tE_P[\min(W/t,1);W\le L]+\tau(L)\,t\Gamma_t\le t(2\Xi_1+\tau(L)^2),
$$

  using (g).

Altogether,

$$
\mathrm{Var}\bar A\le\frac1{4t}+\frac{4\min(t,2d)}t+80\,\Xi_1+32\,\tau(L)^2 .
$$

*Bias.* By Lemma H, Ef(X_i)=E_Q[λ(W₀)^if(X₀)]+E g_i(W₀). Hence

$$
|E\bar A|\le\frac1t E_Q\,s(W_0)+\frac1t\sum_ie_i\le\frac{\min(t,2d)+2}t+\frac{\min(t,2d)}t+4\Gamma_t\le4\min(1,\tfrac dt)+\frac2t+4\Xi_1+4\tau(L).
$$

The second inequality uses (S) and E_Q min(t,2W)≤2. The last uses (g). Square it with (Σ_1^4a_r)²≤4Σa_r², and use ℓ_d²≤ℓ_d (ℓ_d=min(1,d/t)), 1/t²≤1/t and Ξ₁²≤Ξ₁:

$$
(E\bar A)^2\le64\,\ell_d+16/t+64\,\Xi_1+64\,\tau(L)^2 .
$$

Adding the two parts gives the first inequality.

*Absorbing the light terms.*

- 1/t: since P(W≥1/2)≥1/2, either τ(L)≥1/4, or P(1/2≤W≤L)≥1/4 and then Ξ₁≥1/(8t). So 1/t≤8Ξ₁+16τ(L)².
- ℓ_d: if d≤L, either τ(L)≥1/4, or P(d≤W≤L)≥1/4 and then Ξ₁≥ℓ_d/4. If d>L, then τ(L)>1/2. So ℓ_d≤4Ξ₁+16τ(L)².

Substituting gives R≤568Ξ₁+1520τ(L)². ∎

**Reading.** Γ_t charges the full mass τ(t) beyond the window. Ξ splits that mass in two:

- Weights in (t,L] stay at full charge, because the chain had time to reach them.
- Weights beyond L are charged L/W, the chance the cold chain ever reached them, plus a *squared* bias term τ(L)².

Consequently, Γ_t can be loose by a factor of order 1/τ(L) when a small target mass sits far beyond L. The numerics observe Γ_t/Ξ up to 10.7.

## Corollaries

**R1 (near-threshold instances have small variance).** Write G_k=R−Var_P(h)/k. If G_k≥−ε, then Var_P(h)≤k(R+ε)≤k(44Γ_t+ε), or k(1520Ξ_{b,t}+ε) using R′. Contrapositive: in a class with Γ_t≤γ, every h with Var_P(h)>k(44γ+ε) passes the gate by more than ε. Hard instances of the additive-margin problem are therefore low-variance h. Any lower-bound family in the light-window regime must be variance-scaled, which is step 2 of the frontier note §6.

**R2 (Γ-certificate).** The procedure uses three independent banks, each at error δ/3:

1. *Clip level.* From n₀=Cℓ target draws, let ĉ be the empirical 5/8-quantile of the scores S. By DKW, with probability at least 1−δ/3, Zd≤ĉ≤Zd_{1/4}, where d_{1/4}=inf{v: τ(v)≤1/4}.
2. *Normalizer lower bound.* From q proposal scores, let Z_L be the empirical-Bernstein lower confidence bound for E_Q min(S,ĉ). Since E_Q min(S,ĉ)≤Z for every ĉ, the event Z_L≤Z holds with probability at least 1−δ/3 without any assumption.
3. *Γ upper bound.* The quantity Γ̄=E_P min(S/Z_L,t)/t is decreasing in Z_L, so Γ_t≤Γ̄ whenever Z_L≤Z. From p₁ target draws of Y=min(S/Z_L,t)/t∈[0,1], let U_Γ be the empirical-Bernstein upper confidence bound.
4. *Variance lower bound.* From p₂ target draws of h, let V_L be a lower confidence bound for Var_P(h). For example, use the pair U-statistic (h_i−h_j)²/2 with Bernstein.
5. *Decision.* Issue "ESS(b,t)≥k" iff 44U_Γ<V_L/k.

*Validity, on every equivalent pair, for every h and (b,t).* On the joint event, which has probability at least 1−δ,

$$
R\le44\Gamma_t\le44\bar\Gamma\le44U_\Gamma<V_L/k\le\mathrm{Var}_P(h)/k,
$$

so G_k<0. No overlap, tail, or boundedness assumption is used.

*Power.* Suppose Var_P(h)≥C_*kΓ_t with C_*=704. Then the procedure issues with probability at least 1−δ once

$$
p_1,p_2\ge C\,k\ell/\mathrm{Var}_P(h),\qquad q\ge C\,d_{1/4}\ell .
$$

The argument has three steps:

- ĉ≥Zd gives E_Q min(S,ĉ)≥Z·P(W≤d)≥Z/2. Bernstein, with variance at most ĉZ≤d_{1/4}Z², gives Z_L≥Z/4, hence Γ̄≤4Γ_t.
- The Bernstein width for Y has Var(Y)≤EY≤Γ̄≤4Var/(C_*k). With p₁≳kℓ/Var this width is at most Var/(400k). The constant in C_* is not optimized.
- V_L≥Var/2 once p₂≳ℓ/Var.

*Interpretation.*

- The certificate simulates no chain at all. Its cost is independent of (b,t) except through the issuing condition.
- The proposal cost is d_{1/4}ℓ, about ℓ/η when the profile is doubling.
- It certifies with a *multiplicative* margin, whereas the three-channel lower bounds concern instances within an additive ε of the threshold. The two do not conflict. In the long-window flat class, Γ_t≍1 and Var≤1/4<C_*kΓ_t, so the certificate never issues there.
- A Ξ-certificate would need a two-sided bracket on Z, for example the clipped bridge E5, because min(W/t,1,L/W) is not monotone in W.

**R3 (time-averaged cold law).** Let ν=(1/t)Σ_{i=b}^{L−1}law(X_i). Then

$$
E_\nu f^2\le2\mathrm{Var}_P(h)+16\Gamma_t .
$$

*Proof.* By Lemma H,

$$
Ef(X_i)^2\le E_Q\lambda(W_0)^i+E\,E_P[f^2\mid W\le V_i].
$$

The inner conditional expectation is at most 2Var_P(h) on {V_i≥d} and at most 1 otherwise, and P(V_i<d)=λ(d)^i. Averaging over i,

$$
E_\nu f^2\le2\mathrm{Var}_P(h)+\frac{\min(t,2d)+2}t+\frac{\min(t,2d)}t\le2\mathrm{Var}_P(h)+16\Gamma_t,
$$

by (d) and (e). ∎

*Use: the two-time replicate.* Draw I and J independently and uniformly on the window, and form Y=f̂₁(X_I)f̂₂(X_J), where f̂_r=h−μ̂_r uses two independent target estimates of μ. Then EY=R exactly, and

$$
\mathrm{Var}\,Y\le E_\nu\hat f^2\le2\mathrm{Var}_P(h)+16\Gamma_t+O(\mathrm{Var}_P(h)/p).
$$

This is the replicate that E4 generates with P-based lag jumps. It needs two evaluations of h per path.

**R4 (light-window upper bound, sketch).** Near the threshold, R1 gives Var_P(h)≤k(44Γ_t+ε). By R3, the replicate variance is then at most Ck(Γ_t+ε). With Bernstein, m≍kℓ·max(Γ_t,ε)/ε² replicates estimate R to ±ε. Each replicate costs L proposal draws (a direct cold path) or c_P target draws (E4 lag jumps). The variance term Var/k needs p≳Var·ℓ/(kε)²≲A(Γ_t+ε)/k target draws, which is dominated. So the dynamics channel is achievable at

$$
\frac{p}{c_P}+\frac{q}{L}\ \gtrsim\ k\,A\,\max\{\Gamma_t,\varepsilon\},
$$

together with the center channel. This is the *upper half* of the frontier note's conjecture, with ρ_T replaced by the instance quantity Γ_t, or by Ξ_{b,t} with a two-sided Z bracket. In the flat class Γ_t≍1, and the bound reduces to (D) of the frontier note. The k-dependence and the matching lower bound are open.

## Numerical checks

*Exact computation.* The chain uses finite support with 20–120 states and the exact kernel. The risk is computed by backward recursion for E[(Σf)²|X_b] and pushed forward from Q. The instances are lognormal (σ∈[1,5]), Pareto-plus-floor, dark-plus-heavy, and, in the second script, far-tail atoms carrying target mass θ∈[.01,.6] at weights 10²–10⁷. The functions h tried are threshold indicators 1{W≥v} and 1{W≤v} over a grid of v, random binary h, and random real h.

| Check | Cases | Result |
|---|---|---|
| Lemma R: max_h R/Γ_t (`risk_bound.py`) | 150 instances × t∈{8,64,512,2048}, b∈{0,t/4} | max 1.22, median 0.54 |
| Lemma R, with far-tail instances (`risk_bound_refined.py`) | 200 × t∈{16,128,1024}, b∈{0,t/2,4t} | max 1.45 |
| Lemma R′: max_h R/(Ξ+1/t) | same | max 1.36, median 0.39 |
| Lemma D: max QK^i/(3min(1,(i+1)/w)P) over w≥d, i≤512 | 200 instances | 0.64 (the proved constant is 4) |
| Tightness: R/Γ_t for h=1{W>d}, cases with d<t/4 | 355 | 5%, 50%, 95% quantiles 0.33, 0.58, 0.91 |
| Looseness: Γ_t/Ξ | 600 | max 10.7 (far-tail instances) |

*Notes on the numerics.*

- numpy 2.0 with macOS Accelerate emits spurious floating-point warnings in matmul, even on benign 50×50 products. The scripts silence them and assert that every risk is finite instead.
- The proved constants (44 and 1520) are 30–1000 times larger than the observed sup. The certificate R2 would gain directly from a sharper C₀, so this is worth an hour of constant-chasing before submission.

## Consequences for the open regime

- *Step 1 of the frontier note §6 is done.* The proved bound is R≤44Γ_t, with no variance factor and no tail assumption; under (Ψ2), Γ_t≤1/t+C_ΨΨ(t).
- *The conjecture has an instance quantity.* Its ρ_T should be read as Γ_t, or Ξ_{b,t}. The upper half follows from R4 together with E1 and E4.
- *Variance-scaled lower family (step 2): done at constant overlap* in [variance_scaled_floor.md](variance_scaled_floor.md). Both W laws, both h marginals, μ and Var are invariant; only the signed label pairing on two sticky levels at weights t/c_i differs. The result is p+q/t≳A·min{Ψ(t),c/k}. The guidance below is what the construction follows. Lemma R′ says that sticky mass at weight w∈[t,L] is charged in full, while mass beyond L is discounted by L/w. The hard family should therefore put its sticky state at a weight between t and L, not at weight ≫L as in the flat dynamics floor. The target mass there is ρ, and h is tuned so that Var≈kρ.
- *Small overlap (2026-11-22)* in the same file, section "Small overlap". Lemma R and Lemma D feed Proposition N directly: Em²≤44Γ_t, EΣo_g²≤24Γ_t (term C), and over-occupation below d is at most e_b≤min(1,2d/t)e^{−b/(2d)} (Lemma H with fact (f)). Hence, after burn-in b≳2d·log(Γ_tρ_B/ε²), any label difference that moves the gate by 2ε costs KL_P≥ε²/(330Γ_t). The b=0 counterpart is Theorem S, which uses the over-occupied startup level, i.e. the pre-mark segment of the Pro first-mark split.
- *A practical message for the paper.* Certifying with a constant-factor margin is cheap and needs no chain simulation (R2). The three-channel costs appear only near the threshold.

## Open items

1. *Two-sided characterization.* Is sup_hR_Q(b,t)≍Ξ_{b,t} up to absolute constants? The numerics support it: h=1{W>d} already attains 0.33Γ_t. A proof would need a lower bound on the excursion contribution of weights in (d,L].
2. *Variance-scaled form.* A bound like R≲(1/t)E_P[f²min(W,t)]+(light terms) would be sharper for low-variance h. The light terms of order min(1,d/t) are the obstacle. They come from the bulk below d, where (f) is crude.
3. *Constants.* Sharpen C₀=44, for example by keeping min(1,(i+1)/W) in term C and using |U−EU|≤1 in the diagonal.
4. *Literature.* Check Smith & Tierney (1996) on exact IMH n-step kernels and Wang (2022, Bernoulli) on exact IMH convergence rates, then position Lemma H and Lemma D against them.

## Pro–Claude synthesis: cold finite-window risk and what the auditor must observe

Let `R_Q(b,T)` be the expected squared error of the ordinary average of `h(X_b),…,h(X_{b+T-1})`, with `X_0~Q`, centered at `Ph`. Write `f=h-Ph`, `v=Var_P(h)>0`, and `W=dP/dQ`. This is not a terminal-output guarantee. The reduced-kernel identity below assumes positive finite likelihood ratios; the risk bounds must in any case specify how zero-weight proposals are treated.

**Two bounds, two jobs.** Claude's `Γ_T=E_P[min(W/T,1)]` envelope controls **absolute** cold risk for bounded h and can rule out large-variance near-gate instances in a light-tail regime. It is not a relative-risk envelope of the form `C Γ_T v`. Pro's finite-window bound uses the dependence between W and the actual observable, plus a separate low-weight cold-start term. Denote `a_s=E_Q[min(1,W/s)]`, `r_s=1-a_s`, `D_s=E_P[f²1{W>s}]+E_P[f1{W>s}]²/P(W≤s)` (set `D_s=v` when the denominator is zero), and `g_T(r)=(T+2∑_{j=1}^{T-1}(T-j)r^j)/T²`. Pro shows the stationary bound `S_T=g_T(r_d)v+(1-g_T(r_d))D_d` and the cold bound `S_T/m_c+H+2√(χ_c² S_T H)`, where `m_c=c a_c`, `χ_c²=E_P[(min(1,c/W)/m_c-1)²]`, and zero-burn `H=E_Q[(1-W/c)_+ f²](1+2r_c)/(a_c³ T²)`. These are population quantities; adaptively selecting thresholds from estimated quantities requires simultaneous error control. Pro's proof uses an independent regeneration mark at time zero and real proposal uniforms thereafter. No-mark candidates are all below c; therefore the first marked candidate is accepted. The post-mark density relative to P is `min(1,c/W)/m_c`. Conditioning on the mark time makes prefix and suffix independent, and the suffix has mean zero when `c≥ess sup W`. Burn-in suppresses the prefix bound, not the observable-dependent stationary part; the actual risk need not decrease.

**Exact stationary marked spectrum (independent refinement).** Let `M_s` be conditional averaging across states with `W≤s`, leaving all other states unchanged. These orthogonal projections are nested and `||M_s f||²=D_s`. The IMH transition is their mixture with positive-level measure `P(W≤s) ds/s²` (add identity mass at zero for `Q(W=0)>0`). Indeed the nontrivial update density is `∫_{max(W(x),W(y))}^∞ P(dy) ds/s²=P(dy)/max(W(x),W(y))`. Thus for j≥1 the covariance is `∫ D_s d(r_s^j)`. Define the positive finite measure `ν` by `ν((s,∞))=D_s`; since `r_s` is continuous at positive s, integrate by parts (or apply Tonelli) to obtain
```
R_P(0,T) = ∫ g_T(r_s) ν(ds).
```
On finite support, each distinct weight s contributes `(D_{s-}-D_s)g_T(r_s)`; group exact ties before computing jumps. A sorted threshold grid also gives deterministic upper and lower Riemann brackets: multiply each bin mass from the D endpoints by the right/left value of g respectively. Their width is at most `v max_i[g_T(r_{s_i})-g_T(r_{s_{i-1}})]`, including the tail bin (limit g=1). This isolates what a grid must resolve: marked projection-energy mass, not the ordinary histogram of W. It is not a guarantee on the number of samples needed to estimate the bracket.

**Exact cold split (independent cross-check).** Conditional on a mark *not* occurring at any of the first k candidate draws, the prefix itself is IMH with proposal `Q_c^- ∝ (1-W/c)_+ Q` and target `P(·|W<c)`, with initial law `Q_c^-`. For W(x),W(y)<c its conditional acceptance is `[min(1,W(y)/W(x))-W(y)/c]/[1-W(y)/c]`, equal to the IMH ratio clipped at 1 for weight proportional to `W/(1-W/c)`. Conditional on the first mark at k, the suffix is independent and has the original kernel and the marked start above. For each k split the measurement window `[b,b+T)` into prefix and suffix and take `E A²+E B²+2 E A E B` for the two centered sums. Average with weights `a_c r_c^k`, plus the no-mark mass `r_c^{b+T}`, then divide by T². Handle `r_c=0` by using the suffix only. This is an exact benchmark when the laws are known, not a new query-efficient audit. Two independent computations compared the split to direct path-moment recursion. The rational CPU replay checked both this identity and the stationary identity on 96 seeded 2–5-state chains with varying burn-in, horizon and threshold. A separate 300-case floating-point replay of the split agreed to 1.51e-16. In the rational replay, the reported four-state example at T=256 gives relative risks 0.116536386247 and 0.015234326077 under the same kernel. At the common gate `v/12` the first is unsafe, the second is safe (Pro's population bound 0.02537 for the second).

**A matching two-point information calculation, not a minimax frontier.** For the same T=256 example, set `x=1/16`, `P=(x,x,2x,1-4x)`, `Q=(x/T,x,x+x/T,1-2x-2x/T)`. The two binary observables differ on atoms 1–3 only. Both sides have the same P and Q marginal observable distributions, the same v and ordinary weight laws, and the same low-weight marked error at c=1. In a fixed-two-world oracle, the two hypotheses and atom masses are public, but the true labels are only disclosed by scored independent P and Q draws with atom ID. A zero-false certificate may issue exactly when a distinguishing row is seen. With p target and q proposal draws its optimal power is
```
1 - (1 - 4x)^p (1 - 2x - 2x/T)^q.
```
Proof: absent a hit the transcripts have identical conditional laws and probabilities; after a hit the world is known. With false-sign budget δ replace the power by `min(1,1-m+δ)` where m is the no-hit probability, by randomizing only on no-hit. At 95% power with zero false signatures, pure P needs 11 draws; pure Q needs 23. Compare costs per draw using their negative log miss rates, not those counts alone; a named query to a distinguishing atom trivializes the problem. This establishes a scoped information cost for the very same matched-marginal example/gate, in an explicitly stated oracle. It does not reproduce Pro's distinct half-variance comparison, imply a global lower bound, or claim practical gains. A general lower family must keep the same actual overlap and tail class, the same relative gate and both side's advertised statistics, and hide the signed observable/weight dependence under precisely the upper bound's oracle (including whether addresses are allowed). Claude's variance-scaled construction remains open on those terms.

**Audit limits.** An absolute Γ envelope or data-known marked bound is not itself a confidence certificate. A finite-sample version must account for acquiring centers, variance and marked moments, use a single oracle and consistent prices, and report abstention. In Γ-based acquisition arguments, a pilot can be excluded from the *validity* union bound only if downstream intervals are conditional-uniform for any pilot output; if power needs correct pilot calibration it must be budgeted too. Guard nonpositive normalization estimates. Do not substitute the sharper Ξ envelope into a second-moment/query-rate bound proved only for Γ, or treat an approximate high-only lag jump as an exact risk replication. These are gaps before any claim of a matched short-tail frontier.

**Long-window signed lower witness (separate from the open short-tail frontier).** A complementary construction keeps the centered squared tail itself fixed. For `κ∈(0,1/4096]`, use three weight levels `(2κ,400κT,3T)`, with proposal masses `q_M=5/(4T)`, `q_H=[1-2κ-(400κT-2κ)q_M]/(3T-2κ)`, `q_L=1-q_M-q_H`. Target masses are `p_i=W_i q_i`. Set the mark to ±1 within each level, with mean marks `μ_M=x∈[-.8,-.55]`; solve the two linear constraints `q·μ=.9`, `p·μ=0` for the low and high mark means. Equivalently use `h=(1+mark)/2` in [0,1], preserving normalized risk. Both W marginals, scale, overlap, both binary h marginals, all low marked-square moments and all centered-square marked tails are invariant in x. The *signed* pairing is not. The finite-chain normalized risk tends to a five-state integrated CTMC risk (low redraws average to .9, middle/high signs stay explicit). The rational exponential and between-grid derivative enclosures at large-window limit give `R(-.8)<.494106`, `R(-.55)>.507212`, `∂R/∂x>.05956`, uniformly over κ. In the fresh-label/predeclared named-row information model of the dynamics lower bound, mark separation of order ε contributes KL of order `κ ε²` per P row and `ε²/T` per Q or named row: the middle mark moves at order ε, the high mark at order κ ε, and the low mark at order ε/T. This gives the same weighted long-window necessity with the listed marginals disclosed, in that model; it is not a claim for arbitrary named atom-label access to a public fixed two-point family. In this construction, Pro's single-threshold upper cannot certify even the safe side of the half-variance gate: for d<3T the stationary residual marked energy is at least p_H>.87; for d≥3T its geometric factor is at least `(1-1/(3T))^{T-1}>.5`, and `m_c≤1`. This is a strict conservativeness witness for the upper, not a contradiction. The heavy 3T layer means this construction does not settle the regime where the tail is already strongly lightened inside the window.

### Pro source: finite prefix and minimal reproduction

The finite prefix bound (before extending the sum to infinity) is
`H_{b,T}=(U_c r_c^b/T²)∑_{n=0}^{T-1}r_c^n[b(1+2n)+(n+1)(1+3n/2)]`, with `0^0=1`. To obtain the coefficient, group diagonal and unordered pairs in the prefix square by their last time `j=b+n`: the diagonal contributes `b+n+1` and the earlier times `i=b,…,j-1` contribute `i+j+2` each, all with the common factor `U_c r_c^j`. Within Pro's bound, `S_T` may be replaced everywhere by the true stationary risk. If `c≥ess sup W`, the suffix starts exactly from P even after conditioning on the mark time (and any further burn-in). Let `n` be its random length and `A_-` the centered pre-mark window sum. Then
`T²(R_Q(b,T)-R_P(0,T))=E[A_-²]-E[T² R_P(0,T)-n²R_P(0,n)]`, with the zero-length term zero. The two terms have opposing signs, so the cold-chain MSE need not be monotone in burn-in. In the rare-high control at T=64 it rises from 0.28689350 to 0.57093421 after 64 burn-in steps (both relative to v).

Source: Pro's finite-window theorem. The [seven-row evidence table](../../results/pro/pro_cold_core.csv) can be regenerated from the project root with `python3 experiments/finite_state/pro_cold_core.py`. This dependency-free rational replay retains the matched pair at T=64 and T=256 and both failure controls. The displayed decimals are not outward interval endpoints; certificate decisions use exact rational comparisons before rounding. The replay reports population bounds and is not a finite-sample audit.

### Short-tail phase, disclosed overlap, and purchase direction

All claims below concern ordinary cold-start IMH, independent scored rows, known normalized W=dP/dQ, and public μ=Ph, v=Var_P(h)>0, with h∈[0,1]; no named-point queries. The certification gate is R_Q/v≤1/2, with completeness at 1/2−ε and false-sign probability ≤δ. Latest Pro proof (Chinese).

**Exact hard-cap maximum and full-class budget.** Write M=λT≥1, r=1−1/M, A=max(μ,1−μ), B_T=T⁻²∑ᵢⱼr^|i−j| and C_T=T⁻²∑ⱼ(2j+1)r^{j+1}, indices 0,…,T−1. The attained maximum under W≤M is B_T+(A²/v−1)C_T; with b burn-in steps multiply the second term by r^b. The proof jointly splits Q and K by a P refresh of rate 1/M; a three-atom model attains all pairwise bounds. The long-window critical λ_* is the unique root of
$$2\int_0^1[(1-t)+(A^2/v-1)t]e^{-t/\lambda_*}\,dt=1/2.$$
For λ≤λ_* every feasible finite window is safe without queries. For λ>λ_*, fixed μ,v,λ, sufficiently large T and sufficiently small ε, the minimax proposal requirement is Θ(Tε⁻²log(1/δ)), even with any target budget uniformly finite over the class. Independent Q replay gives the upper; the lower hides a sticky intermediate layer with Q mass Θ(1/T) but arbitrarily small P mass. For small v, the separate worst-center theorem matches T/(vε²) up to the confidence factor; that strengthening does not preserve squared marked tails. The hard-class mandatory-Q lower allows overlap to tend to zero, does not cover fixed η₀>0, and does not constrain budgets allowed to diverge as a function of disclosed side marginals.

**Source switch, only within the disclosed directional experiments.** Two four-layer signed-pairing families share both W and h marginals, overlap, v, the full squared-mark tail, and the half-risk baseline. The parameter direction is public. The high-layer and low-layer unknowns have respective resource frontiers n_P+n_Q/T and n_P/T+n_Q of order ε⁻²log(1/δ). At equal row price the preferred source reverses. One dominant row exactly simulates the weak row by retaining common layer mass and converting surplus signs through an affine Bernoulli channel; this also works online for adaptive mixed purchases. This is not a whole-class exchange law, nor does it simulate undisclosed auxiliary state, unnormalized scores, or telemetry.

**Overlap disclosure changes zero-row feasibility.** For μ≠1/2, let η₀ be public, ρ₀=max(0,(η₀−1/M)/r), and choose 1−A<a<A. Then D=(A²−a²)[ρ₀−v/(v+a²)]₊ improves the uniform envelope to B_T+r^b(A²/v−1−D/v)C_T. The no-refresh residual keeps overlap ≥ρ₀ with P by TV contraction; Cantelli forces enough shared mass away from the far endpoint. The original maximum remains attainable at every exact η in (1/M,1/M+r v/(v+A²)], and the new correction is strictly positive above that breakpoint (not a sharp formula there). Exact example: T=1000,M=116,μ=.2,v=.05,η₀=.6,a=.44. Without η₀ the attained maximum is .51727; with η₀ the uniform bound is .42462, so buy no rows.

**Fixed-overlap precision bridge.** For exact small η∈(0,1/4), λ>.391094147… and ε≤c/T, a separate four-layer lower preserves both marginals, η and all squared tails. Two vanishing-P-mass dust layers are visible at startup under Q. Contrasting their signs changes the cross moment with post-escape reads by Θ(1/T) per unit perturbation. It forces n_Q=Ω(log(1/δ)/(T²ε²)) for every uniformly finite n_P. CPU recurrence crosses the same gate through T=8192 (η=.12, λ=.5 or .8). In the opposite regime Tη₀vε≥a universal constant, replace proposals W<τ with a zero-weight dummy. On the same proposal stream the risk changes by at most
$$2[(c+1/T)/\eta_0 + \tau/(c(\eta_0-\tau))],\qquad 0<\tau<c<1.$$
The first cutoff-one threshold refresh controls low-state occupation; a high-weight mismatch occurs with probability ≤τ/w and then has mean duration ≤max(1,w)/(η₀−τ). Estimate retained Q mass with bounded reciprocal weights from P, and acquire exact conditional Q rows by rejection τ/W. Choosing c,τ for perturbation below the gate margin gives a finite uniform pure-P certificate, not a competitive cost bound. These two regimes do not yet determine the optimal two-source frontier or the uniform transition constant.

## Fixed-overlap resource frontier for cold IMH

Let W=dP/dQ>0, h∈[0,1], μ=Ph=.5 and v=Var_P(h)∈(0,1/4] public. Ordinary independent Metropolis–Hastings starts at X₀∼Q; average T reads with zero burn-in. Write f=h−μ and R=E[(Σf(Xᵢ)/T)²]. Purchase independent rows (W,h) from P and Q separately; weights are normalized, with no named queries, additional model access, or rejection telemetry. A δ-certificate signs with probability ≤δ if R>v/2 and ≥1−δ if R≤v(1/2−ε). Budgets are uniform deterministic caps; abstention is permitted. Write L=ε⁻²log(1/δ), 0<ε,δ≤1/4.

**Theorem (up to constants).** Fix 0<η₀≤η₁ for a sufficiently small universal η₁, and λ∈(λ_bal,1), where λ_bal≈.391094 is the root of 2∫₀¹(1−t)e^(−t/λ)=1/2. Promise W≤λT and overlap η=∫min(dP,dQ)≥η₀. For T√v sufficiently large and 0<ε≤c/(T√v), the necessary and sufficient caps, with possibly different constants depending only on η₀,λ, are
```
nP + nQ/T ≳ L/v,      nQ ≳ L/(vT²).
```
Both lower families attain exact η=η₀, so the same result holds when that equality is promised and public. At margins C/(T√v)≤ε≤ε₀ (ε₀ depending only on η₀,λ), the first condition alone is necessary and sufficient. No exact boundary constant is claimed between these regimes. This is a whole-class budget, not an instance-sensitive budget allowed to depend on the observed or publicly supplied weight marginals. Within each lower pair, however, both such marginals are identical. Factor T refers to purchased rows: pure Q uses Θ(TL/v) rows, whereas the hybrid uses O(L/v) P rows plus the smaller Q correction, not a wall-clock or target-generation cost claim.

### Upper: marked threshold projections

Put F(s)=P(W≤s). A threshold S has density F(s)/s². Its projection retains states above S and otherwise refreshes to P conditional on W≤S. Integrating gives transition accepting mass P(dy)/max(W(x),W(y)); projections compose by maximum. A dummy below all positive W becomes Q after one projection. Equivalently a Q row Y with U uniform generates S=W(Y)/U and its own correct conditional mark Y|S. A sorted pair of independent uniform read indices i≤j uses independent maxima over i+1 and j−i slots, skipping the latter on the diagonal. Its expected signed product is R.

An independent P pilot selects c=max(1, empirical η₀/4 quantile). With O(η₀⁻¹log(1/δ)) rows, binomial tails (using left limits at atoms) give F(c)≥η₀/8 and a=Pr(S≥c)=E_P[1/max(W,c)]≥η₀/2. For k slots, generate Binomial(k,1/c) independent Pareto(c) candidates lazily in descending order, buying a fresh P row at each and accepting if W≤candidate. The marked point density is P(dy)1{W(y)≤s}ds/s². First acceptance yields the exact high maximum and mark, or failure certifies no high threshold. Row costs are dominated by a geometric variable of parameter F(c), without traversing k ordinary proposals.

The main pair estimator retains pairs with a high first maximum and, if off diagonal, a high lag maximum. Second moment is O_η₀(v), since the first mark has conditional second centered moment at most v/F(c). For exact correction split the remainder into low-first/high-lag, high-first/low-lag, both low off-diagonal, and low diagonal. Sample each required low segment length k≥1 from π(k)=(1−r₀)r₀^(k−1), r₀=√(1−η₀/2); sample any remaining index uniformly, rejecting illegal pairs. Independently test all k Bernoulli slots for the low event with P coins of high probability a, generated by 1/max(W,c). Conditional on a pass, obtain k iid low-threshold marks by filtering fresh Q rows; return their maximum. Weight each signed product by actual sorted-pair probability (2−1_{i=j})/T² divided by its sampled index probability. Indicator probability (1−a)^k restores the unconditional event mass. Thus the components sum exactly to R before stopping. One-low interval corrections have second moment O_η₀(v/T²), since a high mark bounds one squared factor. Two-low and low-diagonal corrections have O_η₀(T⁻⁴) by summing geometric likelihood factors. For T²v large these are all O_η₀(v/T²). Expected acquisition per correction is O_η₀(1); the k low accepted marks cost k/(1−a) on an event of probability (1−a)^k, with zero charge when a=1. Independent block medians, with deterministic Markov acquisition caps and failed blocks counted as bad, give nP=O_η₀(L/v), nQ=O_η₀((1+1/(vT²ε²))log(1/δ)). At tiny margins the additive 1 is absorbed. If nQ is instead of order TL/v, independent T-row Q replay estimates bounded squared risk near the gate with O(L/v) windows by Bernoulli randomization and one-sided Bernstein.

At larger margins omit the corrections. Deleting reads before the first high threshold changes root-risk by at most √2/(aT). Post-threshold marginals are dominated by P/(ca), since the first high mark has density min(1,c/W)/(ca) and later projections preserve P. Omitting low positive lags costs at most 2v/(T a·ca) in risk (sum the bound (1−a)^k v/(ca)). Both errors fit the gate margin for T≥C max{1/(η₀√v ε),1/(η₀²ε)}. The high-only pairs then require O_η₀(L/v) P rows and no Q rows.

### Lower: two independent families

**Proposal information.** Use stable low and top levels plus three dust levels of total target mass u→0. With M=λT set pH=(1−η)/(1−1/M), qH=pH/M, pL=η−u−qH, qL=2η. The remaining Q mass is dust; its conditional Q law is ((1−η)/2,(1−η)/2,η), and conditional P law is (1/6,1/3,1/2). Dust reads f=±1/2; stable reads f=±s, s²=(v−u/4)/(1−u). Baseline conditional means are sd in the lower levels and −sd(1−pH)/pH on H. Perturb only dust sign means by θe where e is orthogonal to both dust probability vectors, normalized by e₁=−1. Mean, variance, overlap, and both sources' weight and read marginals are identical, as are all squared marked tails.

As u→0 dust appears only during a geometric startup of mean Θ(1/η). Its internal sign derivative at step i is Cη[(1−2η)^i−((1−η)/4)^i], Cη→1/3, from the two nontrivial dust eigenvalues. After exit, conditional mean is positive Θ(s) and decays as (1−1/M)^k. Thus the dust–stable cross term gives slope Θ(1/(Tη√v)) in R/v; within-dust slope is O(1/(vη²T²)). At d=0 only same-row moments contribute and the limiting risk is below half; at d=1 collapsing dust marks to their conditional mean gives the maximal refresh risk 2∫₀¹(1−t)e^(−t/λ) above half, and residual dust noise adds nonnegative variance. Calibrate d at an interior common root, then use perturbations θ=±Θ(Tη√v ε). Per-row KL is O(vη²T²ε²) in Q and O(u vη²T²ε²) in P. Data process onto the sign decision, then send u→0 under the uniform P cap.

**Target information.** Use levels wL<1,w1=γT,w2=λT. Assign a fixed small fraction α of high P mass to level 1; impose qᵢ=pᵢ/wᵢ and pL+q₁+q₂=η. For small v, give L read d√v, high means −d√v pL/(1−pL), and high symmetric ±1/4 noise with activity Θ(v). Transfer activity by ±Θ(vε), conserving total target variance. Per unit activity and P mass, level 2 has positive limiting reuse coefficient 2∫_{s+t≤1}(1−e^(−s/λ))e^(−t/λ)dsdt; level 1 has coefficient ≤2γ/(pL+p₁)² by its occupancy and geometric retention. Choose γ small then α small. At zero activity perturbation, the d=0,1 risks bracket half by continuity from the two-level example, yielding an interior root and constant slope in activity divided by v. Row KL is O(vε²) in P and O(vε²/T) in Q. For v bounded below, use binary ±√v instead, shifting high-level conditional signs oppositely to preserve target mean. With α small fixed, γ↓0 makes the lower two levels equilibrate, leaving an aggregate low mean with nonzero slope at the same kind of root; the binary row KL gives Ω(L), equivalent to Ω(L/v) there. The two lower witnesses constrain the same uniform certificate but need not describe the same hidden law.

CPU checks use expanded-state recurrence a'=aK, b'=bK+a'f, c'=c+2(bK)f+a'f² for occupancy, accumulated first and second moments. Independent finite checks cross the common gate in eight dust and twelve target-information cases. These check examples, not the oracle proof or practical throughput.

**Purchase decision.** Under positive per-row prices c_P,c_Q, the tiny-margin optimal cost order is (L/v) min{T c_Q, c_P+c_Q/T²}. For sufficiency of the two inequalities, a sufficiently large n_P+n_Q/T implies either enough Q for full replay or enough P for the corrected estimator; the second inequality handles that branch's Q floor. Equal prices save a factor of order T relative to Q-only replay. Public moments and independent normalized-row access are crucial; this is not a score-evaluation or end-to-end speedup claim.

**A short-window countercheck outside the balanced long-window theorem.** For T=16, μ=.05 and v=.0475, let P=(.95−s,.05−s,s,s), h=(0,1,0,1), Q_A=(.3968−2s,.0032,.599+s,.001+s) and Q_B swap the last two masses. The exact overlap is .4 and the weight marginals under both sources coincide. Exact rational MSE/variance at s=10^-6 is .258483920619 versus .609661452543 (rounding shown); continuity from s↓0 preserves at least .1 gate margin. A P-only stream misses both differing states with probability (1−2s)^n, so no uniform finite P cap works over this slice. Four Q rows suffice for the specified pair at δ=.05 by declaring unsafe if any h=1; P/Q h marginals are not both held fixed here. This counterexample is a scope check, not a theorem for all short windows.

### Reproducing the fixed-overlap result and distinguishing regimes

This section's normalized-score, public-moment relative-margin result is not the [older long-horizon candidate](long_horizon_two_source_frontier.md). The latter uses an absolute MSE gap, different disclosures, and a long-enough-window condition; its normalized variant can use no proposals. Here the high-precision lower is uniform over the specified balanced class and arises from low-weight startup states, not from estimating a normalizer. The relatively coarser-margin branch here also permits zero Q. A single replicate selects two observation times, not two purchased rows; the accepted threshold marks, repeated independent replicates, and optional Q correction all consume counted rows.

From the project root, run `python3 experiments/finite_state/fixed_overlap_frontier.py` (Python 3.9+ with NumPy). The [cloud CPU output](../../results/pro/fixed_overlap_frontier.log) covers 500 threshold-pair decompositions, 108 correction second-moment bounds and both lower-bound families (8 hidden-startup and 12 target-information cases). These are deterministic floating-point finite-state checks, not a formal verification, field throughput result, or proof of the asymptotic rates. Separate written cross-checks audited the proposal-threshold law and both lower constructions, and verified that the hybrid and direct-replay branches cover the two-constraint region at the stated precision.
