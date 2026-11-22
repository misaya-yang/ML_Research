# Two-resource cold IMH audit: isolated target floor and matched P-only corner

Setup: observations from P, observations or named exact normalized score queries from Q, each counted separately. The risk is the finite-window MSE of the **ordinary** IMH time average from actual Q initialization. The algorithm receives no score table. The known-normalization assumption matters specifically for the P-only upper bound.

## 1. Unknown centering forces target information even with arbitrarily many finite Q probes

Fix public balanced signs h=±1 and explicit uniform Q over N names, half per sign. Choose small rational α>0. Randomly assign αN names to common weight H=1/(2α), with a fraction r of those among plus signs, and all other names to common weight l=1/[2(1−α)]. This is a normalized target, η=1/2+α, and the numerical scores and known normalizer are the same under both hypotheses. Compare interior fractions r0=3/4 and r1=r0+Cε (rational approximants allowed; ε sufficiently small). Denote μ_r=P_r h=(1/2−αl)(2r−1).

Unless any of T independent proposal states (including initialization) is high, all weights are equal and the ordinary time average is exactly the mean of the Q-sign path under the same proposal coupling. The chance of any high state is ≤Tα. Since both squared errors belong to [0,4], |R_r−(1/T+μ_r²)|≤4Tα. For a fixed sufficiently large C the two risks separate by >2ε on choosing α≪ε/T.

Fresh P sample: high/low has (1/2,1/2) independent of r. Conditional on high, sign is Bernoulli(r); conditional on low, its plus probability is (1/2−αr)/(1−α). Its KL is O(ε²). Fresh named query at a given public sign, or Q sample together with its sign: high probability is 2αr or 2α(1−r). The KL for either sign is O(α ε²). Uniform random hidden assignments within each sign give a conditional posterior uniform over remaining assignments, so the finite transcript is TV-close to fresh independent conditional colors, with discrepancy O(K²H/N); choose N≫ K²H/δ respecting integer counts. Adaptive budget caps therefore imply n_P+α n_Q = Ω(ε^−2 log(1/δ)). For each pair of finite caps choose α small enough to erase the n_Q term while preserving risk separation. Thus Ω(ε^−2 log(1/δ)) target draws are necessary, independent of the number of Q/named probes, when centering is not supplied. This strengthens the n_P=0 impossibility in SHARP-QUERY, but does not hold for its known-mean model. This lower bound is distinct from the weighted hardness n_Q/T+κ n_P there (κ=max(1/T,η₀), up to constants), and both can hold for the same auditing class, using different hard subfamilies.

## 2. Long-horizon, known-scale corner: zero Q queries, matching P complexity

The exact IMH mark law is P(dy) 1{w(y)≤v} dv/v². Marks with v≥1 occur independently with probability η=∫min(dP,dQ). Their maxima can be drawn by testing k virtual Pareto(1) heights in descending order, drawing one fresh P label per tested height and keeping it if it qualifies. Expected P draws ≤1/η uniformly over k by the original max-height proof. Modify the procedure: **if all virtual heights fail, keep the current state, and do not simulate any low marks**. When a high mark exists the chosen endpoint agrees with the exact k-step kernel, since lower marks cannot affect the endpoint and the current state comparison is preserved. The coupling error for this transition is at most (1−η)^k. All of this uses only P labels with normalized weights.

For the Q-start state draw one P point and keep it with independent probability 1/w(x) if w(x)≥1; otherwise return a dummy with weight 0 and any h∈[−1,1]. This is exactly the law of a Q state with all states at w<1 identified with the dummy: the retained submeasure is P(dx)/w(x) 1{w≥1}=Q(dx)1{w≥1}. A high mark acts identically on the dummy and its actual corresponding low-weight start.

For two iid uniform times I,J in {0,…,T−1}, let A=min(I,J), B=|I−J|. Use the collapsed Q initialization and high-only jumps over A and B. Couple both jumps with the exact mark draws: if a high mark exists on the first jump, the first pair of endpoints coincide (including when their actual initial state was retained); and conditional on that equality, any high mark on the second makes the endpoint pair coincide. The TV error of the two-time law is at most E[(1−η)^A+(1−η)^B]≤4/(Tη), since each A and B atom is at most 2/T. This upper bound counts short windows, including the diagonal, conservatively. The same bound for the first jump only improves with a positive predetermined burn-in.

Two independent P anchors center the product over the two-time law. For |h|≤1 the product lies in [−4,4], so approximation bias in absolute risk is ≤32/(Tη). If Tη₀≥64/ε, this is ≤ε/2. Expected P-label cost per replicate is at most 3+2/η; no Q/named requests are made. Cap blocks at a constant multiple of expected cost, taking the median over independent blocks as in SHARP-QUERY. With deterministic caps this gives the matched order Θ(η₀^−1 ε^−2 log(1/δ)) P labels, zero Q labels in this long-horizon regime (lower bound: existing n_Q=0 weighted result, which grants centering). The result is for absolute error, or divided by a controlled nonvanishing variance; it does not justify a relative certificate where the true risk tends to zero. If normalization is unknown, the initialization acceptance and pivot must not be used unchanged.

A two-regime lower/upper statement is auditable. No intermediate resource frontier is claimed; neither evidence of speedups on model data nor a practical margin is inferred from these asymptotic constants.

## 3. Sharp two-resource frontier (known scale, unknown mean, long horizon)

There are absolute positive constants c, C, c_ε such that for 0<ε≤c_ε, 0<δ≤.1, 0<η₀≤1/64 and integer T sufficiently large with Tη₀≥C/ε, the following characterizes the fixed, deterministic (n_P,n_Q) budget up to the universal constants. Put A=ε⁻² log(1/δ). Any uniformly (ε,δ)-accurate algorithm, for the known-normalized evidence and classes above, needs both
```
n_P ≥ c A
η₀ n_P + n_Q/T ≥ c A.
```
Conversely the same inequalities with the sufficient constant C in place of c imply achievability. Finite support lower bounds allow its cardinality to depend on the caps and parameters. Upper applies on arbitrary standard Borel targets meeting the stated conditions. Risk is **absolute**; δ is statistical probability over calibration, not probability bounding the future path's squared error.

Necessity of the first inequality is the refined hidden-center argument in §1, fixing α after the candidate finite cap pair is supplied. Compare to an ideal synthetic transcript: for conditioned on a fresh name of a specified sign, the exact real high probability reflects the remaining color counts, within O(K/N) of the unconditioned law. For either sampler the total already-exposed mass is ≤KH/N, H=max w; conditional on new names the color-and-sign probabilities are their population values renormalized after removing at most this mass. Given sign and color, new names are exchangeable; the ideal sampler picks an unseen name uniformly within the sign independent of color. Bayesian conditioning on disclosed names/scores leaves the consistent hidden subsets equiprobable, as any transcript likelihood factor for an already seen name depends only on its disclosed weight, and source choices are functions of the transcript and independent random seed. Thus sum of per-step TV errors is O(K²H/N), letting N be taken arbitrarily large while preserving rational counts. A sampler's sign is observed, unlike a named probe where sign is given. The ideal transcript comparison covers adaptive requested names, either-sign named probes and arbitrary cache reuse. KL from fresh P requests is ≤CΔ² per call (Δ=r₁−r₀=Θ(ε)); from fresh Q/named requests ≤CαΔ² per call. Choose α after (n_P,n_Q) so α n_Q is smaller than half the demanded KL contribution and Tα negligible for risk separation. Then choose N after α, ensuring TV error ≤δ independently under both worlds. Binary-data-processing risk separation yields the stated floor.

Necessity of the second inequality is the **existing** three-level lower bound in SHARP-QUERY, not a new weighted hardness theorem. Its fresh-color KL bound in split form is ≤C ε²(n_Q/T + κ n_P) with κ≈max(1/T,η₀)≈η₀ in the long regime. It grants Ph and Var, so also lower-bounds the unknown-center model. The two necessity arguments use different hard families, both inside the same minimax class: every uniformly successful auditor has to pass both.

Sufficiency requires no randomized interpolation. If η₀ n_P ≥ (C/2)A, use §2's high-only P audit with zero Q probes. If not, n_Q/T ≥ (C/2)A; draw independent whole Q-start ordinary trajectories, then draw two separate true P score/observable anchors per trajectory. Multiply h(X_I)−h(Z₁) by h(X_J)−h(Z₂) for independent random times and anchors. The average / block-median has O(A) replication cost and explicit deterministic caps, hence needs O(A) P and O(TA) Q; the first inequality supplies the P budget. (One can average all time pairs on each trajectory at no extra scoring to reduce variance.) This min of two canonical algorithms suffices up to constants; the proof does not estimate sharp constants or practical break-even points.

For a finite safety decision and fixed r∈[0,1], estimate G=R_Q−r VarP(h) by subtracting r/2·(h(Z₁)−h(Z₂))² from each replicate, with the same two independent anchors; its expectation changes by exactly the desired variance term, and its range stays bounded. The P-only approximation affects the two-time product but not the anchor variance term. Select estimation tolerance ε smaller than the promised unsafe/safe gap and issue only when the estimated value plus an established error radius is ≤0. This certifies the MSE, not the realized error of each path. When selecting from a fixed panel, allocate δ across candidates/windows or use separate validation; no familywise assertion from the one-window theorem. The minimax lower transfers to deciding with a promise gap when the threshold is chosen between the two separated risk values; no power guarantee without a margin, especially if always abstaining is permitted.

Supplementary finite sanity: `OPENBLAS_NUM_THREADS=1 python sanity.py`. Independent dense evaluation obtains the high-only lag kernel as K^k − L^k + (1−η)^k I, with L the exact low-mark subtransition. For a deliberately rare-overlap five-state control (η=.0021), omission of low branches is not credible at T=32: true 1.8868 vs observed-population approximation .1645. It becomes controlled as intended at long horizons: T=32,768 true .039894 vs approximate .038595. These are floating-point finite controls, not empirical inference from finite banks.

## 4. Unknown scale: a second independent oracle floor (all sufficiently large T)

The known-scale label on §3 is essential. If the scored calls instead expose a common unnormalized score S=ZW, even arbitrarily many finite P draws cannot, uniformly over the class, replace Ω(A) scored Q/named probes. This statement grants exact Ph and Var but does not provide Z. Here (A=\varepsilon^{-2}\log(1/\delta)) denotes the previous resource scale, not a color.

### Two normalizers and five shared score levels

Let λ be rational in [2η₀,2.01η₀]. Take a rational β≪λ, with five common raw scores (β/2,2β,10λ,T,T²), called A,B,L,M,H. Let the target core color masses be `pL=λ,pM=1/4,pH=3/4−λ−β`, independent of the normalizer. Compare (Z=1) and (Z=1+\Delta) with rational Δ∈[Cε,2Cε] small. These determine the core Q masses by q_j=Z p_j/S_j, so q_L=Z/10, q_M=Z/(4T), q_H≈3Z/(4T²). Write u=1−q_L−q_M−q_H. Low target masses solving p_A+p_B=β and q_A+q_B=u are
```
p_A = (2β/3)(u/Z−1/2),  p_B=β−p_A,
q_A=2Z p_A/β,         q_B=Z p_B/(2β).
```
For T large and Z≤1.1 the q_A,q_B are bounded above and below by nontrivial positive constants, and p_A,p_B=Θ(β). Both worlds are normalized since q_j=Zp_j/S_j; the known raw scores are identical, but `EQ S=Z`. Raw scores sorted by color because β≪λ. Overlap ≥p_L≥η₀. The state space is N uniform public identifiers, each with a paired fair ± sign; the color is uniformly permuted among identifiers with exact counts N q_j. Thus h=±1, Ph=0 and Var=1 in both worlds.

### Risk separation

All acceptance calculations depend on raw-score ratios (not Z directly). Let a_i,m_i be the probabilities of H,M at time i from actual Q start. Exactly,
```
a_(i+1)=q_H+λ_H a_i, λ_H=1−Z/T²,
m_(i+1)=q_M+λ_M m_i−q_M(1−T/T²)a_i,
λ_M=1−Z(p_M+λ+β)/T−q_H.
```
Uniformly for i≤T, a_i is O(T⁻¹) in the C¹ norm with respect to Z. A geometric sum gives
`m_i = [p_M/(p_M+λ+β)](1−exp(−Z(p_M+λ+β)i/T)) + O_C1(T⁻¹)`,
identically to the middle-level derivation in SHARP-QUERY. On each low color acceptance is at least q_L≥.1, and the derivative of its acceptance is O(1), since only proposal masses vary and all ratios are fixed. To see low occupancy derivatives are uniformly bounded too: the restricted low-level update is (1−q_M−q_H) times the conditional low kernel, whose Dobrushin coefficient is ≤1−q_L/(1−q_M−q_H)≤.9 since every low current level accepts L. Its derivative has bounded entries. The derivative recursion for the low mass vector thus has bounded forcing and uniformly bounded total mass derivative (minus derivatives of a_i,m_i); subtracting that total times a fixed probability vector reduces to a uniformly contracting zero-sum recursion with bounded forcing. Inlet terms and their derivatives are bounded because a_i,m_i and derivatives are bounded. This proves the C¹ bound without assuming a spectral gap for the full chain.

For i<j the sign correlation is exactly `sum_v Pr(V_i=v) λ_v^(j−i)`: any accepted proposal refreshes an independent fair sign even when it shares a color. Low contributions therefore sum to O_C1(1/T) due to λ_v≤.9; H contributes O_C1(1/T); diagonal contributes 1/T. The limiting quadratic risk, uniformly differentiated on the parameter interval, is
```
R_Q(0,T) = F(Z)+O_C1(1/T),
F(Z)=[p_M/(p_M+λ+β)] K_0(Z(p_M+λ+β)).
```
On this short argument range (roughly .25–.32), K'_0 is strictly positive with a universal bound from below: write K'_0(x)=2∫u(1−u)e^(−xu)(1−xu)du. Thus risk separability >2ε follows at Δ=Cε with C fixed sufficiently large, for all sufficiently large T.

### Information argument

Fresh P-color KL is (O(β\Delta^2)) because all core masses are identical and the low split changes by O(βΔ) against masses Θ(β). Fresh Q or fresh named query has color KL (O(\Delta^2)): A and B change by O(Δ) with masses bounded away from zero; each core q_j changes by the fraction Δ, uniformly also for the rarest color. The finite hidden-label prior is coupled to the ideal fresh-color transcript as in §3, choosing N after β; max normalized weight is ≤T². Same scoring support in both worlds is important: no returned exact raw score reveals Z outright. Finite caps give KL upper (C\varepsilon^2(β n_P+n_Q)), and β arbitrarily small forces n_Q≥cA by testing, even with known centering.

Consequently, in the unknown-scale data model there are **three** separate lower inequalities in the long regime with unknown centering: n_P≥cA from §1 (its lower gives even the normalized weights), n_Q≥cA from this section, and η₀ n_P+n_Q/T≥cA from SHARP-QUERY (again granting normalization). Only known-scale §3 is claimed matched. The unknown-scale Q floor falsifies any global substitution claim that would remove Q probes using only a P bank and raw scores.

## 5. Stronger long-horizon scale transition: sharp Ω(A/η₀) proposal floor

Under the long-horizon condition in §3, the unknown-scale proposal lower bound strengthens from Ω(A) in §4 to Ω(A/η₀). The distinct §4 bound is valid without any relation between η₀ and T, but §5 is the useful sharp result when Tη₀≥C/ε.

### Four normalized-by-hidden-scale colors

Choose a rational λ∈[2η₀,2.01η₀] and an arbitrarily small rational β∈(0,min(λ/100,.001)). Use four common raw scores S=(β,1/2,T,T²) for colors D,L,M,H. Fix p_L=λ and p_M=1/4. Set p_H=3/4−λ−p_D, impose q_j=Zp_j/S_j and solve normalization exactly:
```
p_D = β [1/Z − 2λ − 1/(4T) − (3/4−λ)/T²] / (1−β/T²).
```
Take rational Z₀=1, Z₁=1+Δ with Δ∈[100ε,110ε], so both lie in [1,1.1] for sufficiently small ε. Then p_D=Θ(β), p_H≥.7, p_j,q_j are positive, q_D≥.8 uniformly for large T and η₀≤1/64, and each normalized weight is w_j=S_j/Z. The overlap is at least p_L≥η₀. All numeric raw scores and public Q probabilities are identical across worlds; hidden counts of score colors differ. Pair sign ± at each unsigned name, as before, so `Ph=0, VarP(h)=1` regardless of the scale; the observer is given these values. Conditional on learning a numerical raw score at one name there is no direct scale leak.

### Uniform risk separation in the long regime

Let a_i,m_i be H/M occupation probabilities as above. Their recurrences are exact as in §4, with p_D in place of β in λ_M, and with derivatives unnecessary. The geometric sums and Riemann limit give (F(Z)+O(1/T)) for the M contribution, (F(Z)=[p_M/(p_M+λ+p_D(Z))] K_0(Z(p_M+λ+p_D(Z)))); the H contribution is O(1/T). A proposal at level L is accepted from either D or L, so low rejection fractions are ≤1−2λ; the total contribution of low-level off-diagonal sign correlations is bounded by (2/(T\cdot 2λ)). The diagonal is 1/T. In particular, uniformly on Z∈[1,1.1],
```
R_Q(0,T) = F_0(Z) + O(1/(Tη₀)+β),
F_0(Z) = [p_M/(p_M+λ)] K_0(Z(p_M+λ)).
```
The β error comes solely from replacing p_D in smoothly varying parameters. The time recurrences use constants uniform in λ∈(0,.032), β≪λ: the middle hazard is between c/T and C/T with c absolute, and the high occupation is O(1/T). (F_0'(Z)=p_M K_0'(Z(p_M+λ))>.04), using (K_0'(x)≥(1−.32)e^{−.32}/3) when x≤.32. Enforce β≪ε and Tη₀≥C/ε with sufficiently large C; each approximation error becomes <ε/2, so the risks differ by more than 2ε.

### Split implications

Fresh P KL is (O(β\Delta^2)) because only p_D,p_H change, by O(βΔ); p_D≥cβ, p_H≥.7. Fresh Q/named KL is (O((η₀+1/T)\Delta^2)=O(η₀\Delta^2)): q_L=2λZ, q_M=Z/(4T), q_H=Zp_H/T² each varies by O(Δ) times its mass; q_D≥.8 makes its own KL contribution O((η₀+1/T)²Δ²). The same randomized-label transcript proof applies with maximum normalized weight ≤T², choosing N arbitrarily large after β despite integer counts. Pick β arbitrarily small versus any fixed n_P, then two-point testing gives `η₀ n_Q ≥ c A` regardless of n_P, where A=ε⁻² log(1/δ). All anchors, exact raw scores and named probes can be adaptive, but do not supply normalized W or p(x). Providing exact normalized W and S visibly at a P draw determines Z immediately, and is a different model.

Consequently **unknown-scale** fixed budgets, for unknown centering, characterized up to universal constants in the same long regime are
```
n_P ≥ c A,
η₀ n_Q ≥ c A,
η₀ n_P + n_Q/T ≥ c A.
```
Necessity uses the refined mean floor (§1), new scale floor (this section), and existing SHARP-QUERY weighted inequality. Sufficiency: if η₀ n_P≥(C/2)A, the existing bridge-calibrated two-source estimator can operate under both caps `O(A/η₀)` (bridge pilot uses that many per source, and cap the expected per-block calls to each source separately, counting either overrun as block failure). Otherwise n_Q≥(C/2) TA and n_P≥CA suffice for exact direct ordinary Q paths centered by independent P anchors, no scale calibration. The **known-scale** class only needs §3's two inequalities and admits n_Q=0. At fixed nonzero overlap both match the simpler constant-overlap resource floors; here the result additionally identifies the correct η₀ dependence as overlap vanishes. The role of long horizons is explicit; no claim for intermediate regimes if Tη₀<C/ε.

### Equivalent simpler three-color hard family (independently found by dual_prior)

Take rational a∈[2η₀,2.01η₀], tiny rational γ>0, and core scale s=1 vs 1+Δ. Set `q_H=s/(2T), q_L=s(a−1/(2T)), q_D=1−sa`; raw scores fixed at `(γ,1,2Ta−1)`. Core P masses are equal conditional on the two core colors; (Z=2sa-s/T+γ(1-sa)) and (Tα_H=s+O(γ/a)) (the core term alone equals s exactly). For Tη₀ sufficiently large the normalized L,H weights exceed one, hence overlap ≥sa. The highest-level occupation converges uniformly on times i/T to (½(1−e^{-si/T})), by its autonomous balance recurrence. The D and L rejection probabilities are ≤1−sa, so their summed lag correlations contribute only O(1/(Ta)). Therefore the limiting MSE is (K_0(s)/2), up to a uniform O(γ/a+1/(Ta)) error. Its derivative at 1 equals (3−8/e>0), guaranteeing separation for small Δ=Cε. Proposal KL is O(aΔ²) by direct inspection, target KL is O((γ/a)Δ²) since core conditional colors are fair and target D mass is O(γ/a). The same random naming and arbitrary-γ reasoning applies, with max normalized weight O(T). This route gives the same proposal query floor, with fewer synthetic levels.

The plug-in upper can also use only P labels *after* an independent two-source scale pilot: let B be an upper estimate with (1≤B/Z≤e^{cε}), take raw pivot B, and use B/S in the initial high-state accept/reject (applied only when S≥B). The exact pivot-B collapse uses Z/S, so initial TV error ≤B/Z−1. Generate high-only marks with pivot B and no thinning; as in the existing scale argument, thinning the largest nominal high with probability Z/B couples to the true high-only maximum, hence TV error ≤1−Z/B per jump. True eligible high probability satisfies (Q[min(1,S/B)]≥(Z/B)η). Short-lag low-branch omissions therefore average to ≤4(B/Z)/(Tη). The total bias of the bounded two-time score is at most 8 times the total TV upper bound, O(cε+1/(Tη)), controlled by the pilot accuracy and long regime. Expected audit P calls are still O(1/η) per replicate; the pilot takes O(A/η₀) draws per source. This yields the same split upper order while eliminating scored Q replay calls, and leaves the exact path upper available in the Q-large branch.
