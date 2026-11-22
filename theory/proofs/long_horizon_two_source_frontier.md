# Candidate theorem: two oracle budgets for the actual cold-start half-variance gate

> **Scope note.** This is the older absolute-gap long-horizon candidate. For known normalized weights, public target moments, and relative high precision, see the separate [fixed-overlap resource frontier](cold_risk_bound.md#fixed-overlap-resource-frontier-for-cold-imh). Its startup proposal-information lower is not the unknown-normalizer mechanism in this file.

This note consolidates proofs for the same fixed decision, not for selecting samplers or changing the estimator. The setup is a prescribed IMH chain started by one draw from Q; use its ordinary mean of h on X0,…,X_(T−1), with fixed |h|≤1. Independent target and proposal draws come with h and a raw score S=Z dP/dQ (positive finite a.s. under equivalent P,Q). Each independently sampled target point counts against nP, each independently sampled proposal point or named score query counts against nQ; arithmetic and randomization do not count. The lower bounds allow public uniform Q, public h, arbitrary adaptive named scoring, and cached answers. They do not grant a normalized target table or visible target sampler internals. The auditor must have false-issue probability at most δ on G≥0 and issue with probability at least 1−δ on G≤−ε, where
```
G = E_Q[(T^-1 sum_i h(X_i) − Ph)^2] − Var_P(h)/2.
```
This is a confidence statement about MSE over a fresh path, not a pathwise-error guarantee. The upper bound takes h, T and the relevant overlap and ratio class parameters as fixed; it does not need μ=Ph or target variance to be supplied.

Let Aε=ε⁻² log(1/δ), 0<δ≤.1, ε sufficiently small, and known overlap lower bound η₀≤1/64. The unified regime is Tη₀≥C/ε for a sufficiently large absolute C. The assertions are up to universal multiplicative constants on the right-hand sides.

**Known scale Z (or valid external O(ε) log-width disclosure):**
```
nP ≥ Aε,
η₀ nP + nQ/T ≥ Aε.
```

**Unknown scale Z and known lower bound W=dP/dQ≥ω with 0≤ω≤c η₀ for sufficiently small absolute c:**
```
nP ≥ Aε,
η₀ nP + nQ/T ≥ Aε,
ω nP + η₀ nQ ≥ Aε.
```
Here ω=0 denotes no positive lower bound. For exact normalization the lower bound may still give true normalized W at every scored point. For the externally disclosed narrow bracket extension the lower bound follows by the same argument even granting exact Z; the upper uses the validated bracket. An externally supplied interval estimated from the same samples does not qualify for free here: confidence and scored costs must first be accounted for.

At fixed overlap, the scale condition binds precisely when the available harmonic-estimation budget ω nP is too small unless one spends proposal calls η₀ nQ. At the risk-driven target-only cost nP≈Aε/η₀, its left P term is (ω/η₀)Aε, so ω relative to η₀ controls whether zero-Q disclosure is effectively replaceable at that rate. No claim for ω>c η₀, small Tη₀, relative/vanishing-variance margins, time-uniform guarantees, or a practical advantage over other algorithms is implicit.

## Necessity: three distinct hard families

1. **Center-information floor, unknown target center.** Use public fair Q signs h∈{−1,+1}; an arbitrarily small fraction α of names has common high normalized weight .8/α, other names have common weight .2/(1−α). Assign a fraction r of hidden names to plus signs. Exact μ=(.8−αl)(2r−1), Var=1−μ². If all T proposals avoid hidden names, couple the path to the underlying unconditional iid Q-sign sample with mismatch ≤Tα. Thus G approximates 1/T+(3/2)μ²−1/2 within 4Tα. Tune around its interior root μ≈1/√3 and tilt r by Θ(ε), staying away from zero/one. Choose α tiny relative to ε/T and any finite Q cap. Fresh P KL O(ε²), Q or conditional named Q KL O(α ε²); this gives nP≥c Aε. Overlap ≥.2 and minimum W≥.2, so this family meets all stated η₀,ω bounds. See the weighted witness appendix §Independent target floor.

2. **Weighted cold-dynamics floor, even granting the target center and variance.** For η₀≤1/4096 the family in the weighted witness appendix uses l=2κ, m=400Tκ, H=T², κ≈η₀. M all minus, conditional low mean near .9 and conditional heavy mean O(κ), calibrated so Qh=.9 and exact Ph=0, Var=1. With q_M=c/T the cold risk tends to a two-state continuous-time risk F(c,.005) starting low, whose derivative is below −.096 throughout [.9,1.2] and which crosses the actual .5 threshold. Finite discrepancy is O(1/T). Uniform ±Θ(ε) c tilts give separated risks; KL is O(κ ε²) per P and O(ε²/T) per Q/named, proving η₀ nP+nQ/T≥c Aε. Its smallest normalized score l≥2η₀ exceeds ω. **For 1/4096<η₀≤1/64, this particular family is not claimed:** the independent center floor already yields η₀ nP≥(c/4096)Aε and hence the necessary weighted inequality at a smaller absolute constant. The general theorem does not say the center-known weighted floor was verified up to 1/64.

3. **Scale-information floor, unknown Z, even granting target moments.** Use the positive lower-ratio appendix's last construction: public balanced signs in a normalized IMH with shared raw levels γ,2γ,1,H. Core low/high sign deterministic −/+ and exact Ph=0, Var=1 achieved by balancing dark sign mass. Set γ=4aω for a≈2η₀ when ω>0 (small absolute ω/η₀), or arbitrarily tiny γ after fixed caps if ω=0. Then the smallest W=γ/Z≥ω, overlap≥η₀, and the raw alphabet is identical in both worlds. The nontrivial slow mode is the two-state threshold projection; residual signs inside its low side have mean-zero under the reflected conditional target and decay at rate O(a) from a Doeblin bound. Thus all residual risk increments sum to O(1/(Ta)), even with frequent dark jumps, and the projected smooth risk crosses 1/2 via calibrated core-proposal multiplier s≈1 with nonzero derivative. Finite error is O(1/(Ta)), so rational ±Θ(ε) tilts separate the target gate. Fresh P KL O(ω ε²) (O(γ/a·ε²) when ω=0), Q/named KL O(η₀ ε²). The latter condition uses raw scores only: if a request additionally supplies exact normalized target probabilities at the returned label with known Q, it discloses Z and is a different oracle. This proves ω nP+η₀ nQ≥c Aε.

All three families randomize known-count color types over a very large public uniform name set, within fixed public sign classes. Conditioning on the transcript leaves an exchangeable permutation of unrevealed colors (the likelihood of an already disclosed name depends only on its disclosed weight). If K=nP+nQ the next named or sampling response differs from the corresponding fresh-color model by O(K max W/N), accounting for depletion and repeat. Choose N after caps, normalized weights and rational probabilities so sequential coupling gives ≤δ/4 TV. Chain rule on adaptive query choices then charges each source its own KL budget; data processing for safe versus unsafe hypotheses needs Ω(log(1/δ)) KL. The three inequalities use different families in one model class; they are not asserted to hold simultaneously for every individual instance.

## Sufficiency: one fixed-budget switch with independent blocks

Use G directly, not a surrogate threshold or an estimated predictive mean replacement. For a realized two-time pair with i,j iid uniform in [0,T−1] and independent target observable anchors h(Z1),h(Z2), take
```
U = (h(X_i)−h(Z1))(h(X_j)−h(Z2)) − (h(Z1)−h(Z2))²/4.
```
The variance subtraction has expectation Var_P(h)/2 because Z1,Z2 are independent P observations. Its correlation with the product does not alter expectation. |U| is bounded by a universal constant. One-sided estimation via independent capped blocks and a median obtains an error ≤ε/4 with failure ≤δ (allocate pilot error first where applicable); issue whenever a valid upper bound on G is strictly below 0. Any G≤−ε is certified on the good event. For independent exact direct paths this costs O(Aε) target anchors and O(T Aε) proposals.

- **Large Q cap branch.** If nQ≥C' T Aε, run ordinary independent Q-start paths exactly using raw score ratios, no normalization calibration. By nP≥C''Aε there is enough target budget for anchors.
- **Target-rich branch.** Otherwise the weighted inequality, with a suitably large overall sufficient constant, provides nP≥C'' Aε/η₀. Use the high-only approximate endpoint estimator, which employs P-only collapsed initialization and maximum eligible marks in the two random lag jumps. For known scale the bias is O(1/(Tη₀)) and each two-time replicate uses O(1/η₀) expected scored P labels; no Q calls. Cap independent blocks at a constant multiple of expected count, counting each overrun as a bad block, then take the median. For externally supplied positive A≤Z≤B with log(B/A)≤cε, the scale bracket adds O(ε) deterministic bias via uniform-in-lag top-mark thinning; choose c conservatively and keep the separate statistical radius.
- **Scale unknown in the target-rich branch.** If ω nP≥C''Aε, estimate Z independently from a reserved P bank using the harmonic identity E_P(1/S)=1/Z and E_P W^-2≤1/ω; block-median relative accuracy gives a valid O(ε) log bracket using O(Aε/ω) P labels and zero Q. Otherwise the scale inequality gives nQ≥C'' Aε/η₀, while the weighted branch already gives nP≥C'' Aε/η₀; bridge scale calibration with harmonic overlap ≥η₀/2 returns a valid O(ε) log bracket using O(Aε/η₀) independent draws of each source. The audited path estimator then uses only fresh P other than the bridge banks. Large absolute sufficient constants ensure splitting each cap between calibration, audit and anchors is possible; the finite score caps do not rely on optional overrun hidden work.

**Finite verification:** half_gate.py and weighted_half_gate.py independently verify the fixed-sign threshold crossing on finite matrices (exact rational normalization and public sign/target moments, binary64 MSE recursion). These are synthetic controls for the proofs, not a real-workload result. The refined residual projection for ω>0 has an algebraic proof in the positive lower-ratio appendix; ratio_floor_fixed_omega.py independently checks three shrinking-gap finite cases at fixed ω=.00015625 and γ=1.953125e−5, each crossing .5 with exact rational moments.


---

## Appendix: Weighted witness and center floor

# Weighted and target-only information floors for the fixed half-variance gate

These strengthen existing absolute-risk lower bounds to the actual fixed decision G=R_Q(0,T)−Var_P(h)/2, with known-normalized evidence. They also hold under a weaker raw-score oracle. Let A=ε⁻²log(1/δ), δ≤.1, ε sufficiently small. For each, choose finite equal-sign or fixed-sign-frequency populations with exact rational counts; exact normalized scores at named points are allowed. Overlap target is η₀≤1/4096. The long cut Tη₀≥C/ε below is inherited from the claimed regime and simplifies using a continuous-limit root.

## Weighted floor, even with Ph=0 and VarP(h)=1 revealed

Take rational κ∈[η₀,1.01η₀], score levels l=2κ, m=400Tκ, H=T², and c near the interval [.9,1.2]. As in the old 3-level risk-estimation family, set q_M=c/T, q_H=(1−l−(m−l)q_M)/(H−l), q_L=1−q_M−q_H, and p_j=w_j q_j. These define valid normalized probabilities with q_H=Θ(T^-2), p_H bounded below by a positive constant, overlap at least l q_L≥η₀, m>1 and l<1 for the chosen long cut. The new h is binary ±1 with conditional means
```
E[h|M]=−1,
v=E[h|H]=[(m−l)q_M−.9l]/[(H−l)q_H],
u=E[h|L]=[.9+q_M−q_H v]/q_L.
```
Consequently Qh=.9 and Ph=0 exactly, and Var_P(h)=1 exactly. The same public sign table (95% plus, 5% minus) applies in both hypotheses. u=.9+O(1/T), v=O(κ), with lower/upper h-cell probabilities bounded away from zero on L,H absolutely (only M has a fixed zero plus cell). The explicit formulas and compact c range give |u'(c)|=O(1/T), |v'(c)|=O(κ). At a parameter pair separated by Δ, target cell KL is O(κΔ²) and proposal cell KL O(Δ²/T): M masses Θ(κ) and Θ(1/T) respectively move by relative O(Δ); L observed-sign cells under Q move by O(Δ/T) over constant masses, under P they are multiplied by l; H Q masses Θ(T^-2) move by O(κΔ/T²) and H P masses Θ(1) move by O(κΔ). Conditioned on a chosen public sign, the same proposal KL bound holds up to a factor 20, because Q sign probability is fixed and at least .05 under both hypotheses.

### Verification of a genuine crossing

The coupling of T underlying proposals (including initialization) to the H-free process can replace each H proposal by a new conditional-L proposal; discrepancy probability ≤T q_H=O(1/T). The replacement proposal is iid with middle probability q_M. In the resulting trajectory low states always accept, and labels of low states are independent conditional on the level paths and the acceptance uniforms: each surviving low state is a fresh initial or accepted low proposal at each low time. Thus replacing low marks by their conditional mean u changes E[(path average)²] by O(1/T), because conditional variance is at most 1/T. Replacing u by .9 incurs O(1/T) additional error. The remaining two-state level chain has L→M=c/T and M→L=(1−c/T)(l/m)=(1−c/T)d/T, d=.005. Initialization differs from L deterministically by only c/T. Coupling the discrete transitions to unit-normalized time CTMC observations spaced by 1/T costs O(1/T) in total mismatch probability: exponentiating the bounded generator gives per-transition O(1/T²) difference. On the CTMC trajectory replacing the grid average by the time integral costs O(1/T) in expected absolute difference, since only intervals with a jump contribute and the expected number of jumps is bounded. The MSE therefore equals F(c,d)+O(1/T) uniformly, where F(c,d) is the CTMC expectation of the squared integral of .9 on L and −1 on M, starting from L.

At d=0 the low holding time is E~Exp(c), so F(c,0)=E[(1.9 min(E,1)−1)²]. Write b=1.9. Exact bounds checked with damped Taylor polynomials in weighted integrals (see script):
```
F(.9,0)>.522, F(1.2,0)<.479,
∂_c F(c,0)=−2b integral_0^1 u(bu−1)e^(−cu) du < −.106 on [.9,1.2].
```
For the derivative bound, split at u=1/b and choose smaller c=.9 on the negative subintegral, larger c=1.2 on the positive subintegral to lower-bound that integral. Alternating exp polynomials degrees 13/14 give rigorous upper/lower integrands on the compact interval, and polynomial integrals are rational. The down-clock perturbation satisfies |F(c,d)−F(c,0)|≤d because using the same up-clock makes trajectories identical if the extra rate-d Poisson clock never ticks; the functional is in [0,1]. It also satisfies |∂_c F(c,d)−∂_c F(c,0)|≤2d. Indeed the Poisson derivative in c equals the integral over uniform added up marks of the expected effect of adding one such mark. This identity follows by differentiating the absolutely convergent series conditional on number of up marks (given that count, marks are iid uniform); use a shared down-clock for both comparisons. For each condition the functional differences coincide if there is no down mark and differ by at most 2 otherwise. Hence F(c,.005) has an interior half-risk root and strictly negative derivative bounded away from zero on the entire interval. Choose rational c_* close enough to that root to make |F(c_*,d)−.5|<ε/4, then rational alternatives c_±=c_*±Θ(ε) at a sufficiently large fixed multiplier. With T sufficiently large of order ≥1/ε, the O(1/T) discrepancy does not erase the crossing, yielding R(c_−)≥.5+ε, R(c_+)≤.5−ε. This establishes the desired half-variance separation on the authentic finite window.

### Query testing

Reveal publicly an explicit uniform Q on N names and the sign table, with all N×q_j×conditional sign counts integral, randomly permute colors within each sign. As in the earlier lower proof, conditioned on any adaptive transcript, the posterior over remaining labels within each sign is uniform because the history likelihood depends only on revealed weights. Depletion and repeats affect each next type probability by TV O(KH/N), K=n_P+n_Q; the new name given type/sign is uniform across unseen names of that sign. Taking N arbitrarily large gives TV ≤δ/4 of the fresh ideal color transcript (including adaptive names, caches, private randomness and either sampler). KL chain rule supplies total O(ε²(n_Q/T+κ n_P)). Safe-vs-unsafe testing with ideal errors ≤5δ/4 needs Ω(log(1/δ)) KL. Thus `n_Q/T+η₀n_P≥c A` for the fixed gate, with no need for unknown centering. This uses the old weighted family recurrence structure but new signed colors and true threshold calibration.

## Independent target floor when centering is not supplied

Take an explicit Q with equally many public plus/minus names and a tiny fraction α of hidden high-score names, weight .8/α, fixed target high mass .8. Low weight l=.2/(1−α). Among hidden names choose plus fraction r∈(0,1), uniformly random locations in each sign matching counts. Exact Ph=μ=(.8−αl)(2r−1) and Var=1−μ², neither given to the auditor. At a fixed finite T no-hidden proposal path agrees with unconditional iid balanced Q signs in the common-proposal coupling; mismatch probability ≤Tα. Therefore G differs from 1/T+1.5μ²−.5 by at most 4Tα, uniformly (squared errors lie in [0,4]). The limiting calibration μ_*=sqrt((1−2/T)/3) is bounded away from zero and from .8 for large T, so the corresponding r_*≈.861 is an interior root with nonzero slope. Select rational alternatives r_±=r_*±Θ(ε), choose α≪ε/T, and cross ±ε exactly at the fixed half-variance gate. Overlap≥.2. Source KL exactly as earlier mean-floor family: every P query supplies O(ε²), Q or named O(α ε²), with names handled by arbitrary N. Let α tend as small as necessary after any finite n_Q cap. Hence `n_P≥c A`. It is essential that the target moments not be revealed for this second family.

Together with half_gate.md (unknown scale's extra Q floor) and the earlier P-only/direct-path upper proof, these make all previously claimed long-horizon two-resource constraints valid for the actual fixed half-variance decision with promise gap. The result does not establish the full frontier at shorter horizons, certificates for trajectories one by one, or new empirical efficacy.


---

## Appendix: Positive lower-ratio and scale witness

# A continuous substitute for disclosure: known lower bound on W

This is a separate analytical continuation after the Mac snapshot and should be independently audited. It does not yet extend the fixed half-variance risk theorem outside its original scope.

## Normalizer-only problem

In the same exact-ratio black-box oracle, assume overlap >= η0 and known W=dP/dQ >= ω everywhere, for 0<ω<=cη0 and small η0. Let A=ε^-2 log(1/δ). The minimax hard cap frontier for estimating log Z up to ε is, up to universal constants,
```
η0 nP >= A,
ω nP + η0 nQ >= A.
```
Q-side paid exact named queries are allowed. There is no upper bound on W. Conditions are worst case over unbounded finite support.

### Sufficiency

If ω nP is large enough, use P-only median-of-block-means of 1/S=1/(ZW). Its mean is 1/Z (equivalent laws), and its normalized second moment is E_P W^-2 <= ω^-1 E_P W^-1=ω^-1. Chebyshev with block size O(ω^-1 ε^-2) and median amplification returns relative accuracy, hence log accuracy, with zero Q. Algorithm need not know Z to do this. Otherwise if the displayed inequalities hold with sufficiently large constants, both nP and nQ are O(A/η0) or larger. Use the balanced bridge bound for relative/log-scale accuracy with harmonic overlap comparable to η0.

### Necessity

Separate target floor: the target-hard two-color family in prior/NORMALIZER-MINIMAX.md has S_L=1, S_H=1/(βα), q_H=αs, β≈2η0. Worlds s≈1 differ at rate Θ(ε). For α tiny, low normalized weight 1/Z_s is of order β, >=ω after choosing c small, high weight greater. The original KL result gives η0 nP>=cA despite arbitrarily many Q calls.

Mixed scale floor: let Q(H)=βs, Q(L)=1−βs, raw scores (S_H,S_L)=(1,γ), β rational near 2η0, γ rational near 4βω. With s∈[1,1.1], Z_s=βs+γ(1−βs), the smallest normalized score is γ/Z_s >= ω, overlap >= β. Two rational s at distance Θ(ε) separate log Z by >2ε. P(L)=γ(1−βs)/Z_s=Θ(ω), derivative O(ω), and Q(H)=Θ(η0), derivative Θ(η0). The two color KL are respectively O(ω ε²) and O(η0 ε²). Public uniform Q and hidden random permutations reduce adaptive named and scored draws (separate caps) to those colored transcripts as in the original proof. Binary data processing gives the second inequality.

In particular the unknown-scale lower does not say no P could ever substitute in a bounded-score class. It is the missing *lower* bound on W (inverse ratio tail) that made Q irreplaceable in the earlier worst-case class, unlike Pro's hard-to-estimate target center from the *upper* ratio tail. This provides a clean contrast between the two mechanisms.

## Transfer back to cold risk, partial

The P-only upper carries over in the long regime by independent harmonic calibration to O(ε) log width + bracket endpoint simulator; nP=O(A/ω+A/η0). The finite half-variance scale witness should give a matching resource inequality on a sub-regime with ω small relative to ε, since the dark state no longer can be made arbitrarily rare. Need an exact finite dynamic sensitivity bound at fixed ω to reach arbitrary ε; do not claim full risk theorem from the pure-normalizer case. A bounded reward tilt offers a concrete way of obtaining a deterministic W lower bound from score range when both normalization and target generation costs are accounted for.

## Fixed half-variance risk: removing the restriction ω << ε

Derivation added after the initial partial note. This is a candidate sharp scale constraint for the same fixed gate; seek independent proof audit.

Use the public-sign six-color half-gate construction (raw scores γ,2γ,1,H with dark colors split between ±1) but set γ=c_γ aω with a≈2η0 and a sufficiently large fixed c_γ instead of sending γ to zero. Take ω≤cη0 with sufficiently small absolute c; then all weights are >0 and the minimum γ/Z_s≥ω for s in a fixed neighborhood of 1, since Z_s=2sℓ+1.4γ, ℓ=a−d, d=c_*/(2T), H=ℓ/d; e.g. c_γ=4 and s∈[.97,1.03] work for sufficiently small ω and large T a. Exact P h=0, VarP h=1, Q h=0 and common raw alphabet still hold. For the two worlds s≈1 separated by O(ε), per-draw KL is now O(ω ε²) on P and O(η0 ε²) on Q/named. Need show finite risk crosses .5 after an O(1/(Ta)) approximation whose constant is uniform in ω, not the old O(Tγ).

### Exact projection and decay of mark residual

Let B be the high raw state H and L its complement. Write h=f(B-or-L)+g, where f(B)=1, f(L)=−v with π=P(B), v=π/(1−π), and g has zero P conditional mean inside L and is identically 0 on B. The two-state indicator chain is exactly Markov: a low state goes to B with rate s d, independently of the low name, and a high state goes to low with probability (sℓ+1.4γ)/H. The reflected (conditional-on-next-low) kernel K_L on L preserves P_L, since detailed balance restricts off-diagonals and its survival row sum is 1−s d for all low states. From B, conditional on accepting a move to L, the arriving label has exactly law P_L. Also K_L(x,dy)≥Z_s P(dy) on L because each pair of low raw scores has maximum at most 1 (minorization holds on all sets including diagonal, whose off-diagonal expression can be extended using nonnegative rejection). Hence the chain on L contracts deviations in total variation by (1-c a) each step, uniformly given ω small.

The above imply |K^k g|≤C(1-ca)^k, since a high passage restarts the low labels in P_L and the only other case stayed low throughout. Likewise |E_Q g(X_i)|≤C(1-ca)^i by separating paths that already visited B. Expand E[h(X_i) h(X_j)] for i≤j. Terms f K^(j−i)g and g K^(j−i)g are bounded by C(1−ca)^(j−i); E[g(X_i) K^(j−i)f(X_i)] is K_B-or-L^(j−i)f(L) times E g(X_i), uniformly bounded by C(1−ca)^i. The average over the square is O(1/(T a)); no need for the dark event never to occur.

### Calibrate limiting projected risk

Let x=T Z_s/H = c_*[s + .7γ/ℓ]. The nontrivial two-state eigenvalue is 1−x/T, its stationary high prevalence is π=sℓ/(2sℓ+1.4γ), and f is a centered eigenfunction. Hence for i≤j,
```
E[f(X_i)f(X_j)] = v (1−x/T)^(j−i)
               + (E_Q f(X_0)^2 − v)(1−x/T)^j.
```
Here E_Q f(X_0)^2=v²+O(1/T). Finite triangular sums converge uniformly on compact x>0, with error O(1/T), to
```
R_proj = v J(x) + (v²−v) Γ(x),
J(x)=2∫_0^1 (1−u)e^(−xu)du,
Γ(x)=2∫_0^1 u e^(−xu)du.
```
In these smooth quantities replace ℓ by a at loss O(1/(Ta)). For fixed λ=.7γ/a = O(ω), the limit has x=c_*(s+λ) and v=s/(s+2λ). At λ=0 it is J(c_*s), with unique .5 root c*≈2.557 at s=1 and a negative nonzero derivative in s. On a sufficiently small universal λ neighborhood the derivative in s stays bounded away from zero at c* near that root (smooth, compact perturbation). Pick a rational c_* arbitrarily close to the root of limiting risk at s=1 for the actual λ, respecting integer count requirements. Risk in the actual finite chain differs by O(1/(Ta)) uniformly, so Tη0≥C/ε with large C suffices to cross ±ε at rational s±=1±C′ε. Binary data processing and the hidden-name reduction give
```
ω nP + η0 nQ >= c ε^-2 log(1/δ)
```
even for the half-variance gate and donated target moments. This is a scale-information constraint, not yet a complete new minimax characterization of the half-gate class. It matches the scale calibration requirement when using the existing long-window risk estimator. It is a genuine intermediate-access law, rather than assuming the scale is either fully disclosed or essentially unrecoverable from P.

Potential fixed-gate center floor adaptation, independent of this proof: the center-hard family uses hidden target mass .8 rather than .5, then G≈1/T+1.5μ²−.5 can cross at μ≈1/√3. This is not needed for the scale comparison; check separately if extending all lower inequalities to the half gate.

## Bounded-soak witness for the weighted dynamics channel (half-variance gate)

This supplements the earlier unconstrained long-window witness, and applies to the dynamics term only. It is an independently audited construction, not a claim for a tail envelope that drops before 3T.

Assume normalized scored oracle, h=±1, exact center 0 and variance 1 granted, unburned Q-start window T, κ rational near overlap floor η0 with 0<κ≤1/4096. Fix normalized weight alphabet l=2κ, m=400Tκ, H=3T. Let q_M=c/T; q_H=[1−l−(m−l)q_M]/(H−l), q_L=1−q_H−q_M. If low conditional sign mean is u, middle sign is always −1, and high conditional sign mean is v, let

v=[(m−l)q_M−.9l]/[(H−l)q_H],   u=(.9+q_M−q_H v)/q_L.

This preserves exact P h=0, Q h=.9, positive masses uniformly for c in [1,9/8] and all large T. Exact overlap ≥l q_L≥κ, minimum W=2κ, supremum W=3T. Public named scores use uniform hidden labels with fixed public sign counts .95/.05; the score alphabet is identical across c.

Coarse chain L,M,H+,H− is exactly Markov with Q-start low up to O(1/T), and all distinct-time low signs independently refreshed conditional on coarse path (on low→low a low proposal was drawn and accepted; on heavier-to-low acceptance does not depend on the low sign). Thus for integrated squared path average it is safe to replace low labels by u=.9+O(1/T) except diagonal difference O(1/T). The CTMC generator G on the time interval [0,1] is the limit of T(K−I), with rates:
- L→M=c; M→L=.005;
- L and M → H± at rate a±=(S±B)/6, S=1−2κ−400κc, B=400κc−1.8κ.
- H±→L=2κ/3, H±→M=400κc/3, H±→H∓=a∓.

Generator/initial/observable errors in finite time yield a uniform O(1/T) difference in risk (apply telescoping power bounds for K vs exp(G/T) and bounded Riemann quadrature). f=(.9,−1,1,−1). For CTMC moment, let U'=GU+f, V'=GV+2 diag(f)U with 0 initial data. R(c,κ)=V_L(1). The exact-rational augmented matrix exponential and derivative verification are reproducible with the code below. The infinity norm Taylor remainder after N=40 terms is at most 2 ∥M∥∞^(N+1)/(N+1)! (the ratio of successive terms at most 1/2).

Grid covers c∈[1,9/8] by 17 nodes and κ∈[0,1/4096] by 33 nodes. Straight row sums give ∥G(c,κ)−G(c,κ')∥∞≤610|κ−κ'|, ∥∂_c(G(c,κ)−G(c,κ'))∥∞≤534|κ−κ'|, ∥∂_c G∥∞≤2.131. For two-time product P(u) diag(f) P(v) f with u=min and v=lag, Duhamel plus ∥P∥∞=1 bounds κ perturbation by 610|Δκ|(u+v). Its c derivative's κ perturbation is bounded by [534(u+v)+610*2.131(u+v)^2]|Δκ|, and c second derivative by 2.131²(u+v)^2. Average using E max=2/3 and E max²=1/2; conservatively Lipschitz 410 for R in κ, 1020 for ∂cR in κ, and 2.4 for ∂cR in c. Grid envelopes after these slacks verify R(1,κ)>.50351623, R(9/8,κ)<.49619194, ∂cR<−.09222457. Strict rational assertions use .501/.499/−.07. This is interval proof over a continuous rectangle, not a point estimate or fitted numerical derivative. Positive spacing near each endpoint and nonzero slope allow fixed opposite Θ(ε) c perturbations of the interior critical value; finite O(1/T) error negligible for T≫1/ε. Rational approximants retain fixed signs and overlap.

Independent finite verification used exact Fraction arithmetic for P/Q normalization and moments and extended-precision direct five-state propagation. With T=32768,65536,262144, κ=8/T and c offsets ±.025, finite risks were (.502956,.497236), (.502980,.497165), (.502998,.497112). This independent check confirms the implementations and not by itself the asymptotic inequalities.

Per-draw KL from changing c by Θ(ε): Q middle mass Θ(1/T) changes Θ(ε/T); Q each high sign mass Θ(1/T) changes O(κ ε/T), low Q signs positive constants shift O(ε/T). Thus O(ε²/T) per Q/named (public sign correction bounded by 20). P middle mass Θ(κ) changes O(κ ε), each high sign constant mass changes O(κ ε), low signs Θ(κ) change O(κ ε/T). Thus O(κ ε²) per P. Hide each type by exact integer counts over arbitrary sufficiently large public support within each sign. Named-probe depletion/repeat coupling O(K²(3T)/N) as in earlier proof. Adaptive KL chain rule and testing then give κ n_P+n_Q/T ≥ c ε^−2 log(1/δ), even known Ph and Var. Does not prove independent center or scale floors in the additionally W≤3T class. The centered-hidden-state hard family needed arbitrarily high W. In this E3 example actual point-tail P(W>x) has a dominant atom (order-one P mass) at 3T and zero beyond 3T; a restrictive upper envelope below that atom excludes it. A truncated-mean tail is not zero beyond 3T, but decays exactly as first moment divided by threshold there. W≥ω is valid for ω≤2κ. D_median=3T, so this worst-case witness does not imply a local cost optimum in a class with small target median.

### Exact rational certificate

The following standard-library Python checks every Taylor enclosure and the continuity slack. Successful assertion completion is the certificate; floating-point displays are illustrative.

```python
"""Check the soak ceiling W_max=3T in the coarse CT half-gate family."""
from fractions import Fraction as F
from math import factorial
import numpy as np
from scipy.linalg import expm
ZERO=F(0);ONES=F(1)
def mat(c,k,C=F(3)):
 f=[F(9,10),-ONES,ONES,-ONES]
 S=1-2*k-400*k*c
 sv=400*k*c-F(18,10)*k
 arP=(S+sv)/(2*C);arM=(S-sv)/(2*C)
 G=[[ZERO,c,arP,arM],[F(1,200),ZERO,arP,arM],[2*k/C,400*k*c/C,ZERO,arM],[2*k/C,400*k*c/C,arP,ZERO]]
 for i in range(4):G[i][i]=-sum(G[i])
 B=[[ZERO]*9 for _ in range(9)]
 for i in range(4):
  B[i+1][0]=f[i];B[i+5][i+1]=2*f[i]
  for j in range(4):B[i+1][j+1]=G[i][j];B[i+5][j+5]=G[i][j]
 return B
def exp_r(c,k):
 return expm(np.array(mat(c,k),float))[5,0]
def cert(c,k,N=75):
 A=mat(c,k);B=mat(c+1,k);C=[[B[i][j]-A[i][j] for j in range(9)] for i in range(9)]
 M=[v+[ZERO]*9 for v in A]+[v+w for v,w in zip(C,A)]
 d=max(sum(abs(v) for v in row) for row in M)
 assert d/(N+2)<F(1,2)
 tail=2*d**(N+1)/factorial(N+1)
 v=[ONES]+[ZERO]*17
 risk=der=ZERO
 for j in range(1,N+1):
    v=[sum(a*b for a,b in zip(row,v))/j for row in M]
    risk+=v[5];der+=v[14]
 return tuple(map(float,(risk-tail,risk+tail,der-tail,der+tail))),risk,der,tail

"""Exact rational grid and semigroup-Lipschitz extension in c and kappa."""
import json
from fractions import Fraction as F
J=32;K=16
kl=[F(j,4096*J) for j in range(J+1)]
cl=[1+F(j,8*K) for j in range(K+1)]
a=F(1,2);b=F(1,2);d=F(-100)
# d bound max derivative (closest to 0); a min left risk, b max right risk.
r=[]
for k in kl:
    row=[]
    for c in cl:
        _,rv,dv,eps=cert(c,k,40)
        row.append((c,rv,dv,eps))
    a=min(a,row[0][1]-row[0][3]) if r else row[0][1]-row[0][3]
    b=max(b,row[-1][1]+row[-1][3]) if r else row[-1][1]+row[-1][3]
    d=max(d,max(x[2]+x[3] for x in row))
    r.append(dict(k=float(k),left=float(row[0][1]-row[0][3]),right=float(row[-1][1]+row[-1][3]),dRdc_upper=float(max(x[2]+x[3] for x in row))))
    print(r[-1],flush=True)
err_v=F(410)/(4096*J*2)
err_d=F(1020)/(4096*J*2)+F(12,5)/(8*K*2)
print('test worst grid and continuous lower-left/upper-right/upper derivative',*[float(x) for x in (a,b,d,a-err_v,b+err_v,d+err_d)],flush=True)
assert a-err_v>F(501,1000) and b+err_v<F(499,1000) and d+err_d<-F(7,100)
```
