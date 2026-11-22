# Fixed-target-budget power and the equality boundary

Sources: Pro result and complete reproduction archive.

Part I is the original Pro theorem as independently checked. Part II is an additional equality-cap obstruction, not a general equality classification. The pointwise envelope concerns randomized certification as proposal budgets tend to infinity, with no uniform finite-budget approach rate.

## Part I: Exact pointwise power

# Exact pointwise power ceiling with a fixed target bank

## Statistical experiment

Fix a public proposal law Q, a measurable nonconstant binary readout h and a finite window b,t. A candidate target is represented by a positive finite measurable raw score S, with Z=Q S finite and positive, and P(dx)=S(x)Q(dx)/Z. Z is not supplied. A scored P row draws X independently from P and reveals its name, h and S. A Q row draws independently from Q and reveals the same marks. For the necessity result, any of the q queries can instead score an arbitrary named point. Source draws/prior scores can be used adaptively up to the two hard quotas. No structural restrictions on the unknown score other than positivity and integrability are available for certifying false-issue control. The target distribution and HMM code used to synthesize draws cannot be inspected. Computational costs are outside this oracle model.

From X0~Q, use the ordinary independent-Metropolis kernel, observe A=t^(-1) sum h(X_i) over times b,...,b+t-1, and write a=EA,d=EA², mu=Ph. For binary h the true deployment margin is

$$F(m)=d-(2a+1/2)m+(3/2)m^2,\qquad G=F(\mu).$$

An issuer is honest at level δ uniformly if every admissible raw S with G≥0 has probability of issuing at most δ. The observation window, h and proposal are fixed before the audit. Private randomization is allowed. Fix a strictly safe base model S0 with 0<μ0<1. The base quadratic has two distinct roots ℓ<μ0<u in (0,1). Indeed F(0)=EA²>0 and F(1)=E(1−A)²>0, since a run of all zero proposal labels and one of all one proposal labels have positive probability. Throughout, ℓ,u,a,d,μ0 refer to the base population and are unknown to the issuer.

Define the contamination fractions
$$\kappa_0=1-\ell/\mu_0,\qquad
\kappa_1=1-(1-u)/(1-\mu_0),\qquad
\kappa=\min(\kappa_0,\kappa_1).$$

Assume that for at least one minimizing sign j, {h=j} contains arbitrarily many disjoint measurable subsets of strictly positive Q mass. This holds in nonatomic models for both signs; multiple atoms among one or both labels are also allowed. At any fixed integer P cap p≥0, every honest issuer and every finite q has clean/base issue probability at most

$$\Pi_p=\min\{1,\delta/(1-\kappa)^p\}.$$

There is a single procedure (no unknown clean parameters as input) that is honest for each finite q and whose power tends to Π_p as q→∞ at every such base simultaneously, in the pointwise sense. There is no uniform rate in q in this theorem. The upper construction works generally on fixed finite support but the stated necessity need not. Requiring probability 1−β at this base needs p≥ceil(log((1−β)/δ)/[−log(1−κ)]) when 1−β>δ. Exceeding this power by a fixed margin at the limiting ceiling is enough to attain the requested power at all sufficiently large q. At the integer boundary, equality of limiting power does not imply finite-q attainment.

## Proof of necessity, including adaptive named probes

Choose a minimizing sign j and a number α strictly between κ and 1. For any measurable D⊂{h=j} of positive Q mass, set
$$P_1=(1-\alpha)P_0+\alpha P_0(\cdot\mid D).$$
Use the raw score S1=S0 outside D and S1=S0·[1+α/((1−α)P0D)] on D. Its integral is Z0/(1−α). The alternative has mean m1=(1−α)μ0+αj, strictly past the relevant root of the base quadratic. In particular F0(m1)>0, with a positive margin independent of D. Couple both cold chains using their L=b+t proposal draws (counting X0) and their acceptance uniforms. Until a proposal enters D, their raw scores and therefore all acceptance decisions agree. Consequently |a1−a0| and |d1−d0| are at most 1−(1−QD)^L≤LQD. Thus G1≥F0(m1)−3LQD>0 when QD is sufficiently small. This is a genuine unsafe counterpart for the deployed MSE, including its start, target center and target variance.

Let w=Pr0(issue). For any J, obtain J disjoint positive-Q cells Dk within the selected label. Under the base protocol, a single query enters at most one cell, so the sum of their ever-visited probabilities is ≤p+q. The sum of their Q masses is ≤1. Choose a cell with QD+Pr0(any probe/sample encounters D)≤(1+p+q)/J. This simple averaging applies to data-dependent named probes and random sampling, and does not require that names be secret or that the base protocol ignore exact coordinates. Under the alternative, on any trajectory avoiding D the raw scores on named queries are identical, Q rows have likelihood multiplier 1, and every P row contributes exactly 1−α. For every no-hit path the alternative-to-base subprobability likelihood ratio is therefore (1−α)^nP≥(1−α)^p. This chain-rule statement applies to randomized early stopping and follows by conditioning on the common private random seed, then iterating the observation kernels. For J sufficiently large the alternative is unsafe, so

$$\delta\ge\Pr_1(\text{issue})\ge(1-\alpha)^p [ w-(1+p+q)/J].$$

Pass to J→∞ for this fixed finite q, then α↓κ. This proves the ceiling. The adversarial targets may vary with q. Each has finite positive raw scores and a finite integral, but a fixed public upper bound on a higher density-ratio moment generally excludes the arbitrarily narrow spikes used in this argument.

## Proof of attainability by conditional source assignment

Reserve an increasing independent Q-only bank for estimating the two path moments by iid whole cold trajectories, using a number of draws that is a fixed fraction of q. Estimate a lower bound aL and upper bound dU, paid at combined error γq→0 with log(1/γq)=o(q) and 0<γq<δ. Hoeffding with b,t fixed suffices. With probability at least 1−γq, U(m)=dU−(2aL+1/2)m+1.5m²≥F(m) for every m∈[0,1]. A safe interval (l,v) strictly within (0,1) given by U(m)<0 is therefore valid. If there is no interval, abstain. When there is, a model with true G≥0 satisfies μ≤l or μ≥v on the valid path event. The root confidence estimates converge to the true roots at every strictly safe base because they are simple. Endpoints at 0 or 1 can be handled by abstention in the finite procedure; this does not modify the limiting result at strictly safe nonconstant base models.

Randomly permute the remaining q' iid Q rows and all p iid P rows together. Conditional on the pooled *ordered values after the shuffle*, with the p source positions concealed, the law of the unknown subset I of P positions is proportional to ∏_(i in I) S_i among all subsets of size p. To see this, take Q product measure as the dominating measure, use the uniform prior on positions, multiply densities and cancel Z^(−p). This proof, with positions rather than unique state names, includes ties. No shuffle is needed computationally for the following symmetric statistics. For e_p the elementary symmetric polynomial (e_0=1), set
$$B_+=\frac{e_p(S_1h_1,\ldots,S_nh_n)}{e_p(S_1,\ldots,S_n)},\qquad
 B_-=\frac{e_p(S_1(1-h_1),\ldots,S_n(1-h_n))}{e_p(S_1,\ldots,S_n)}.$$
These are conditional all-one and all-zero probabilities of the P bank. Thus EB+=μ^p and EB−=(1−μ)^p, exactly for every finite p,q'. The path-estimation bank is independent of these statistics. Given path roots l,v, define
$$E=\min\{B_+/l^p,\ B_-/(1-v)^p\}.$$
Whenever the model is unsafe and the path event valid, one of the fractions has conditional expectation ≤1 (conditioning on the independent path bank). Therefore E does too, even though that controlling fraction need not be known and the two fractions may be dependent. Using an independent U∼Unif(0,1), issue with probability min(1,(δ−γq)E). The error incurred on a valid path event is ≤δ−γq; the exceptional event contributes ≤γq. The fixed protocol is uniformly honest. Set no-h=(p=0) interpretation B±=1. A deterministic rule E≥1/(δ−γq) has the same limiting full-power threshold when the threshold is exceeded strictly, but it need not attain the randomized envelope in the interior.

Let p be fixed and q'→∞. For denominator and each numerator, e_p of iid Q observations divided by binom(q',p) obeys the strong law for integrable fixed-order U-statistics; integrability follows from Q S<∞ and independent factors. Equivalently, for nonnegative integrable independent Q readings Yi, (Σ Yi/q')^p→(QY)^p; terms with repeated indices in the expansion are at most binom(p,2)[max_i(Yi)/q'](Σ Yi/q')^(p−1)→0 almost surely (the maximum ratio tends to zero by integrability and Borel–Cantelli). The finitely many P observations have finite scores almost surely and add an asymptotically negligible number of weighted terms: the contributions using j≥1 of the finitely many P positions are constant multiples of e_(p−j) of the Q rows and are o(q'^p). No finite variance of raw scores under P is required. Thus B+→μ0^p and B−→(1−μ0)^p in probability (almost surely under nested Q banks). On the clean model the randomized probability converges to min(1,δ min{(μ0/ℓ)^p,[(1−μ0)/(1−u)]^p})=Π_p. Bounded convergence proves the claimed power limit.

The upper and lower thus meet on a *bounded number* of independent targets and unbounded finite proposal scoring. They do not exchange the order of two statistical guarantees: with a known moment envelope, finite-dimensional parametric raw family, known target normalization, or a known finite support accessible to exhaustive enumeration, different information is available. In particular, imposing a known global upper bound on a higher moment can enable a Q-only uniformly honest estimator over that smaller class. The distribution-free pointwise limit derived here does not contradict a two-large-budgets minimax theorem with such an envelope.

## Finite check and limits of its interpretation

For a fixed visible rare set D and two known raw alternatives agreeing outside D and distinct on D, the ideal simple-hypothesis ROC has a separate literal finite sample formula. For iid P/Q draws at fixed quotas let ei=(1−QD)^q(1−PiD)^p. Conditional on no hit the two transcript distributions are identical when P1(.|Dᶜ)=P0(.|Dᶜ). Any hit reveals the world if its expected raw score under both is known. The best clean power is 1−e0+e0 min(1,δ/e1), by issuing on uniquely clean observations and randomizing on the no-hit event. This formula for two known worlds is diagnostic only, and named probes directly targeted at known D invalidate the no-hit interpretation.

CPU code in verify_local.py checks the actual finite-window risk of 12 rational three-state perturbations with varying rare Q mass. In 16 GUM tasks, it enumerates full finite-state risks after hiding a contamination of size κ±0.001 on the rarest available state of the critical label. All 32 re-evaluated cold risks straddle the half-variance threshold in the predicted direction. The full finite-state recursion in verify_local.py runs in extended precision. An independent check in verify_interval.py encloses the unchanged model path moments and hidden-state probability by the prior 140-bit Arb exponentials and outward binary64 sums, then encloses the perturbed margins with rational arithmetic and the 3LQD coupling bound. All 32 signs are thereby verified; the maximum enclosure width is 2.44e-8. This verification applies to the stored binary64 potentials and the specified perturbations. No scores or true roots from the evaluation enter the auditor. GUM supports of 4,913 or 83,521 states cannot satisfy the infinite-refinement assumption. The limiting ceilings on these rows describe the formula, or an analytically specified nonatomic enlargement in which a state is accompanied by independent ancillary U~Uniform[0,1] and the unknown score may depend on both. This enlargement preserves all clean GUM risks and clean score distributions. The finite-pair check does not claim any impossibility for the actual HMM parameter family. Source-only inputs at p=32 and real clean availability on the original 16 tasks are evaluated separately.


---

## Part II: Equality-cap obstruction

# Exact pointwise-power verification and cap equality: an unsafe adjacent family

The binary fixed-p attainability in LOCAL-POWER.md is valid. Fix exactly p independent scored P draws for the algorithm; for an adaptive quota ≤p, the lower still holds since (1−α)^{nP}≥(1−α)^p. Conditional on pooled ordered positions shuffled independent of all rows, product S weights yield the source subset law; canceling Z^{-p} uses exactly fixed p in the conditional experiment. An independently collected proposal-path bank returns roots l,v such that either μ≤l or μ≥v on the unsafe valid path event. Conditional on this bank, E B+=μ^p and E B−=(1−μ)^p, so either fraction has expectation at most 1; their minimum is pointwise at most that fraction. This uses no across-sides independence and requires no division of the declared error by two. Allocate enough independent proposal paths that failure probability γ_q→0 while path concentration vanishes. The fixed-order symmetric sums require only integrability of Q S, not a P moment or Q S². The capped randomized probability tends pointwise to the asserted envelope. It is not a finite-q attainment claim, not conditional-on-issue posterior safety, and not an exact no-randomization statement.

## Boundary Z0/B=1−κ can retain the ceiling

Take nonatomic Q (uniform on [0,1] suffices), fair binary readout h, S0≡1 (so P0=Q, Z0=1), b=0,t=4. The cold average is a mean of four iid fair bits, so F0(m)=3/2(m−.5)^2−1/16. Its lower root is ℓ=.5−1/√24, upper root u=1−ℓ, and κ=1/√6. Let public cap B=1/(1−κ), exactly the previously unresolved equality. For any positive-Q region D⊂{h=0} of arbitrarily small Q mass x, set
```
S1=S0 off D,  S1=1+κ/[(1−κ)x] on D.
```
This is a finite positive score with exactly Z1=B. Its target mean is m1=(1−κ)/2=ℓ, on the dangerous frozen root. Its *actual* risk is nevertheless strictly unsafe for all sufficiently small x.

Couple the base and alternative using the four proposal names and acceptance uniforms. If there is no proposal in D the paths agree; if there are two or more hits the probability is O(x²). Conditional on a unique D proposal at time i, the spike chain takes that proposal (base scores outside all 1), and its chance of subsequently leaving by time 3 is O(x), so those paths contribute only O(x²) in total. On the remaining first-order event its future marks are all zero. In the base, the conditioned previous and future non-D proposal marks are independent Bernoulli .5/(1−x), which can be replaced by independent fair marks at O(x) conditional error. If X is the sum of i preceding fair bits and Y is the sum of s=3−i following fair bits, the change in squared error centered at ℓ is
```
E[(X/4−ℓ)²−((X+Y)/4−ℓ)²] = (s/4)(ℓ−1/4−i/16).
```
Exactly one D at each possible i has probability x+O(x²). The variance shift is already incorporated by evaluating both path MSEs at ℓ and subtracting the same ℓ(1−ℓ)/2. Therefore
```
G1(x)−F0(ℓ) = (3/2)(ℓ−7/24)x + O(x²).
```
This coefficient is strictly positive: ℓ>7/24 iff 1/√24<5/24, and squaring gives 24<25. Hence each sufficiently small D is a genuinely unsafe alternative in the cap-constrained model, not only a frozen-moment tie. The original adaptive no-hit argument now uses α=κ exactly and gives the same pointwise ceiling at this base and public cap; the original unrestricted honest procedure supplies attainability. In particular, there cannot be a theorem of universal limit-one Q-only power at equality in this class. This example does not classify all equality bases. For general t≥3 in the same setup, the first-order coefficient is (t−1)/2×[ℓ_t−(1/3−1/(6t))], with the opposite sign for t≥5. Decreasing margins for one chosen family do not prove limit-one power, since other near-invisible alternatives may remain possible.

Reproduce the first-order check: python cap_equality.py uses 70-digit matrix arithmetic on the complete three-state IMH and reports the derivative quotients.

## All fair iid horizons t≥3 via a two-cell perturbation

On the same uniform base, fix any t≥3 with no burn-in and let ℓ_t be the lower root (strict safety holds). At equality, choose disjoint cells E,D⊂{h=0} with respective Q masses y and x=o(y). Change E's score to y² and put its released raw mass into D in addition to the entire permitted cap increase. Then Z=B and the unnormalized mass of label one is unchanged, so μ_1=ℓ_t exactly. The spike contributes only O(x) to the path-moment change. To first order in y, a unique E proposal at time i≥1 is rejected instead of accepting the zero mark, leaving the preceding fair mark in its place. If that preceding mark is zero, no change; if it is one (probability 1/2 to leading order), the average rises by 1/t from a baseline containing the previous one and t−2 other iid fair bits. In this event the conditional squared-loss increase at center ℓ_t is [(t+1)/t−2ℓ_t]/t, positive. A unique E at i=0 does not alter the path: the initial mark is still zero and either chain accepts the first subsequent ordinary proposal. Double hits and accidental acceptance of E from normal states have error o(y). Thus G_1(y)=((t−1)/(2t))[(t+1)/t−2ℓ_t]y+o(y)>0. Since Q is uniform, arbitrarily many disjoint such two-cell pairs can be selected, with x=o(y) uniformly and both masses decreasing, and the same no-hit argument applies off E∪D. This strengthens the iid equality counterexample, without claiming a classification of non-iid bases.


The local numerical check is [cap_equality.py](../../experiments/finite_state/cap_equality.py).