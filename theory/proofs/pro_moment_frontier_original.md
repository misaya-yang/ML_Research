# A nonlinear two-source frontier for a cold IMH risk decision

## Statement and information contract

Let equivalent probability laws Q and P have normalized density ratio W=dP/dQ. Independent samples from either law carry a public measurable observable $0\le h\le1$ and a raw positive score S=ZW, for the same arbitrary unknown constant Z. A P call obtains one independent P state; processing its score costs at most one auxiliary score if those are billed separately. A Q call obtains one scored Q state; a named score query may be substituted for it. Auxiliary computations and private randomness are free. For the upper bound there is no named-query requirement. For the lower bound even public uniform Q, public h, exact normalized scores and adaptive named queries are allowed. Supplying h on every public name does not reveal its hidden weight.

Fix b≥0, t≥4, L=b+t and A=t⁻¹ sum of h on states X_b,…,X_(b+t−1), where X₀~Q and the subsequent independent-Metropolis chain uses proposal Q. Its acceptance probability is min(1,S(y)/S(x)). Put $\mu=\mathbb E_P h$, $\nu=\mathbb E_P h^2$ and

$$
G=\mathbb E[A^2]-2\mu\mathbb E[A]+\tfrac32\mu^2-\tfrac12\nu=R_Q(b,t)-\tfrac12\operatorname{Var}_P(h).
$$

For a binary observable, $\nu=\mu$. For a bounded real observable the auditor uses the same sample labels to estimate both moments. The operational score zero is the actual half-variance threshold. To avoid vacuity under abstention, a correct auditor issues with probability at most δ on all G≥0, and at least 1−δ on all G≤−ε. One may instead require high accuracy on the two separated sides G≤−ε and G≥ε, which suffices for the lower bound.

Assume η=E_Q min(1,W)≥1/8 and E_Q W^(1+r)≤M, for fixed 0<r<1. For integer hard caps p,q, put K=q/p and require

$$
M\ge2(32L)^r,\qquad 0<M/K^r\le\tfrac12,\qquad 0<\varepsilon\le M/(2000K^r),\qquad 0<\delta\le\tfrac1{10}.
$$

In this nonempty resource band, a uniform audit exists whenever

$$
p^{1-r}q^r\ge C M\varepsilon^{-2}\log(1/\delta).
$$

Every uniform audit in this same class requires

$$
p^{1-r}q^r\ge c M\varepsilon^{-2}\log(1/\delta).
$$

Constants do not depend on L, M, K, p, q, ε, δ. Fixed per-sample raw-score billings can change q by O(p), a constant-factor difference here. Absolute margins G are essential: without a variance lower bound the theorem does not promise a fixed gap after division by Var(h). The lower examples have μ bounded away from 0 and 1, so also separate normalized decisions. The executable upper certificate controls false issuance for every equivalent P,Q with finite positive raw scores. Its power rate holds on the stated class. Its decisions do not require the auditor to know r, M, η, Z, or ε.

## An honest, scale-free upper certificate

Split off half the Q stream for the target mean and leave half for independent ordinary paths. Use a fixed fraction of the mean bank as an independent raw-scale pilot. Let $A_0$ be the pilot 31/32 empirical quantile of $S$, $q_c$ the number of remaining scored Q rows in the mean bank, $K_c=q_c/p$, and $H=K_c A_0$. Fixed floors only change numerical constants. The following steps use $p,q,\delta$ and the scores. They never use a moment bound, an overlap bound, or the true normalizer.

Set $D=H/Z$, $g=\min(1,H/S)$, $v=\min(1,S/H)$, and
$$
(u_1,u_0)=\mathbb E_Q(vh,v(1-h)),\qquad
(c_1,c_0)=\mathbb E_P((1-g)h,(1-g)(1-h)).
$$
The following formula estimates $\mu$. Replacing $h$ by $h^2$ everywhere in the four input means and in the definition of $\theta$ estimates $\nu$ on the identical sampled states. No sampling independence between these two invocations is assumed. Estimate both Q means on the same $q_c$ rows, and both P means on the same $p$ rows. Let $\theta=u_1/(u_1+u_0)$. Since $Wg=Dv$, $\mathbb E_P[g(h-\theta)]=0$, giving the exact identity
$$
\mu=F(\theta,c_1,c_0)=\theta+(1-\theta)c_1-\theta c_0.
$$

Give all eight means simultaneous confidence intervals (four if $h$ is known to be binary). For each bounded variable $X\in[0,1]$, an empirical-Bernstein interval of halfwidth
$$
\sqrt{2s_n^2\ell/n}+7\ell/[3(n-1)],\qquad \ell=\log(C_0/\delta),
$$
is valid by applying Maurer–Pontil (2009), Theorem 4, to $X$ and $1-X$. Here $s_n^2$ is the unbiased sample variance, and $C_0$ pays for all union bounds below. For $n<2$ use $[0,1]$. The optional intersections with binomial-KL bounds for bounded means and, when $h$ is known to be binary, an exact binomial interval for the direct P mean in the code divide their error budgets first. They can only shorten a simultaneous interval on the good event. Apply the monotone ratio endpoints to the head rectangle, using $[0,1]$ if necessary, to bound $\theta$ by $[t_-,t_+]$. Use $F(t_-,c_{1,-},c_{0,+})$ and $F(t_+,c_{1,+},c_{0,-})$ for the outer bounds of $\mu$, truncating to $[0,1]$. These bounds follow first by monotonicity in $\theta$ at the *true* residuals ($c_1+c_0\le1$), then by separate monotonicity in the residuals. This proof needs no independence between coordinates measured on the same rows. The identity is exact for every positive pilot, including on a bad pilot event.

On the path bank, simulate independent cold length-$L$ blocks (one scored Q initial state and $L-1$ scored proposals each). Denote the path averages by $A_i$ and put $a=\mathbb EA$, $d=\mathbb EA^2$. Compute a lower Hoeffding bound $a_-$ using the $A_i$ and an upper bound $d_+$ using the $A_i^2$; allocate error before looking. For the resulting mean interval $[m_-,m_+]$ and second-moment lower bound $v_-$, issue precisely when
$$
\max_{m\in\{m_-,m_+\}}\{d_+-2ma_-+\tfrac32m^2-\tfrac12v_-\}<0 .
$$
If $h$ is known to be binary, replace $v_-$ by $m$ at each endpoint, using the exact identity $\nu=\mu$. The tested polynomial is convex on $[0,1]$. On simultaneous coverage this upper-bounds $G$ even when the center interval has positive width. This proves distribution-free false-issue control for the fixed task, including outside the overlap and moment class. Additional windows or arbitrary looks need their own simultaneous protection.

We prove power using the envelope only in the analysis. Since $\mathbb E_Q\min(1,W)\ge1/8$, $Q(W\ge1/16)\ge1/15$. Markov gives $Q(W>64)\le1/64$. Hoeffding for the two indicator proportions with tolerance $1/128$ makes them straddle the empirical quantile with failure at most $\delta/C_0$ for a sufficiently large constant times $\ell$ pilot size (order-statistic rounding is absorbed). Therefore $Z/16\le A_0\le64Z$. It follows that $D$ lies within constant multiples of $K=q/p$ and $D\ge1$, since $K\ge(2M)^{1/r}\ge4(32L)$. Further,
$$
u_1+u_0=\mathbb E_Q\min(W,D)/D\ge\eta/D,\quad
\mathbb E_Q v^2\le M D^{-1-r},\quad
\mathbb E_P(1-g)^2\le P(W>D)\le M D^{-r}.
$$
The second bound follows by pointwise comparison of $\min(W,D)^2$ with $D^{1-r}W^{1+r}$; the last uses $\mathbb E_P W^r=\mathbb E_Q W^{1+r}$.

For clarity the empirical widths concentrate at the claimed rate as well as covering their means. For $0\le X\le1$ and $v_2=\mathbb E X^2$, Bernstein applied to $X^2$ yields $\overline{X^2}\le 2v_2+C\ell/n$ except with probability $e^{-\ell}$, since $\operatorname{Var}(X^2)\le v_2$. Also $s_n^2\le n\overline{X^2}/(n-1)$. Thus the empirical interval width is $O(\sqrt{v_2\ell/n}+\ell/n)$. Let $e_u$ bound both head interval widths. If $2e_u\le (u_1+u_0)/2$, every point in the head rectangle has a positive denominator, and the ratio differs from its true value by at most $2(2e_u)/(u_1+u_0)$ (subtract numerators after multiplying by the true ratio). The lower bound $u_1+u_0\ge\eta/D$ makes the head interval width
$$
O\!\left(\sqrt{M K^{1-r}\ell/q}+K\ell/q\right).
$$
The sum of residual interval widths is $O(\sqrt{M\ell/(K^r p)}+\ell/p)$. Each hard cap is counted once. The map $F$ is Lipschitz in its three coordinates by the separate monotonicity argument above, with coordinate constants at most one at feasible residuals. The intervals for both target moments thus have width of order the sum of these head and residual terms. Using eight coordinates only increases the absolute union-bound constant.

Write $J_*=pK^r=p^{1-r}q^r$. If $J_*\ge C M\varepsilon^{-2}\ell$, the two squared-root terms are both $O(\varepsilon/\sqrt C)$, and $K\ell/q=\ell/p\le\varepsilon^2 K^r/(CM)\le\varepsilon/(2000C)$. Enlarging $C$ makes the desired head denominator condition hold as well. For the path bank both one-sided bound slacks are $O(\sqrt{L\ell/q})$ simultaneously by Hoeffding. This is also $O(\varepsilon/\sqrt C)$ because $M K^{1-r}\ge L$. Finally $|\partial G/\partial\mu|\le3$, $|\partial G/\partial\nu|=1/2$ and $|\partial G/\partial a|\le2$. The displayed upper certificate exceeds the true $G$ by strictly less than $\varepsilon$ on the simultaneous good event, of probability at least $1-\delta$. Thus all $G\le-\varepsilon$ instances sign. The same estimator works for every $r,M$ in the band without knowing either. Constants can be chosen uniform even as $r$ tends to zero, although the band then requires a correspondingly large $K$. The real-arithmetic construction is implemented in `fixed_gate.py` with binary64 operations, a small outward numerical guard and SciPy confidence endpoints. The prototype has finite numerical resolution and is not a directed-rounding verification of the asymptotic theorem. The panel also examines a different finite-grid betting gate using the same center module; the finite-grid gate is not used to prove this small-margin guarantee.

### Exact executable for rational scores and observables

The algorithm in `rational_gate.py` takes exact positive rational raw scores, rational observable values in $[0,1]$, a rational failure probability, and independent source rows. The auditor can declare a public binary-observable promise to use the four-coordinate gate. In the general case it uses eight bounded-mean intervals on the same rows. The clipping quantile, both ratios, empirical variances, confidence endpoints and the final strict sign comparison are rational computations. Integer squared-root rounding returns the smallest multiple of $1/n$ above the true square root by applying integer square root to the squared numerator scaled by $n^2$, then checking the exact squared inequality. Thus each radius has at most $1/n$ additional slack. This is absorbed by the existing $O(1/n)$ residual term (and by $O(K/q)$ after head normalization); the path rounding is $O(1/n_{path})$. The integer bit length of the ceiling of $128/\delta$ upper-bounds $\log(128/\delta)$ within a universal constant factor, so no floating point logarithm is needed. Exact acceptance when the score ratio is $i/j<1$ draws an unbiased uniform integer in $[0,j)$ and accepts exactly the first $i$ outcomes. The default system random generator uses rejection to draw uniform bounded integers. The sole idealization in the probability proof is access to independent fair random bits and the stated independent source samples. Confidence validity and the upper rate have no fixed numerical gap floor. Bit complexity can grow with rational input denominators; query complexity counts score/sample calls.

Run `python dual_round/math/rational_gate.py` for three seeded smoke tests. At 2,048 P rows and 65,536 Q rows in each, the exact half-variance margin is $-3/32$ for the binary i.i.d. positive control, $36595109707840193/51018336000000000$ for the binary cold-start unsafe control, and $-3/128$ for the general bounded-observable control. The exact rational gate signs the first and abstains on both of the latter. Full rational endpoints and sample counts are in `rational-smoke.log`. These are executable checks, separate from the preregistered transfer panel.

## Lower bound: the certified threshold actually crosses

It suffices to exhibit hard binary observables within the full bounded-observable class. Here $\nu=\mu$ throughout. Take rational mass m in [M/(16K^r), M/(8K^r)] and B₀=3/4, J=32L. The three levels low, bulk and hidden have normalized scores l,J,K. Their proposal masses are

$$
q_B=B_0/J,\quad q_H=m/K,\quad q_L=1-q_B-q_H,\quad l=(1-B_0-m)/q_L.
$$

We have 0<l<1, q_L>.99, m≤1/16, K≥J≥128. Their target masses are obtained by multiplying by the corresponding weight. The class constraints hold:

$$
\mathbb E_Q[W^{1+r}]\le 1+\tfrac34J^r+M/8\le M,\qquad\eta\ge q_Ll=1-B_0-m\ge \tfrac3{16}.
$$

Conditional on level B the h=1 probability is β; conditional on level H it is θ. On level L set u_L=(1/2−q_Bβ−q_Hθ)/q_L; thus Q(h=1)=1/2 throughout. We next pick a single common public β and vary only θ. The statistical observations may include the level, as well as their state names, h and score. For simplicity in the finite-table construction S=W (Z=1); the lower remains true with that normalization explicitly given.

Couple the entire chain to its underlying proposal sequence. If all L proposals, including the initial one, fall in the low level, every proposal is accepted. The ordinary window average then equals the mean of the corresponding t iid Bernoulli(1/2) proposal labels. This is a coupling to the *unconditional* iid proposal experiment; we never claim that labels conditional on all-low have mean 1/2. The discrepancy probability is ≤L(q_B+q_H)≤13/512<.0254. Since both squared centered averages belong to [0,1], the true G differs by at most 13/512 from

$$
\phi_t(\mu)=\tfrac32(\mu-\tfrac12)^2-\tfrac18+\tfrac1{4t}.
$$

Also |EA−1/2|≤13/512.

At θ=1/2, β=1/2 gives μ=.5 and φ_t≤−.0625. At β=1, μ=.5+(B₀−l q_B)/2≥.872, so φ_t>.08. Both endpoints are strictly on the appropriate sides even after the coupling error. Exact G depends continuously on β (finite-state path probabilities). Choose a β_* between them with G=0. On the entire interval .5≤μ≤.65 one has φ_t≤−.02875, so the root has μ>.65. Pick a rational β arbitrarily close to the root so that |G(β,.5)|<ε/4 and μ>.649; this permits a finite rational instance. True h frequencies at low exceed .48 and are less than .52 under all subsequent tilts. These choice steps use only the hard-family construction, never the auditor's input stream.

Compare θ=.5 with θ₁=.5±s, where s is rational and 8ε/m≤s≤9ε/m (≤.072). The target mean difference is

$$
\Delta\mu=m(1-l/K)\Delta\theta.
$$

Couple identical sequences of proposal levels and acceptance uniforms (acceptance depends only on levels). A label changes on any given proposal with probability q_H|Δθ|+q_L|Δu_L|=2q_H|Δθ|. Hence |ΔEA| and |ΔEA²| are each ≤2L q_H|Δθ|. Writing G₁−G₀ as

$$
\big[-2a_0+\tfrac32(\mu_1+\mu_0)-\tfrac12\big]\Delta\mu+\Delta d-2\mu_1\Delta a
$$

shows that the multiplier in brackets exceeds .38 (μ₀>.649, |Δμ|≤9ε≤.00225 and a₀≤.5254). Its signed contribution is thus ≥.38(1−1/128)m|Δθ| in the tilt's direction. The rest has absolute value ≤6Lq_H|Δθ|≤(6/32)m|Δθ|. The net change in the tilt's direction therefore exceeds m|Δθ|/6. The two tilted worlds fall on opposite sides of the genuine half-variance threshold with gaps >ε, since (8/6)ε−ε/4>ε. All probabilities can be rational with no zero-probability changes between the two worlds. The construction keeps the actual deployed target mean bounded away from both endpoints.

## Separate transcript information of the two oracles

In the ideal single-observation experiment the sampled type is (level,h). Let laws 0 and 1 correspond to the two tilts. All bulk cell masses are exactly identical. The hidden cells differ by 2q_H s each in Q, 2ms each in P. The low cells vary by 2q_Hs in Q and 2l q_Hs in P. Every hidden conditional sign probability lies between .4 and .6, and the two low sign probabilities are between .48 and .52. Thus D(U||V)≤sum_i(U_i−V_i)²/V_i gives

$$
D_{\rm KL}(Q_0\Vert Q_1)\le Cq_Hs^2,\qquad D_{\rm KL}(P_0\Vert P_1)\le Cms^2.
$$

Conditioning a fresh named query on a publicly chosen sign changes the Q bound by at most factor 2, since Q(h)=1/2 in both worlds and the same conditional cell estimates apply. This is a finite-sample two-source KL calculation, with the signed risk separation proved above.

To guard against informative names, make Q uniform on an arbitrarily large even population N with publicly known half-0 and half-1 names. In each sign class assign the exact prescribed counts for its three levels by an independent uniform permutation. A named score reveals its level. Use N a common multiple of all cell denominators. The ideal without-replacement correction can be made arbitrarily small as follows. Cache disclosed names. In an ideal transcript a fresh named query generates its level independently drawn from the Q conditional level law for the chosen sign. A request for an independent P or Q state draws its (level,h) from the corresponding joint law, then attaches a uniformly random unused name in that sign. Requests at cached names return their cached response. Names given their sign contribute no KL.

After any real history, conditioning on the revealed assignments leaves a uniform permutation on remaining names within each sign: probabilities of the history depend on the hidden assignment only through already encountered weights. A fresh named call's color probabilities change from their ideal values by at most O((p+q)/(N/2−p−q)) in TV. A fresh sample from either oracle hits a disclosed name with probability at most (p+q)K/N. Given no repeat, its joint type law differs from the ideal by at most O((p+q)K/(N−(p+q)K)), by removing already revealed mass and renormalizing; the new name remains conditionally uniform in its public sign class. Sequential coupling thus bounds each real-to-ideal transcript TV by C(p+q)²K/N. Pick N so large this is ≤δ/4. This handles adaptive quotas, arbitrary named probes and early stopping, with only their separate upper caps charged.

Conditioning step by step on the algorithm's private seed and its query choices, the KL chain rule for the ideal transcripts gives total KL at most C(p m+q m/K)s². A uniformly valid, powerful auditor distinguishes the two priors with each ideal error ≤5δ/4 after the TV correction. Binary data processing gives KL≥kl(1−5δ/4,5δ/4)≥c'log(1/δ) for δ≤.1. Substituting s≤9ε/m, K=q/p and m≥M/(16K^r) yields

$$
C'\frac{\varepsilon^2}{M}\,pK^r\ge c'\log(1/\delta).
$$

This proves the claimed necessary condition. Large finite support, possibly depending on the budgets, is part of the minimax model. A public vector of target probabilities or a signed list of every rare state is a different oracle.

## The split carries information that the total loses

Fix r=1/2, t≥4, b fixed, M≥2(32L)^r fixed, and δ=.05. Take N=floor(ε^(−5/2)) total names. With p=floor(ε^(−5/4)), q=N−p, the geometric information sqrt(pq) is asymptotic to ε^(−15/8), too small. With p=floor(ε^(−7/4)), q=N−p it is asymptotic to ε^(−17/8), sufficient. Both remain in the stipulated band as ε tends to zero. Thus a total-count rate cannot classify these two allocations.

For the whole same class the P-only endpoint is Ω(ε⁻²log(1/δ)), by the same tilt family at a fixed m>0. For the Q-only endpoint choose m between 100ε and 110ε and K of order (M/ε)^(1/r), with mK^r≤M/8, in the construction (the construction permits any sufficiently large rational K here). Its Q KL is O(ε²/(mK)), giving Ω(M^(1/r) ε^(−(1+r)/r)log(1/δ)). Thus the successful mixed split just exhibited also uses fewer rows of each type than its single-source worst-case requirement. This compares uniform performance over a class, not a claim that samples from the two sources exhibit multiplicative synergistic information on one fixed binary hypothesis pair: KL on each fixed pair adds.

## What is newly needed, and what this theorem cannot diagnose alone

Clipping large weights, deterministic-mixture weighting and estimating an unknown scale by bridges have long precedents. The moment upper for a *static* mean is not the novelty claim, and another multiple-importance-sampling estimator can have the same order. The new part is a matched split-oracle statement for an actual finite-window cold-chain threshold: even public weight levels, public h, a known normalizer and arbitrary named score probes do not beat the frontier when the assignment of weights to state names is private, because rapidly changing hidden labels perturb the true target center with little visible change to the cold paths; the dependence on burn, retained window and functional is accounted for before the information argument. For these binary instances, if μ is supplied for free, ordinary Q trajectories already estimate G with O(L ε⁻²log(1/δ)) Q, so this lower does not survive a donated target mean. This cleanly identifies the mechanism tested, and does not imply that all failures of an exact-jump auditor have that cause. It gives no data-derived bound on M, unrestricted real-sequence power prediction or uniform sharp frontier at long horizons L≫M^(1/r).

**What an empty tail in the observed bank tells us.** Here is an exact fixed-pair obstruction for iid scored P/Q streams. Use the preceding bulk construction with $K\ge512L$ and split the hidden set evenly between the two public signs. At the central instance both hidden weights equal $K$, and choose the common bulk sign frequency so the central margin is zero (or arbitrarily close at rational precision). In the other two worlds change only the hidden weights to $K[1\pm(2h-1)/2]$. Both normalizations remain identical because the two hidden Q masses coincide. The complement, its point scores, and its target masses agree exactly; the hidden Q and P masses stay $m/K$ and $m$. Consequently the transcript laws agree exactly conditional on no hidden row, an event of the same probability $(1-m)^p(1-m/K)^q$ in both worlds for iid source streams at fixed caps. The two error probabilities of any binary test sum to at least that probability, counting failure to certify the safe world as its error. The cold margins actually split: each mean moves by $\pm m/4$. Relative to the central instance each cold path agrees until some hidden proposal occurs, so $|\Delta d|+2|\Delta a|\le 3Lm/K$. At the central root $\mu>.649$, $a\le.526$ and $m\le1/16$, hence the coefficient in the preceding exact difference formula exceeds .34. Each shift in the direction of its tilt exceeds $(.34/4-3/512)m>.07m$ before rational rounding. If preserving the same moment envelope, take $m\le M/(16K^r)$; the tilted cloud contributes at most $(3/2)^2mK^r$. Thus observing no extreme rows, by itself, cannot authorize a data-dependent switch to a uniformly stronger tail bound. This finite-pair illustration concerns iid streams; if informed of the hidden names an arbitrarily targeted named probe can distinguish it. The randomized-name lower theorem above grants named probes explicitly.
