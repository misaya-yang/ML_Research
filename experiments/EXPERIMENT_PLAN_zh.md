# Experiment Plan (for the Experiment Agent)

> **2026-11-23 execution-priority update:** See [Research Question and Main Direction](../RESEARCH_QUESTION.md) for the sole current research direction and its three gates. The cold-start IMH audit plan below is from 2026-11-22; its original criteria, results, and resource rules are retained for reference. Legacy experiments that conflict with the main direction are not launched automatically in this round. Freeze and schedule a held-out experiment only after the value of the same real, unchanged policy and the fully costed mechanism have both cleared their gates.

Claude drafted this plan from the theory notes on 2026-11-22. Each experiment maps to a theoretical prediction. The code is pseudocode only; the experiment agent implements it. Follow the operational rules in [EXPERIMENTS_README.md](../EXPERIMENTS_README.md); this file only supplements them with the experimental details.

Theory references:
- [Theorem A: Tail-Profile Frontier](../theory/proofs/tail_profile_frontier.md)
- [Fixed-p Extensions](../theory/proofs/fixed_target_extensions.md)
- [Three-Channel Overview](../theory/proofs/three_channels.md)
- [Pro's Moment Frontier](../theory/proofs/pro_moment_frontier_original.md)
- [Original Fixed-p Theorem](../theory/proofs/fixed_target_power.md)

## 0. Rules

1. **Freeze before running.** Write a `config.json` for each experiment with its parameter grid, seeds, and success criteria, and record the file's SHA-256. Do not change the grid or criteria after freezing. If a change is necessary, start a separate run and explain the reason in the summary.
2. **Output location.** Write outputs only to `results/local_runs/<实验号>-<YYYYMMDD>-XXXX/`: `config.json`, `raw.csv`, `summary.md`, and `figs/`. The first placeholder is the experiment ID. Do not overwrite frozen directories under `results/`.
3. **Report results as observed.** Report failures, disagreements with theory, and anything that could not be completed. Do not tune parameters to pass a criterion. Do not claim that “the method is better.”
4. **Cost accounting.** Count target samples, proposal samples, and queries for scores at specified points separately. For LLM runs, also record the number of generated sequences, rejection sampling attempts, and GPU time.
5. **Local resources.** Run only one job at a time. Limit threads (OPENBLAS/OMP/MKL/torch) to ≤2, use `nice -n 19`, and leave ≥6GB of memory free.
6. **Kaggle.** Follow `~/kaggle_t4_template/README.md`: keep runs private, run SMOKE before full runs, and checkpoint every 10 minutes. Weekly allowance: 30h. Session limit: ≤9h. Total GPU budget for this plan: ≤12h.
7. Do not commit or push with git. Do not install new packages unless the user asks. The Kaggle image already includes torch/transformers.

## 1. Theoretical Predictions and Criteria

| ID | Prediction | Source | Experiment | Pass criterion (frozen) |
|---|---|---|---|---|
| P1 | Center-channel frontier p≍AΨ(q/p); the same auditor works for any Ψ | Theorem A | X1 | See X1 |
| P2 | The frontier bends at the tail's change point; slope −r/(1−r) | Theorem A, Corollary 3 | X1c | The 95% confidence intervals of the two fitted slopes cover −1/3 and −3, respectively |
| P3 | At fixed p, power is bounded above by δ/(1−κ)^p; the e_p procedure attains the bound | B1 | X2a, X2b | At the largest q, \|power−Π_p\| ≤ 3 MC standard errors |
| P4 | Pointwise ε⁻¹ versus uniform ε⁻² | B2 | X2d | The log-log slopes lie in [−1.15,−0.85] and [−2.3,−1.7], respectively |
| P5 | With known Z, power can approach 1 even at p=0; with unknown Z, power at p=0 is ≤δ | B3 | X2e, X4a | Known Z: power ≥0.9 at the largest q; unknown Z: power ≤δ+3SE at every q |
| P6 | Rate of approach to the ceiling: gap ≤C(M/q^r)^{1/(1+r)} | B4 | X2c | Fitted slope ≤ −r/(1+r)+0.1 (the theorem provides only an upper bound, so the observed slope can be steeper) |
| P7 | Construction holds for general k (center root μ_*=(k+1)^{−1/2}, calibrated three-layer family, bracketed term ≥c/√k) | Theorem A, B2 | X3 | The two worlds straddle ±ε for each k |
| P8 | Validity: false-certification rate ≤δ | All auditors | X6 | The 95% upper bound on the pooled false-certification rate is ≤δ |
| P9 | The center frontier is unchanged under small overlap (grid certificate); at very large q, the frontier saturates at ε⁻¹ | E1, E1′ (small_overlap_joint_frontier.md) | X1e | See X1e |
| P10 | Per-jump P cost ≈C·min{1/η₀, 1+D_{1/2}/(η₀t)}; cost drops sharply as the β sweep crosses D_{1/2}≈η₀t | E4 | X8 | The log-log slope of observed cost against 1+D_{1/2}/(η₀t) is in [0.8,1.2] before the 1/η₀ cap |
| P11 | Cold-start risk bound: R_Q(b,t)≤44Γ_t for every h (Γ_t=E_P min(W,t)/t); refined bound R≤1520·Ξ_{b,t} (Ξ=E_P min(W/t,1,L/W)+τ_P(L)²) | Lemma R, R′ (cold_risk_bound.md) | X9a | Hard criterion: no instance exceeds 44 or 1520. Also report the observed supremum (about 1.4 in local tests) |
| P12 | The Γ certificate is valid everywhere and does not run the chain; it issues a certificate when Var_P(h)/(kΓ_t) is sufficiently large; p for 80% power is proportional to k/Var | Corollary R2 | X9b, X9c | Count false certifications in X6 (P8); the log-log slope of minimum p against k/Var is in [0.8,1.2]. Fit only instances with Var/(kΓ_t)≥10³. The transition is expected between 10² and 10³ because C₀=44 and confidence bounds inflate it |
| P13 | Light-window dynamics lower-bound family: the two worlds have identical W distributions, h marginals, μ, and Var, and G_k equals ∓ε, respectively; G_k(θ) is exactly quadratic; optimal testing requires q ∝ ρt/ε² and p ∝ ρ/ε² | Theorem V (variance_scaled_floor.md) | X10a, X10b | Invariant differences ≤1e-12; \|G\|≥ε; quadratic residual ≤1e-10×the scale of \|G\|; log-log slopes of minimum q against ρ and t, and of minimum p against ρ, are all in [0.8,1.2] |
| P14 | Small-overlap lower bound: (i) with identical W distributions and tail class, as η=(2−ρ)/d→0 the two worlds share μ and Q(h=1) while G_k crosses the threshold; (ii) the slope a_U of the startup-layer label witness S matches its closed form; (iii) Proposition N's two-step inequality holds over random compensating label pairs; (iv) gain K grows linearly with t | “Small overlap” in variance_scaled_floor.md: Lemma F, Theorem V_η, Proposition N, Theorem S | X11a, X11b | Hard criteria: invariant differences ≤1e-12; relative difference between quadratic-form R and the explicit cell chain ≤1e-12; CS ratio ≤1; low-layer excess/e_b ≤1. Soft criteria (calibratable rows with ρ_B≥ρ/2 only): a_U/pred∈[0.85,1.15]; the two G∓/ε have opposite signs and absolute value ≥1; KL·d/Δ²∈[8.7,9.2] on both P and Q; at fixed (ρ,ρ_B/ρ), the log-log slope of K against t is in [0.8,1.2] |

## 2. Shared Components (Pseudocode)

### 2.1 Exact Path Moments for Finite-Support IMH

This applies to discrete families and LLM sample banks. Transitions depend only on the weights. Sort by w, then use prefix sums to compute one step in O(N) time.

```python
def exact_path_moments(w, qm, h, b, t):
    # w: normalized weights dP/dQ (a common scale factor is fine: acceptance depends only on ratios); qm: Q mass; h ∈ [0,1]
    o = argsort(w); w, qm, h = w[o], qm[o], h[o]
    alpha = Qmass(w_y >= w_x) + (1/w_x) * sum_{w_y < w_x} qm_y w_y      # use searchsorted for ties
    def K(g):    # (Kg)(x) = Σ_y qm_y min(1, w_y/w_x) g_y + (1-alpha_x) g_x
        return suffix_sum(qm*g)[first idx with w_y>=w_x] + prefix_sum(qm*w*g)[w_y<w_x]/w + (1-alpha)*g
    def KT(pi):  # (πK)(y) = qm_y Σ_x π_x min(1, w_y/w_x) + π_y (1-alpha_y)
        return qm*(prefix_sum(pi)[w_x<=w_y] + w*suffix_sum(pi/w)[w_x>w_y]) + pi*(1-alpha)
    pis = [qm];  for i in 1..b+t-1: pis.append(KT(pis[-1]))   # distribution of X_i, cold start X_0~Q
    gs  = [h];   for dd in 1..t-1:  gs.append(K(gs[-1]))      # E[h(X_{i+dd}) | X_i = x]
    Eh  = [pi @ h for pi in pis[b:b+t]]
    M   = (stack(pis[b:b+t]) * h) @ stack(gs).T               # M[i,dd] = E h(X_{b+i}) h(X_{b+i+dd})
    a   = mean(Eh)
    d   = (sum_i M[i,0] + 2*sum_{i, dd>=1, i+dd<t} M[i,dd]) / t**2
    P   = qm*w / sum(qm*w);  mu = P@h;  nu = P@(h*h)
    return a, d, mu, nu
def G_k(a, d, mu, nu, k): return d - 2*mu*a + mu**2 - (nu - mu**2)/k
```

Unit tests:
- For random instances with N≤2000, relative error against the dense matrix K is <1e-10.
- Against 10⁶ MC chains, discrepancy is <4 standard errors.
- Reproduce G/x→0.0063138 from `experiments/finite_state/cap_equality.py`.

### 2.2 Gate Polynomial and κ

```python
def roots_Fk(a, d, k):  # binary h: F_k(m) = d - (2a+1/k) m + (1+1/k) m^2; safety interval is between the roots
    A2, A1, A0 = 1+1/k, -(2*a+1/k), d
    disc = A1**2 - 4*A2*A0;  if disc <= 0: return None
    return ((-A1 - sqrt(disc))/(2*A2), (-A1 + sqrt(disc))/(2*A2))
def kappa(mu0, l, u): return min(1 - l/mu0 if l > 0 else 1, 1 - (1-u)/(1-mu0) if u < 1 else 1)
def Pi_p(delta, kap, p): return min(1, delta/(1-kap)**p)
```

### 2.3 Lossless Count Simulation

States in the discrete family are cells (weight layer × h); only the count in each cell needs to be recorded.

```python
nP = rng.multinomial(p, P_cells); nQ = rng.multinomial(q, Q_cells)   # q can reach 2^40
# Cold path means: use DP to obtain the exact distribution of the window sum Σ_{i=b}^{b+t-1} h(X_i); state is (cell, count), with size cells×(t+1)
pmf = window_sum_pmf(cells, b, t);  path_counts = rng.multinomial(n_paths, pmf)
```

### 2.4 The e_p Statistic

Use Newton's identities with count weights.

```python
def e_p_ratio(x, y, cnt, p):           # returns e_p(y)/e_p(x), with per-cell values x and y and per-cell counts cnt
    s = sum(cnt*x); x, y = x/s, y/s    # normalize first to avoid overflow
    def e(z):
        P = [sum(cnt*z**i) for i in 1..p]; E = [1]
        for j in 1..p: E.append(sum((-1)**(i-1)*E[j-i]*P[i-1] for i in 1..j)/j)
        return E[p]
    return e(y)/e(x)
# For small cases, verify against exact DP (fractions or mpmath, item-wise E[j] += z*E[j-1]); relative error <1e-9
```

### 2.5 Auditors

Every auditor returns a certification probability. Randomized auditors return min(1,δ'E); average that probability directly in summaries.

```python
def A_ep(Prows, Qrows, delta, k, gamma):              # procedure that attains the ceiling (B1)
    Qpool, Qpath = split_half(Qrows)
    a_lo, d_hi = hoeffding(path_avgs(Qpath), gamma)    # joint error budget gamma across both quantities
    r = roots_Fk(a_lo, d_hi, k);  if r is None: return 0
    l, v = r
    x = S[Prows ∪ Qpool];  Bp = e_p_ratio(x, x*h, ...);  Bm = e_p_ratio(x, x*(1-h), ...)
    E = min(Bp/l**p, Bm/(1-v)**p);  return min(1, (delta-gamma)*E)

def A_knownZ(Qrows, Z, delta, k):                     # p=0, known Z (B3a)
    Qmean, Qpath = split_half(Qrows); W = S[Qmean]/Z; c = len(Qmean)**0.25
    mu_L = mean(min(W*h, c)) - c*sqrt(log(8/delta)/(2q))
    mu_U = 1 - (mean(min(W*(1-h), c)) - c*sqrt(log(8/delta)/(2q)))
    a_lo, d_hi = hoeffding(path_avgs(Qpath), delta/2); r = roots_Fk(a_lo, d_hi, k)
    return 1 if r and r[0] < mu_L and mu_U < r[1] else 0

def A_clip(Prows, Qrows, delta, k):                   # Pro's clipping certificate; see pro_moment_frontier_original.md
    Qmean, Qpath = split_half(Qrows); pilot, rest = split(Qmean, frac=1/8)
    H = (len(rest)/p) * quantile(S[pilot], 31/32)
    v = min(1, S_rest/H); g = min(1, H/S_P)
    u1, u0 = mean(v*h), mean(v*(1-h)) over rest;  c1, c0 = mean((1-g)*h), mean((1-g)*(1-h)) over P
    # Use jointly valid empirical Bernstein intervals (Maurer–Pontil) for the four quantities, with error budget delta/C0
    th_lo, th_hi = u1_lo/(u1_lo+u0_hi), u1_hi/(u1_hi+u0_lo)
    F = lambda th, c1, c0: th + (1-th)*c1 - th*c0
    mu_lo, mu_hi = clip01(F(th_lo, c1_lo, c0_hi)), clip01(F(th_hi, c1_hi, c0_lo))
    a_lo, d_hi = hoeffding(path_avgs(Qpath))
    return int(max(d_hi - (2*a_lo + 1/k)*m + (1+1/k)*m**2 for m in (mu_lo, mu_hi)) < 0)

def A_replay(Prows, Qrows, delta, k):                 # baseline: Clopper–Pearson on P labels alone for the center; otherwise as above
    ...

def LR_oracle(cellsP0, cellsP1, cellsQ0, cellsQ1, p, q, delta, reps=20000):
    # With both worlds known, simulate log LR from counts; control type I error at delta in the unsafe world and report power in the safe world
```

## 3. Experiments

### X1 Center-Channel Frontier (local CPU, about 2–4h)

**Instances.** Theorem A's three-layer family:
- L=b+t, with (b,t)=(0,8) and k=2; J=128kL, K=ρ:=q/p, m=Ψ(K)/12.
- Binary h. Calibrate bulk probability β of h=1 by bisection so G_k=0 at θ=1/2 (compute exactly with Section 2.1).
- Two worlds θ±=1/2±s, with s=C·ε/m. Choose the smallest C for which the worlds straddle ±ε and record the observed value.
- Use low-layer u_L to compensate so Q(h=1)=1/2 remains fixed.

**Shapes of Ψ.**
- Power law: r∈{0.25,0.5,0.75}, M=D₀^r.
- Piecewise power law: r₁=0.25 before D_b and r₂=0.75 afterwards, with continuity at the join.

**Grid.** ρ∈2^{12..40}, exponent step 2; ε=Ψ(ρ)/2000; δ=0.05.
- Keep only points in the theorem's range: K=ρ≥D₀=2048 and Ψ(ρ)≤1/2.
- Move the breakpoint for the piecewise Ψ to D_b=2^20 so both segments contain ≥4 points.
- Counts may exceed int64. For cells with expected count >10^12, use the normal approximation N(nπ, nπ(1−π)). Do not run points with q>2^60; document them in the summary.

**X1a (lower-bound check).** For each ρ, use bisection to find the smallest p at which LR_oracle reaches power ≥0.95. Set y=p_min·ε²/log(1/δ). Criterion: the ratio of maximum to minimum y/Ψ(ρ) across all ρ and Ψ shapes is ≤4.

**X1b (upper bound, main result).** On the same instances, set p=c·Ψ(ρ)·A, c∈2^{0..14}, and find the smallest c*(ρ,Ψ) at which A_clip reaches power ≥0.95. Criterion: the maximum-to-minimum ratio of c* across all ρ and Ψ is ≤8. Also report the absolute size of c* (Pro observed large constants).

**X1c (breakpoint).** Plot log p_min versus log q from A_clip on the piecewise Ψ (fixed ρ grid, q=ρ·p_min). Fit the two slopes separately; see P2 for the criterion.

**X1d (adversary's K).** Take 3 values of ρ and set K/ρ∈2^{−4..4}. Check whether the p needed by LR peaks near K≈ρ. Report descriptively; no pass criterion.

**X1e (small overlap and saturation).**
- Implement A_clip_grid. Use the 1/4 quantile of independent P samples as anchor a. Compute A_clip's center interval at each H_j=a·(q/p)·2^{−j} (j=−3..⌈log₂(q/p)⌉+1), giving each interval error budget δ/(4J). Intersect the intervals, then apply the path test.
- Low-overlap variant: mix the three-layer proposal with a “dark layer” of Q mass 1−η₀. Give it a raw score of 10^{−6} times the minimum score in the original family; its target mass is negligible. Multiply the Q masses of the original layers by η₀. The normalized weights then increase by about 1/η₀, actual overlap is roughly η₀ times its original value, the tail profile becomes Ψ(η₀D), and the hidden layer is placed at K/η₀.
- Use η₀∈{2^{−8},2^{−12}}. The original pilot (31/32 Q quantile) will fall in the dark layer, so A_clip is expected to fail. Compare c* for A_clip and A_clip_grid. Criterion: A_clip_grid's c* is independent of η₀ (maximum-to-minimum ratio ≤4).
- Saturation: fix a power-law Ψ with r=0.5 and increase ρ until Ψ(ρ)<ε. Criterion: the p needed by A_clip_grid approaches a constant ×log(1/δ)/ε instead of falling with Ψ(ρ).

### X2 Fixed-p Ceiling, Convergence Rate, and Dichotomy (local CPU, about 2h)

**Instances.**
- I1: the fair coin from Part II, uniform Q, S≡1, t=4, b=0, k=2, κ=1/√6. Here B₊ is a hypergeometric ratio with a closed form.
- I2(κ): the three-layer family near a root, κ∈{0.02,0.05,0.1,0.2}. Adjust β so μ₀ is Θ(κ) away from the root, and record exact κ and G.
- I3(r): heavy-tailed family. Weight layers w_j=2^j, j=−6..30; Q mass ∝2^{−j(1+r+0.05)} for j≥0, with low layers used to enforce E_QW=1; r∈{0.5,1}. Vary the probability of h across layers so μ₀ lies in (0.3,0.7) and each instance is strictly safe.

**X2a.** Use the two-world exact ROC formula from Part I, 1−e₀+e₀·min(1,δ/e₁), where e_i=(1−QD)^q(1−P_iD)^p. As QD→0, plot convergence to δ/(1−α)^p. Evaluate the formula directly; no simulation.

**X2b.** On I1 and I2, use p∈{0,1,2,4,8,16} and q∈2^{8..30}. Use A_ep to estimate mean certification probability (2000 repetitions per cell) and compare to Π_p (P3).

**X2c.** On I3, record Π_p−Power_q as a function of q for p∈{2,8}, and fit the slope on log-log axes (P6).

**X2d.** Main figure for Theorem 2.
- Pointwise: on I2-type instances set G=−ε, ε∈{0.04,0.02,0.01,0.005,0.0025}. Compute p_min=min{p: δ/(1−κ)^p≥0.8} from the formula. Verify 3 points with A_ep at q=2^30.
- Uniform: use the center-floor family from the long-window note, α≪1/q and q=2^20. Use LR_oracle to find p_min(ε) for power ≥0.8.
- Plot both lines together on log-log axes (P4).

**X2e.** On I1 and I3 with p=0, A_knownZ's power rises with q; A_ep's power remains ≤δ (P5).

### X3 Verification for General k (local CPU, <1h)

- k∈{1.5,2,3,4,8}.
- Center floor: use ±1 signs. Compute the root of G_k=1/T+(1+1/k)μ²−1/k and compare it to μ_*=(k+1)^{−1/2} at T=2^16.
- Theorem A's three-layer family: use t=2k and J=128kL. Check that the calibration root exists and the two worlds straddle ±ε. Record the bracketed term across k and check whether it is ≳c/√k.
- See P7 for the criterion. Report any failures, since they directly affect the theorem statement.

### X0 Proposal-Head Screening (Kaggle, ≤1h)

Purpose: check that the LLM tasks are not solved directly by cheaply enumerating high-probability sequences. See the higher-order reward comparison in `results/disclosure/README.md`.

```python
model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb", torch_dtype=float16).cuda()   # π_ref = Q
rm    = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb").half().cuda()
prompts = first 12 IMDB test examples sorted by sha256(review_id), with first 8 gpt2 tokens as prefix  # freeze in config
for x in prompts:
    head = beam_search(x, num_beams=512, num_return_sequences=512, max_new_tokens=32)        # approximate top 512 sequences
    logq_head = seq_logprob(model, x, head); r_head = clip(rm_logit_diff(x+head), -4, 4)
    bank_small = sample(x, n=8192, top_k=0, top_p=1, temperature=1, max_new_tokens=32)
    Zhat_beta = mean(exp(r_bank/beta))                     # screening only
    head_Qmass = sum(exp(logq_head)); head_Pmass(beta) = sum(exp(logq_head + r_head/beta)) / Zhat_beta
    dup_rate = 1 - unique(bank_small)/8192
```

Pass condition: head_Pmass<0.5 and dup_rate<0.05 for β∈{0.5,1,2,4}. Drop prompts that fail. If fewer than 8 pass, stop and report.

### X4 Reward-Tilted LLM Sample Bank (Kaggle bank construction ≤8h; audit on local CPU)

**Build the bank** once X0 passes:
- For each prompt, draw N=2^18 sequences with 32 new tokens, top_k=0, by unmodified sampling.
- Save for each sequence: r=clip(logit_pos−logit_neg,−4,4) as float32, π_ref log probability, text hash (uint64), length, and h bits.
- Freeze h in the config, with at least 3 observables: h1=contains a negation word (freeze the list); h2=1{sigmoid(r)>0.95}; h3=1{contains "!"}.
- Checkpoint after each prompt.
- The bank defines Q for this surrogate problem (uniform on the bank). P∝exp(r/β) is exactly computable on the bank for evaluation only. Auditors may not read the full table; they may only draw paid samples within the budget.

**Freeze tasks.**
- Sweep β∈{0.5,1,2,4}, h, k∈{2,4}, and (b,t)∈{0,4,16,64}×{4,8,16,32,64,128}.
- Compute exact G_k/Var, κ, η, ω=min W, and the τ_P(D) curve using Section 2.1.
- Select tasks with 0.03≤|G_k|/Var≤0.3, split evenly between safe and unsafe, with at least 60 of each.
- Save as `tasks.json` and record its SHA-256. Begin audits only after that.

**Audit** (200 repetitions per task and setting):
- P samples: draw from the bank categorically with weights exp(r/β). Also simulate the number of rejection-sampling attempts from a geometric law with success probability Z·e^{−4/β}, and charge those attempts.
- Q samples: draw uniformly from the bank.
- X4a (P5): over q∈2^{10..18}, compare the minimum p for power 0.8 under A_ep (unknown Z) and A_knownZ (given exact Z). Expected: A_knownZ needs p=0 at large q; A_ep approaches ⌈log(0.8/δ)/(−log(1−κ))⌉.
- X4b (P1, descriptive): compare the shapes of the minimum-p-versus-q curves for A_clip and A_replay to Ψ_emp(q/p)·log(1/δ)/G², substituting τ_P measured on the bank. Report rank correlation; no pass threshold.
- X4c (P3): at q=2^18, plot A_ep's power for p∈{0,1,2,4,8,16,32} and compare to Π_p computed using the bank's κ.
- Count every certification in X6.

### X5 Transfer to Live Generation (optional; Kaggle ≤3h)

- Choose 2 prompts × β∈{1,2}.
- Do not use the sample bank. Draw Q from fresh generations and P from actual rejection sampling (proposal π_ref, acceptance probability exp((r−4)/β)).
- Use budgets p≤256 and q≤2^16. Use a separate large bank (2^18) for reference truth.
- Report whether audit decisions agree with those made on the bank, plus actual costs.

### X6 Validity Summary (local; accumulate alongside X1–X4)

- Pool certifications on all unsafe instances: at least 200 synthetic instances, and report LLM tasks separately.
- Report the false-certification rate and Clopper–Pearson 95% upper bound separately for each auditor. See P8 for the criterion.

### X8 Tail-Dependent P Jump Cost (local CPU, <1h; end of Stage 1)

- Start with the existing simulation `experiments/finite_state/jump_cost.py` (number of tests for a single lag k). Extend it to a full two-time replicate: draw i and j uniformly from [b,b+t), then add the number of tests for the two jumps.
- Target law: Pareto (scale D₁=1/η₀, exponent r) plus low-weight mass η₀/2. Grid: η₀∈{2^{−8},2^{−10}}, r∈{0.25,0.5}, t∈2^{8..20}; 2000 repetitions per cell.
- Record the mean test count per replicate and x=D_{1/2}/(η₀t), and plot on log axes (P10).
- If X4 has built its bank, also compute D_{1/2}, η, and mean cost on the bank for each β. Check whether the β sweep crosses D_{1/2}≈η₀t. Report descriptively; no pass criterion.

### X9 Cold-Start Risk Bound and Γ Certificate (local CPU; synthetic portion <1h, bank portion in Stage 3)

Starting point: `experiments/finite_state/risk_bound.py` and `risk_bound_refined.py` (already run locally; logs in `results/local_runs/theory_checks-20261122/`).

**X9a (P11, exact verification).**
- Synthetic instances: reuse both scripts' generators, plus X1's three-layer family and X8's Pareto family with a low-weight layer.
- t∈2^{3..12}, b∈{0,t/4,4t}. Use threshold indicators h=1{W≥v} or 1{W≤v}, with v across 12 quantiles of W, and additional random h.
- For each (instance,b,t), record max_h R/Γ_t, max_h R/Ξ_{b,t}, and Γ_t/Ξ_{b,t}.
- If the X4 bank is ready, also compute R, Γ_t, Ξ for every β, frozen h, and (b,t) using exact path moments from Section 2.1, and record the same three ratios.
- Output `x9a.csv` and a scatter plot of R against Γ_t on log axes, with reference lines y=44x and y=x.

**X9b, X9c (P12, Γ certificate).** Pseudocode:

```python
def A_gamma(Prows, Qrows, t, k, delta, C0=44):        # Corollary R2; three independent banks, each allocated delta/3
    P0, P1, P2 = split(Prows, [ceil(32*log(6/delta)), half, half])
    c  = quantile(S[P0], 5/8)                          # estimate an upper-side value for Z·d
    ZL = c * eb_lcb(minimum(S[Qrows], c)/c, delta/3)   # Maurer–Pontil lower bound; E_Q min(S,c)<=Z always holds
    if ZL <= 0: return 0
    UG = eb_ucb(minimum(S[P1]/ZL, t)/t, delta/3)       # upper bound on Γ_t (monotone in Z)
    VL = var_lcb(h[P2], delta/3)                        # paired U-statistic (h_i-h_j)^2/2 with Bernstein
    return int(C0*UG < VL/k)
```

- Instances: synthetic families (lognormal σ∈{1,2,3}; Pareto r∈{0.5,1,1.5} plus low-weight mass η₀/2), t∈2^{4..12}, k∈{2,4}. Choose threshold indicators h so Var/(kΓ_t) covers [1,10^4].
- Budget grid: p∈2^{6..16}, q∈2^{8..18}; 200 repetitions per cell, δ=0.05.
- X9b: plot certification probability against x=Var_P(h)/(kΓ_t), pooling instances. Judge every certificate using exact G_k and count it in X6. Expected false-certification count: 0.
- X9c: at the largest q, plot the minimum p needed for an 80% certification rate against k·log(1/δ)/Var and fit the log-log slope (P12), using only instances with x≥10³. Report the x at which the certification rate first exceeds 50%.
- Baseline: on the same instances, audit the threshold with A_clip and repeated cold paths. Compare total draw counts (P draws + Q draws, with L per path). Report descriptively; no pass criterion.
- Exploratory variant: use C0=2, the empirical constant. Explicitly label it “not guaranteed valid by theory.” Use it only to illustrate the cost of the constants, not for X6 validity conclusions.
- If the X4 bank is ready, run A_gamma on the frozen tasks at each β. Report how many can be certified with p≤2^12 and their G_k/Var distribution.

### X10 Variance-Scaled Dynamics Lower-Bound Family (local CPU, <1h; end of Stage 1)

Starting point: `experiments/finite_state/variance_floor.py` (already run locally; log at `results/local_runs/theory_checks-20261122/variance_floor.log`). See variance_scaled_floor.md for the family: 8 cells (B±, D±, M₁±, M₂±), with the worlds differing only in θ.

**X10a (P13, exact verification on an expanded grid).**
- k∈{1.5,2,3,4,8}; β∈{0,1,2,4}; (c₁,c₂)∈{(1,4),(0.5,2),(3,10)}. Keep only combinations satisfying condition (K), σ=(kΦ̄−1)/2−0.0111>0.
- t∈2^{10..16}, ρ∈{0.005,0.01,0.02,0.05}, ε=ρ|Φ₁−Φ₂|/16 (tilt Δ=1/4).
- For each combination record the differences in the worlds' invariants (W distributions under P and Q, P(h=1), Q(h=1)); G(θ*∓Δ)/ε; the quadratic fit residual for G(θ); R/ρ against Φ̄/2; and the two KL constants.

**X10b (P13, sample-size scaling for the optimal test).**
- Single-draw distributions in both worlds are known: both P and Q draws fall in 8 cells. The likelihood-ratio test therefore depends only on cell counts; simulate the multinomial directly.
- On the (ρ,t) grid, find the minimum q (with p=0) and minimum p (with q=0) for which both error rates are ≤δ=0.05, using 20000 repetitions per point. Fix ε=ρ_min|Φ₁−Φ₂|/16. Recompute θ* and Δ=4ε/(ρ|Φ₁−Φ₂|) whenever ρ changes.
- Prediction: minimum q ∝ ρt/ε² and minimum p ∝ ρ/ε².

```python
def min_budget(cellsA, cellsB, delta, which, reps=20000):   # which='P' or 'Q'
    llr = log(cellsA/cellsB)                                # log-likelihood ratio for each cell
    for n in geometric_grid():                              # bisect to find minimum sample size
        LA = multinomial(n, cellsA, reps) @ llr;  LB = multinomial(n, cellsB, reps) @ llr
        thr = quantile(LB, 1-delta)                         # false-certification rate in world B <= delta
        if mean(LA <= thr) <= delta: return n
```

### X11 Exact Verification of the Small-Overlap Lower Bound (local CPU, <1h; end of Stage 1, after X10)

Starting point: `experiments/finite_state/small_overlap_floor.py` (already run locally; log at `results/local_runs/theory_checks-20261122/small_overlap_floor.log`). Base B(t,d,ρ;c₁,c₂) has 5 layers (dark D, startup U, bulk B, sticky M₁ and M₂), each with two labels, for 10 cells total. See “Small overlap” in variance_scaled_floor.md for definitions. b=0.

**X11a (P14 (i)–(iv), exact verification on an expanded grid).**
- k∈{4,6,8}; (c₁,c₂)∈{(3,10),(2,6)}. Keep only combinations with kΦ̄₀>1.
- t∈2^{12..20}, ρ∈{0.02,0.05,0.1}, ρ_B/ρ∈{1/16,1/4,1/2,1}. Require both 2kρ_B<1 and dρ≥5. Mark combinations that fail these conditions n/a, not as failures.
- Tilt Δ=1/4. Set ε to the actual margin min|G(±)|.
- For each combination record:
  - R from the quadratic form and from the explicit cell chain;
  - differences in the invariants (μ, Q(h=1));
  - for the S pair: a_U/pred, G∓/ε, KL_P·d/Δ², KL_Q·d/Δ²;
  - for the V_η pair: G∓/ε, KL_P/ε²;
  - gain K;
  - the maximum CS ratio and the maximum low-layer excess/e_b across 200 random compensating label pairs.
- Also report one row at ρ_B=ρ/16: a_U is near the cancellation point, and K≈0 is expected. This row is descriptive only and has no pass criterion.

**X11b (optional, P14 (iv), sample size for the optimal test).**
- Single-draw distributions in both worlds are known over 10 cells, so simulate the likelihood-ratio test directly using multinomial counts. Reuse X10b's `min_budget`.
- Fix ρ=0.05, ρ_B∈{ρ/2,ρ}, t∈2^{12..18}, and δ=0.05, with 20000 repetitions per point.
- Find the minimum p (q=0) separately for the S and V_η pairs. Record the ratio of n_S·ε_S² to n_V·ε_V².
- Prediction: the ratio is within a factor of 2 of K, and its log-log slope against t is in [0.8,1.2].

```python
for t, rho, rB in grid:
    d = rB*t; w, q = base(t, d, rho)                 # 5 layers; Σq=1 and Σqw=1 hold exactly
    m = moments(w, q, t)                             # compute C and S with backward recursion (O(n^3 t))
    ref = calibrate(m, th0, k)                       # tune bulk-layer labels until G_k=0; if no root, n/a
    S_pm = [tilt(m, ref, 1, s*0.25, k) for s in (+1, -1)]          # startup-layer labels ±Δ; dark and bulk layers compensate μ and Q(h=1)
    V_pm = [tilt(m, ref, 3, s*0.25, k) for s in (+1, -1)]          # sticky layers pair the signs θ / 1−θ
    eps = min(abs(gate(m, x, k)) for x in S_pm)
    log(a_over_pred, G/eps, kl(m, *S_pm, 'P')*d/D**2, kl(m, *S_pm, 'Q')*d/D**2, K, inv, cs_ratio, excess_over_eb)
```

### X7 Improving Auditor Constants (optional; run last)

- Replace A_clip's four intervals with betting confidence intervals (Waudby-Smith & Ramdas, 2024).
- For θ, use a conditional binomial Clopper–Pearson interval on h among accepted Q samples (accepted with probability v).
- On X1b and X4b, compare improvements in c* and minimum p. Do not change the theoretical pass criteria.

## 4. Order, Budgets, and Stop Points

| Stage | Work | Resources | Stop point |
|---|---|---|---|
| 1 | Shared components and unit tests → X2 → X1 (including X1e) → X3 → X8 → X9 (synthetic portion) → X10 → X11 | Local CPU | Write a stage report; wait for user confirmation before using GPU |
| 2 | X0 (SMOKE → full run) → X4 bank construction | Kaggle ≤9h | Stop if X0 fails |
| 3 | Freeze tasks → X4 audit → X9 (bank portion) → X6 | Local CPU | Report |
| 4 | X5, X7 | Optional | User decides |

## 5. Reporting Format

At the end of each stage, update `results/local_runs/STATUS_zh.md` in place, including:
- Status of each prediction (P1–P8): passed / failed / not run, with key figures and paths to plots.
- Any disagreement with theory, reported as observed.
- Costs: CPU hours, GPU hours, and sample counts.
- Deviations from the frozen config, with reasons.
