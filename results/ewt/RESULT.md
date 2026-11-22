# Zero-Q normalized baseline on the six original disclosure-pilot EWT cells

No new holdout. Same six cells, binary gold-token indicator, q(a=.72), Q-start budget T=1024, no Q queries, and score access includes true W=dP/dQ. Target mean hidden. One preset fresh P seed per cell. Each reported score budget is a separate fixed-sample audit with nominal per-cell error ≤.025, matching the original high-mark pilot's union error allocation. As with the original, numerical calculation is binary64, not outward-verified. No comparison across the budgets is itself a confidence statement.

Choose a maximal initial coupling between Q and P by matching on min(P,Q). Its mismatch mass expressed under P draws of W>1 is (1−1/W). Given mismatch, the cold draw has W<1 and the stationary draw has W>1. In an ordinary common-proposal/common-uniform IMH coupling, they coalesce at the stationary state’s first acceptance, whose hazard α(w)=E_P[1/max(w,W′)]. This follows because the smaller-weight state accepts whenever the larger-weight state accepts; any proposal capable of making the smaller state larger is already accepted by both. For duration τ≥1, let D=min(τ,T)/T. Then |bar h_Q−bar h_P|≤D, and both duration moments under P residual weight are computable by finite geometric sums.

Use P-only normalized score lower α bounds from DKW on an independent pilot, via α(w)=w⁻¹E_P min(1,w/W′); integrand is monotone bounded by one. Apply empirical Bernstein on remaining independent target rows to upper-bound E D and E D². For E D² the test uses sums of (2i+1)(1−α)^i/T² (alpha sufficiently close to zero uses bound 1). The original normalized stationary spectral confidence band supplies Rstat/Var upper u, and Clopper–Pearson supplies target mean band and v=Var range. The tested half-variance gap is bounded by the tighter of two true inequalities:
- R_Q−v/2≤(u−.5)v+g(E D), where M=max(m,1−m), g(d)=2Md−d² for d≤M, saturated at M². Difference of binary squared errors is bounded by g of their absolute difference, and g is concave.
- R_Q−v/2≤(u−.5)v+2√(u v E D²)+E D² by Minkowski.
Both optimize over the same confidence sets; choose min under the joint event. Allocate error .01 stationary, .005 mean/variance, .005 pilot lower α, and .0025 per duration moment. Same target draws can be reused for stationary and target confidence because only the pilot-to-duration EB stage needs conditional independence. No exact means are read during the audit.

## Results

| Scored P | Certified of 6 |
|---:|---:|
|128|0|
|256|0|
|1024|4|
|4096|6|
|8192|6|
|16384|6|
|29400|6|

Original high-mark pilot certified 4/6 with 29,491–29,511 consumed scored P; matched normalized access beats this at markedly less cost. Its comparison to ordinary standalone trajectories does not establish a method advantage. The same six cells are all truly safe according to the archived numerical trajectory evaluation; no claim of independent transfer or unsafe-case power. None of these observations applies to unknown normalization or to arbitrary fixed b,t/h.

Reproduce from the project root: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/ewt/baseline.py`. See [high-mark code](../../experiments/ewt/sources/pilot.py), [high-mark results](high_mark/pilot.csv) and [baseline results](stationary_coupling/baseline.csv). New runs use separate directories.
