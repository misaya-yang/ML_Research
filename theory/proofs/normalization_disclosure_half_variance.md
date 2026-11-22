# Fixed half-target-variance sign at unknown normalization
Coordinate derivation audited independently; symmetric construction proposed by root.

For any fixed finite caps nP,nQ, sufficiently small ε, 0<η₀≤1/64, 0<δ≤.1, and Tη₀≥C/ε, deciding whether actual Q-start finite-window MSE is below or above half the target variance with ±ε promise and both errors ≤δ requires nQ=Ω(ε⁻²log(1/δ)/η₀) when only normalized-to-an-unknown-scale positive raw scores S=ZW are exposed. This holds with the known values Ph=0, VarP(h)=1 and public h and Q. In the known-scale interface zero Q queries suffice with O(η₀⁻¹ε⁻²log(1/δ)) P draws by the high-only endpoint estimator; this is a separation for the actual fixed half-variance test. The latter is an upper, not a claim that the exact known-normalized sign lower bound itself was proved here.

## Hard family

Let a∈[2η₀,2.01η₀] be rational; let c near the (2,3) root of J(c)=1/2, where J(c)=2∫₀¹(1−v)e^(−cv)dv. Put H₀=c/(2T), L₀=a−H₀, and take common raw scores γ,2γ,1,L₀/H₀. Use h∈{−1,+1}, with low core L all minus, high core H all plus. Compare core multiplier s close to 1 with another close multiplier, keeping all scores fixed:
```
q_L=s L₀, q_H=s H₀
d_−=1/2−q_L, d_+=1/2−q_H
q_D1,sign=2d_sign−B, q_D2,sign=B−d_sign, B=7/10.
```
For s∈[.9,1.1], all dark cells positive (d_sign∈(.46,.5], since a≤.032); |w_H|>1 and |w_L|>1 for large Tη₀, because Z=2sL₀+2γB<1. (The absolute-value notation is immaterial; weights are strictly positive.) Q signs exactly balanced. The raw dark target mass in each sign equals γ(q_D1+2q_D2)=γB and raw core mass in each sign equals sL₀. Thus Ph=0 and Var=1 exactly, despite variation in Z. Overlap is at least q_H+q_L=sa≥1.8η₀. Hidden-weight labels are uniformly and independently randomized within equal half-sign populations of an explicit uniform Q; all assigned counts can be exact with rational a,c,s,γ and arbitrarily large suitable N.

## Actual half-threshold crossing

Let τ count initial dark states before the first core proposal. From any dark state a core proposal is accepted, so E[min(τ,T)]≤1/(sa); dark behavior until τ cannot alter that time. Couple the path to one that suppresses every attempted dark move once a core state has first been reached, keeping all behavior before τ. Each step after τ differs with probability at most γ q_D1,all+2γ q_D2,all=2γB conditional on the common core state. Therefore true risk differs from suppressed-dark risk by O(Tγ), since squared means lie in [0,1].

After τ the suppressed-dark chain is symmetric two-state: L→H probability q_H, H→L probability q_L/(L₀/H₀)=q_H. For any i,j∈{0,…,T−1} with i≤j and i≥τ, the conditional product expectation given the history through i equals λ^(j−i), λ=1−2q_H, regardless of the sign at first entry. Treating all pairs as core can alter the expectation by at most twice the fraction of pairs with minimum index <τ, at most 4 E[min(τ,T)]/T. Thus
```
R(s) = [T+2 Σ_{j=1}^{T−1}(T−j)(1−sc/T)^j]/T² + O(1/(Ta)+Tγ)
     = J(sc)+O(1/(Ta)+Tγ).
```
The latter finite Riemann error O(1/T) is uniform on c∈[2,3], s∈[.9,1.1]; it is covered by the startup term. The derivative in c of J is −2∫₀¹ v(1−v)e^(−cv)dv, strictly negative with absolute value bounded below near the unique root. Choose rational c so |J(c)−.5|<ε/4, and choose rational s_±=1 ± Θ(ε) with a fixed, sufficiently large multiplier in Θ. Event errors can be made <ε/4 by Tη₀≥C/ε and γ tiny enough. Then R(s_−)≥.5+ε and R(s_+)≤.5−ε. The normalization-independent calibrated margin is actual R−VarP(h)/2, not a threshold fitted to data after observation.

## Oracle information

Q cores each change by a relative O(Δ) where Δ=|s_+−s_−|, core mass total O(a). Dark Q cell masses are Θ(1), absolute changes O(aΔ). Thus KL for fresh Q is O(aΔ²). A fresh named query at a chosen public sign has conditional color KL of at most twice the same bound because Q sign masses are each 1/2 in both worlds. P dark cells are Θ(γ/a) and change by O((γ/a)Δ); P core cells are each (1−P(dark))/2 and also change by O((γ/a)Δ). Fresh P KL is O((γ/a)Δ²). Fixed raw score support is essential: there is no exact unknown Z disclosure at a known name.

Cache all previously seen names. After any consistent history, hidden assignments in each public sign class are still uniform over remaining arrangements because the likelihood of that history depends only on recorded weights. Thus conditional type law for next unqueried label is its fresh ideal type law up to TV O(K Hmax/N) under either source, where K=nP+nQ and Hmax=max W=O(T); on a sampler request hitting an already seen name contributes at most the same bound. Conditional new names in a given public sign class remain uniform. Sequential coupling yields total TV O(K²Hmax/N), made <δ/4 by picking N after all rational parameters/caps. Apply KL chain rule in the ideal transcript, conditioning on choices (adaptive named queries or stopping included), then binary testing. This gives a(ε² nQ)+γ/a·(ε² nP) ≥ c log(1/δ). Pick γ arbitrarily small after any finite nP, while preserving the exact target moments and risk separation; hence nQ floor follows. It disappears if the scored observations disclose normalized W or exact target atomic probabilities along with raw scores, revealing Z.

## Reproduction

`OPENBLAS_NUM_THREADS=1 python half_gate.py`. It checks normalization and target/proposal sign symmetry in exact fractions, then finite risk with an independent dense six-state conditional moment recursion in binary64. This control is not the analytical small-ε proof.
