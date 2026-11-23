"""Frozen designs for the phase-1 remainder. Written to config.json before each run."""

from experiments.audit_lab.counts import NORMAL_THRESHOLD

COMMON = dict(phase=1, delta=0.05, nice=19, gpu_hours_max=0, memory_reserve_bytes=6_000_000_000,
              workers=9, worker_threads=1, load_ceiling=11.0,
              resource_note="M4 Max (14 cores, 36 GB): user allows <=10 workers, back off at load>11")


def config_for(name):
    if name == "X1E":
        return dict(
            COMMON, experiment="X1e", seed=2026112211,
            lowoverlap=dict(
                t=8, k=2, delta=.05, J=2048, bulk_mass=.75, mass_divisor=12, eps_divisor=2000,
                max_tilt=.072, tilt_tolerance=1e-12, c_grid=[2**i for i in range(15)], q_cap=2**60,
                reps=2000, n_anchor=163, normal_threshold=NORMAL_THRESHOLD, c0=16, power_target=.95,
                burn_in_over_eta0=0, eta0_grid=[1, 2**-8, 2**-12], r_grid=[.25, .5, .75],
                rho_grid=[2**24, 2**26, 2**28],
                dark_layer="Q-mass 1-eta0, raw score 1e-6*min family score, split evenly h=0/h=1",
                b_rule="b=0 as in X1 (plan gives no burn-in); with b>=1/eta0 the mixed chain is unsafe "
                       "for every beta (checked before freezing), so no calibrated pair exists there",
            ),
            saturation=dict(
                t=8, k=2, delta=.05, J=2048, r=.5, epsilon=2**-6, bulk_mass=.75, mass_divisor=12,
                rho_grid=[2**i for i in range(14, 45, 2)], c_grid=[2**(i / 4) for i in range(81)],
                c_unit="p = ceil(c * log(1/delta) / epsilon)", q_cap=2**60, reps=2000, n_anchor=163,
                normal_threshold=NORMAL_THRESHOLD, c0=16, power_target=.95,
                worlds="theta=1/2, beta tuned so exact G_k=-eps (safe) and +eps (unsafe)",
            ),
            auditor=dict(
                A_clip_grid="anchor a = empirical 1/4-quantile of S over n_anchor=ceil(32 log(8/delta)) "
                            "independent target rows (taken out of p); H_j=a*(q/p)*2^-j, "
                            "j=-3..ceil(log2(q/p))+1; each level delta/(4J) (8 one-sided Maurer-Pontil "
                            "bounds at delta/(32J)); intervals intersected; path test at delta/2 (as A_clip)",
                A_clip="unchanged audit_lab A_clip (C0=16) on the same banks",
            ),
            criteria=dict(
                P9_lowoverlap="for every design point (r, rho) with J/eta0 <= rho/2 for both small eta0: "
                              "A_clip_grid c* reached for eta0 in {2^-8, 2^-12} and max/min <= 4. "
                              "c* is on p = c*Psi(rho)*A with the original profile Psi and A=log(1/delta)/eps^2. "
                              "eta0=1 and A_clip are descriptive. Any censored (q_cap) c* makes the point fail.",
                P9_saturation="on points with Psi(rho) <= eps/4: all reached, max/min of p_min*eps/log(1/delta) <= 2, "
                              "and OLS slope of log p_min on log Psi(rho) in [-0.2, 0.2]",
                P9="both parts pass",
            ),
        )
    if name == "X8":
        return dict(
            COMMON, experiment="X8", seed=2026112212, eta0_grid=[2**-8, 2**-10], r_grid=[.25, .5],
            t_grid=[2**i for i in range(8, 21)], reps=2000, b=0,
            replicate="i,j iid uniform on {0..t-1}; jump lags min(i,j) and |i-j| (lag 0 costs 0); "
                      "tests of the two high-only lag jumps (jump_cost.py, no height cap) are summed",
            target_law="(1-eta0/2) Pareto(scale 1/eta0, index r) + eta0/2 at W=0.5",
            criteria=dict(
                P10="per (eta0, r): OLS slope of log mean cost on log(1+x), x=D_half/(eta0 t), "
                    "over points with 1+x <= 1/(4 eta0) (a factor 4 below the 1/eta0 truncation), in [0.8, 1.2]; "
                    "P10 passes iff all four slopes pass. Pooled slope descriptive.",
            ),
        )
    if name == "X9A":
        return dict(
            COMMON, experiment="X9a", seed=2026112213, t_grid=[2**i for i in range(3, 13)],
            b_rules=["0", "t/4", "4t"], random_per_kind=60, random_kinds=[0, 1, 2, 3],
            three_layer=dict(rho_grid=[2**12, 2**20, 2**28, 2**36], r_grid=[.25, .5, .75], beta=.8,
                             dark_eta0=[None, 2**-8, 2**-12]),
            pareto_low=dict(eta0_grid=[2**-8, 2**-10], r_grid=[.25, .5], bins=160),
            h_set="1{W>=v}, 1{W<=v} for v at 12 evenly spaced indices of sorted unique W; "
                  "3 random binary and 2 uniform h; plus the family's own h where defined",
            Xi="E_P min(W/t,1,L/W) + tau_P(L)^2 (plan definition; no 1/t term)",
            criteria=dict(P11="no case with max_h R/Gamma_t > 44 or max_h R/Xi > 1520; report sups"),
        )
    if name == "X9B":
        return dict(
            COMMON, experiment="X9b/X9c", seed=2026112214,
            families=[dict(name=f"lognormal s={s}", family="lognormal", sigma=s) for s in (1, 2, 3)]
            + [dict(name=f"pareto r={r} eta0=1/4", family="pareto_low", r=r, eta0=.25) for r in (.5, 1, 1.5)],
            t_grid=[2**i for i in range(4, 13)], k_grid=[2, 4], b=0,
            h_design=dict(upper=[.5, .7, .85, .95, .99], lower=[.5, .3, .15, .05, .01]),
            p_grid=[2**i for i in range(6, 17)], q_grid=[2**i for i in range(8, 19)], reps=200,
            C0=44, C0_explore=2,
            certificate="Cor. R2 with banks at delta/3: n0=ceil(32 log(6/delta)) target rows for the 5/8 "
                        "quantile c; Z_L = c * MP-LCB(min(S,c)/c) over all q rows; U_Gamma = MP-UCB of "
                        "min(S/Z_L,t)/t over (p-n0)//2 rows; V_L = MP-LCB of the Bernoulli(2Var) discordance "
                        "rate of disjoint pairs from the remaining rows, halved",
            compare=dict(k=2, t_grid=[2**4, 2**8, 2**12], auditor="A_clip (C0=16) with cold paths, "
                         "same (p,q) grid, Q split half center/half paths"),
            criteria=dict(
                P12_validity="false issues (exact G_k>=0) counted for X6; prediction 0",
                P12_slope="at q=2^18: minimum grid p with issue rate >= 0.8, OLS slope of log p_min on "
                          "log(k log(1/delta)/Var) using only instances with Var/(k Gamma_t) >= 1e3, in [0.8, 1.2]",
                P12_descriptive="same slope over all reached instances; x where the issue rate crosses 0.5",
                feasibility_note="E_P min(W,t) >= 1 for t >= 2, so Var/(k Gamma_t) <= t/(4k) <= 512 on this grid; "
                                 "recorded before running",
            ),
        )
    if name == "X10":
        return dict(
            COMMON, experiment="X10", seed=2026112215,
            x10a=dict(k_grid=[1.5, 2, 3, 4, 8], beta_grid=[0, 1, 2, 4], c_grid=[[1, 4], [.5, 2], [3, 10]],
                      t_grid=[2**i for i in range(10, 17)], rho_grid=[.005, .01, .02, .05],
                      eps_den=16, tilt=.25, fit_thetas=[.1 + .1 * i for i in range(9)],
                      keep="sigma=(k Phibar-1)/2-0.0111>0 (condition K)"),
            x10b=dict(k=2, beta=2, c=[1, 4], t_grid=[2**i for i in range(10, 17)],
                      rho_grid=[.005, .01, .02, .05], rho_min=.005, eps_den=16, reps=20000, delta=.05,
                      cap=2**55, fit_thetas=[.1 + .1 * i for i in range(9)],
                      search="doubling then integer bisection on the LR test power (both errors <= delta)"),
            criteria=dict(
                P13a="every kept combination: feasible; invariant diffs <= 1e-12; G(theta*-sgn*Delta)/eps <= -1 "
                     "and G(theta*+sgn*Delta)/eps >= 1; max quadratic-fit residual <= 1e-10 * max|G| over the fit points",
                P13b="pooled OLS log q_min ~ log rho + log t: both slopes in [0.8,1.2]; "
                     "OLS log p_min ~ log rho (all t pooled) slope in [0.8,1.2]",
                P13="both parts pass",
            ),
        )
    if name == "X11":
        return dict(
            COMMON, experiment="X11", seed=2026112216,
            x11a=dict(k_grid=[4, 6, 8], c_grid=[[3, 10], [2, 6]], t_grid=[2**i for i in range(12, 21)],
                      rho_grid=[.02, .05, .1], fr_grid=[1 / 16, 1 / 4, 1 / 2, 1], tilt=.25, cs_pairs=200, b=0, u=1.0,
                      keep="k*Phibar0>1 (combination dropped otherwise); rows with 2k rho_B>=1 or d rho<5, "
                           "or no calibrating bulk label, are n/a (not failures)",
                      eps="honest margin min|G(+-)| per pair (S and V separately)",
                      source="experiments/finite_state/small_overlap_floor.py imported unchanged"),
            x11b=dict(rho=.05, fr_grid=[1 / 2, 1], t_grid=[2**i for i in range(12, 19)], k=6, c=[3, 10],
                      reps=20000, delta=.05, cap=2**50,
                      search="LR test on 10-cell P counts (q=0), doubling then integer bisection, both errors <= delta"),
            criteria=dict(
                P14_hard="all ok rows: invariant diff <= 1e-12; quadratic-form R vs explicit 10-cell chain relative "
                         "diff <= 1e-12 (at the S reference and both S worlds); CS ratio <= 1; excess/e_low <= 1",
                P14_soft="ok rows with rho_B >= rho/2: a_U/pred in [0.85,1.15]; S and V pairs: G-/eps and G+/eps of "
                         "opposite sign with |.|>=1; S pair KL_P d/Delta^2 and KL_Q d/Delta^2 in [8.7,9.2]; for each "
                         "(k,c,rho,rho_B/rho) with >=3 t values, OLS slope of log K on log t in [0.8,1.2]",
                P14_descriptive="rho_B=rho/16 rows (a_U near cancellation, K expected ~0); tilt_infeasible rows listed",
                P14_x11b="optional: n_V eps_V^2 / (n_S eps_S^2) within a factor 2 of K at every point, and its log "
                         "slope in t in [0.8,1.2] for each rho_B",
                P14="hard and soft parts pass (X11b reported separately)",
            ),
        )
    raise ValueError(name)
