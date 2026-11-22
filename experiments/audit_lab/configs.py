"""Phase 1 design choices, written to config.json before each experiment."""

from .counts import NORMAL_THRESHOLD


def config_for(experiment):
    common = dict(experiment=experiment, phase=1, delta=0.05, cpu_threads=1,
                  memory_reserve_bytes=6_000_000_000, nice=19, gpu_hours_max=0)
    if experiment == "X2":
        return dict(
            common, seed=2026112202, q_grid=[2**i for i in range(8, 31)],
            p_grid=[0, 1, 2, 4, 8, 16], k=2, b=0, coin_t=4, layer_t=8,
            reps=2000, lr_reps=20000, gamma="1/q", bulk_mass=.75,
            i2=dict(J=2048, K=2**14, tail_r=.5, mass_divisor=12,
                    kappas=[.02, .05, .1, .2]),
            i3=dict(r_grid=[.5, 1.], j_min=-6, j_max=30, extra_exponent=.05,
                    label_amplitude=.125, low_q_rule="uniform within j<0"),
            roc=dict(q=2**8, qd_grid=[10.0**-i for i in range(3, 11)],
                     alpha="I1 kappa (equality-cap alternative)"),
            rate_p=[2, 8],
            scaling=dict(epsilon_grid=[.04, .02, .01, .005, .0025],
                         verify_epsilons=[.04, .01, .0025], verify_q=2**30,
                         uniform_q=2**20, uniform_t=2**16, uniform_alpha=2.0**-50,
                         heavy_mass=.8, numerical_margin=1e-9),
            criteria=dict(
                P3=dict(abs_error_se_multiplier=3, finite_q_at="max q, each X2b cell"),
                P4=dict(target=.8, pointwise_slope=[-1.15, -.85],
                        uniform_slope=[-2.3, -1.7]),
                P5=dict(known_min_power=.9, unknown_null_se_multiplier=3),
                P6=dict(fit="OLS on all strictly positive gaps, report remaining gaps",
                        slope_rule="-r/(1+r)+0.1", slack=.1),
            ),
        )
    if experiment == "X1":
        return dict(
            common, seed=2026112201, b=0, t=8, k=2, J=2048, bulk_mass=.75,
            rho_grid=[2**i for i in range(12, 41, 2)],
            profiles=[dict(name="r=.25", r=.25), dict(name="r=.5", r=.5),
                      dict(name="r=.75", r=.75),
                      dict(name="broken", r=.25, knee=2**20, right_r=.75)],
            psi_max=.5, eps_divisor=2000, mass_divisor=12, max_tilt=.072,
            tilt_tolerance=1e-12, c_grid=[2**i for i in range(15)],
            q_cap=2**60, normal_expected_threshold=NORMAL_THRESHOLD,
            reps=2000, lr_reps=20000, c0=16,
            adversary=dict(r=.5, rho_grid=[2**18, 2**22, 2**26],
                           K_over_rho_exponents=list(range(-4, 5))),
            criteria=dict(
                P1=dict(power_target=.95, normalized_lr_ratio_max=4, clip_ratio_max=8,
                        scope="all in-band design points",
                        censored="report observed ratios and mark full-grid criterion unverified"),
                P2=dict(min_points_each_side=4, ci_level=.95, slopes=[-1 / 3, -3],
                        knee_assignment="exclude the kink itself",
                        insufficient_points="未运行"),
                X1d="descriptive",
            ),
        )
    if experiment == "X3":
        return dict(
            common, seed=2026112203, k_grid=[1.5, 2, 3, 4, 8],
            floor_t=2**16, b=0, t_rule="t=2*k", J_rule="J=128*k*(b+t)",
            K_over_J=16, profile_r=.5, bulk_mass=.75, mass_divisor=12,
            eps_divisor=2000, max_tilt=.072, tilt_tolerance=1e-12,
            criteria=dict(P7="all k: G_safe <= -epsilon and G_unsafe >= epsilon"),
        )
    raise ValueError(experiment)
