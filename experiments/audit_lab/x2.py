"""X2: finite-p ceilings, rates, scale disclosure, and budget exponents."""

import math

import matplotlib.pyplot as plt
import numpy as np

from .bench import Bank, evaluate_audit, minimum_lr, save_figure, slope
from .families import ThreeLayers, center_floor, fair_coin, gap_and_kappa, heavy_tail, i2_cells
from .imh import Pi_p
from .run_io import Finding


def _roc(run, config, kap):
    design, delta = config["roc"], config["delta"]
    q, grid = design["q"], design["qd_grid"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    differences = []
    for p in config["p_grid"]:
        ceiling = Pi_p(delta, kap, p)
        powers = []
        for x in grid:
            log_e0 = (p + q) * math.log1p(-x)
            log_e1 = log_e0 + p * math.log1p(-kap)
            power = -math.expm1(log_e0) + math.exp(log_e0 + min(0, math.log(delta) - log_e1))
            run.record("X2a", p=p, q=q, QD=x, P0D=x, P1D=kap + (1 - kap) * x,
                       alpha=kap, power=power, ceiling=ceiling)
            powers.append(power)
        differences.append(abs(powers[-1] - ceiling))
        line, = ax.semilogx(grid, powers, label=f"p={p}")
        ax.axhline(ceiling, color=line.get_color(), linestyle=":", linewidth=1)
    ax.set(xlabel="Q(D)", ylabel="Exact simple-world power", title="X2a. Rare-set ROC")
    ax.invert_xaxis()
    ax.legend(ncols=2)
    save_figure(run, fig, "X2a_roc.png")
    return max(differences)


def _instances(run, config):
    b, k = config["b"], config["k"]
    setup = config["i2"]
    m = (setup["J"] / setup["K"])**setup["tail_r"] / setup["mass_divisor"]
    family = ThreeLayers(setup["J"], setup["K"], m, config["bulk_mass"])
    coin = Bank(fair_coin(), b, config["coin_t"], k)
    regular = {"I1": coin}
    heavy, truth = {}, {}
    for kap in setup["kappas"]:
        cells, beta = i2_cells(family, b, config["layer_t"], k, kap=kap)
        name = f"I2 kappa={kap:g}"
        regular[name] = Bank(cells, b, config["layer_t"], k)
        run.record("instances", instance=name, beta=beta)
    setup = config["i3"]
    for r in setup["r_grid"]:
        cells = heavy_tail(r, setup["j_min"], setup["j_max"],
                           setup["extra_exponent"], setup["label_amplitude"])
        name = f"I3 r={r:g}"
        heavy[name] = Bank(cells, b, config["layer_t"], k)
        moment = float(np.sum(cells.q * cells.w**(1 + r)))
        run.record("instances", instance=name, r=r, moment_M=moment)
    for name, bank in {**regular, **heavy}.items():
        gap, kap, mu, lo, hi = gap_and_kappa(bank.cells, b, bank.t, k)
        if name in heavy and not .3 < mu < .7:
            raise ValueError(f"{name} violates the prespecified mean band")
        truth[name] = kap
        run.record("truth", instance=name, b=b, t=bank.t, k=k, G=gap, kappa=kap,
                   mu=mu, root_lo=lo, root_hi=hi)
    return family, regular, heavy, truth


def _ceilings(run, config, rng, banks, truth, roc_error):
    q_grid, reps, delta = (config[key] for key in ("q_grid", "reps", "delta"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    failures = []
    count, worst_gap = 0, 0.
    for (name, bank), ax in zip(banks.items(), axes.flat):
        for p in config["p_grid"]:
            ceiling = Pi_p(delta, truth[name], p)
            powers = []
            for q in q_grid:
                est = evaluate_audit(run, rng, "X2b", bank, "A_ep", p, q, reps, delta,
                                     instance=name, ceiling=ceiling, kappa=truth[name])
                powers.append(est.power)
            err = abs(est.power - ceiling)
            limit = config["criteria"]["P3"]["abs_error_se_multiplier"] * est.se
            passed = err <= limit
            run.record("X2b_criterion", instance=name, p=p, q=q_grid[-1], power=est.power,
                       se=est.se, ceiling=ceiling, error=err, passed=passed)
            count += 1
            worst_gap = max(worst_gap, err)
            if not passed:
                failures.append((name, p, est.power, ceiling, est.se))
            line, = ax.semilogx(q_grid, powers, label=f"p={p}")
            ax.axhline(ceiling, color=line.get_color(), linestyle=":", linewidth=.8)
        ax.set(title=name, xlabel="q", ylabel="Issue probability", ylim=(-.02, 1.04))
    axes.flat[5].axis("off")
    axes.flat[0].legend(ncols=2, fontsize="small")
    save_figure(run, fig, "X2b_ceiling.png")
    detail = (f"X2a 最小 Q(D) 处与上限最大差 {roc_error:.3g}。"
              f"X2b 在 q=2³⁰ 有 {count - len(failures)}/{count} 格落在 3 MC SE 内，"
              f"最大绝对差 {worst_gap:.4g}。")
    if failures:
        name, p, power, ceiling, se = max(failures, key=lambda row: abs(row[2] - row[3]))
        detail += f"最大偏差格为 {name}、p={p}，功效 {power:.6f}、上限 {ceiling:.6f}、SE {se:.3g}。"
    return Finding("未通过" if failures else "通过", detail, ("X2a_roc.png", "X2b_ceiling.png"))


def _rates(run, config, rng, banks, truth):
    q_grid, reps, delta = (config[key] for key in ("q_grid", "reps", "delta"))
    fig, axes = plt.subplots(1, len(banks), figsize=(10, 4.4))
    results = []
    q_array = np.asarray(q_grid)
    for (name, bank), r, ax in zip(banks.items(), config["i3"]["r_grid"], axes):
        limit = -r / (1 + r) + config["criteria"]["P6"]["slack"]
        for p in config["rate_p"]:
            ceiling = Pi_p(delta, truth[name], p)
            gaps = []
            for q in q_grid:
                est = evaluate_audit(run, rng, "X2c", bank, "A_ep", p, q, reps, delta,
                                     instance=name, r=r, ceiling=ceiling, kappa=truth[name])
                gaps.append(ceiling - est.power)
                run.record("X2c_gap", instance=name, r=r, p=p, q=q, gap=gaps[-1])
            gaps = np.asarray(gaps)
            positive = gaps > 0
            use = int(positive.sum())
            fit = slope(q_array[positive], gaps[positive]) if use >= 2 else {}
            passed = fit["slope"] <= limit if fit else None
            run.record("X2c_fit", instance=name, r=r, p=p, slope_limit=limit,
                       positive_points=use, nonpositive_points=len(gaps) - use, passed=passed, **fit)
            results.append((r, p, fit.get("slope"), limit, len(gaps) - use, passed))
            ax.loglog(q_array[positive], gaps[positive], "o-", label=f"p={p}", markersize=3)
        ax.set(xlabel="q", ylabel="Ceiling − power", title=name)
        ax.legend()
    save_figure(run, fig, "X2c_gap.png")
    detail = "，".join(
        f"r={r:g} / p={p} 为 {value:.3f}（阈值 {limit:.3f}，{skipped} 点缺口≤0）"
        if value is not None else f"r={r:g} / p={p} 只有 {len(q_grid) - skipped} 个正缺口"
        for r, p, value, limit, skipped, _ in results
    )
    verdicts = [x[-1] for x in results]
    state = "未通过" if False in verdicts else ("未运行" if None in verdicts else "通过")
    return Finding(state, f"全 q 网格的正缺口斜率为 {detail}。", ("X2c_gap.png",))


def _scaling(run, config, rng, family):
    design, criterion = config["scaling"], config["criteria"]["P4"]
    epsilons = design["epsilon_grid"]
    goal, delta, k, b, t = (criterion["target"], config["delta"], config["k"],
                           config["b"], config["layer_t"])
    pointwise, uniform, verification = [], [], []
    margins_valid = True
    for eps in epsilons:
        run.guard.check()
        cells, beta = i2_cells(family, b, t, k, epsilon=eps)
        gap, kap, mu, lo, hi = gap_and_kappa(cells, b, t, k)
        p = math.ceil(math.log(goal / delta) / -math.log1p(-kap))
        pointwise.append(p)
        run.record("X2d_pointwise", epsilon=eps, G=gap, beta=beta, kappa=kap,
                   mu=mu, root_lo=lo, root_hi=hi, p_min=p, ceiling=Pi_p(delta, kap, p))
        if eps in design["verify_epsilons"]:
            bank = Bank(cells, b, t, k)
            est = evaluate_audit(run, rng, "X2d_verify", bank, "A_ep", p, design["verify_q"],
                                 config["reps"], delta, epsilon=eps, ceiling=Pi_p(delta, kap, p))
            verification.append((eps, est.power))

        floor_args = (design["uniform_alpha"], design["uniform_t"], k,
                      design["heavy_mass"], design["numerical_margin"])
        safe, g_safe, theta_safe = center_floor(eps, -1, *floor_args)
        run.guard.check()
        unsafe, g_unsafe, theta_unsafe = center_floor(eps, 1, *floor_args)
        margins_valid &= g_safe <= -eps and g_unsafe >= eps
        p, result, status = minimum_lr(
            run, rng, "X2d_uniform", unsafe, safe, lambda _: design["uniform_q"],
            goal, config["lr_reps"], delta, epsilon=eps,
        )
        uniform.append(p)
        run.record("X2d_uniform_min", epsilon=eps, G_safe=g_safe, G_unsafe=g_unsafe,
                   theta_safe=theta_safe, theta_unsafe=theta_unsafe, p_min=p,
                   power=result.power, se=result.se, status=status, q=design["uniform_q"])

    slopes = [slope(epsilons, ps)["slope"] for ps in (pointwise, uniform)]
    run.record("X2d_fit", slope_pointwise=slopes[0], slope_uniform=slopes[1],
               margins_valid=margins_valid)
    passed = margins_valid and all(
        criterion[key][0] <= value <= criterion[key][1]
        for key, value in zip(("pointwise_slope", "uniform_slope"), slopes)
    )
    fig, ax = plt.subplots(figsize=(7, 4.7))
    ax.loglog(epsilons, pointwise, "o-", label="Pointwise, ceiling")
    ax.loglog(epsilons, uniform, "s-", label="Uniform, LR")
    ax.set(xlabel="Gate margin epsilon", ylabel="Minimum p for 0.8 power", title="X2d. Target budget")
    ax.legend()
    save_figure(run, fig, "X2d_scaling.png")
    values = "，".join(f"ε={eps:g} 时 {power:.4f}" for eps, power in verification)
    detail = (f"逐点斜率 {slopes[0]:.3f}，一致斜率 {slopes[1]:.3f}。"
              f"逐点公式预算在 q=2³⁰ 的 A_ep 实测为 {values}。")
    if not margins_valid:
        detail += "至少一个一致族实例没有跨过指定间隔。"
    return Finding("通过" if passed else "未通过", detail, ("X2d_scaling.png",))


def _disclosure(run, config, rng, banks):
    q_grid, reps, delta = (config[key] for key in ("q_grid", "reps", "delta"))
    criterion = config["criteria"]["P5"]
    fig, axes = plt.subplots(1, len(banks), figsize=(12, 4), sharey=True)
    end_known, max_unknown = [], 0.
    passed = True
    for (name, bank), ax in zip(banks.items(), axes):
        plot = {}
        for method in ("A_ep", "A_knownZ"):
            values = []
            for q in q_grid:
                est = evaluate_audit(run, rng, "X2e", bank, method, 0, q, reps,
                                     delta, instance=name)
                values.append(est.power)
                if method == "A_ep":
                    passed &= est.power <= delta + criterion["unknown_null_se_multiplier"] * est.se
                    max_unknown = max(max_unknown, est.power)
            plot[method] = values
        end_known.append(plot["A_knownZ"][-1])
        passed &= end_known[-1] >= criterion["known_min_power"]
        for method, values in plot.items():
            ax.semilogx(q_grid, values, label=method)
        ax.axhline(delta, color="0.5", linestyle=":")
        ax.set(xlabel="q", title=name)
    axes[0].set(ylabel="Issue probability at p=0")
    axes[0].legend()
    save_figure(run, fig, "X2e_disclosure.png")
    detail = (f"X2e 在 q=2³⁰ 的已知 Z 功效（I1、I3 的 r=.5、1）为 "
              f"{'、'.join(f'{v:.4f}' for v in end_known)}，未知 Z 的全网格最大值 "
              f"{max_unknown:.9f}。X4a 待运行。")
    return Finding("通过" if passed else "未通过", detail, ("X2e_disclosure.png",))


def run(run, config):
    rng = np.random.default_rng(config["seed"])
    print("X2 instances and X2a", flush=True)
    family, regular, heavy, truth = _instances(run, config)
    roc_error = _roc(run, config, truth["I1"])
    print("X2b ceilings", flush=True)
    p3 = _ceilings(run, config, rng, regular, truth, roc_error)
    print("X2c rates", flush=True)
    p6 = _rates(run, config, rng, heavy, truth)
    print("X2d target scaling", flush=True)
    p4 = _scaling(run, config, rng, family)
    print("X2e disclosure", flush=True)
    p5 = _disclosure(run, config, rng, {"I1": regular["I1"], **heavy})
    notes = [
        "I2 和 I3 共用 b=0、t=8、k=2。I2 标定到指定 κ 或 G，I3 的低层 "
        "Q 质量均分，标签概率取 0.5+0.125j/(1+|j|)。",
        "X2a 用 I1 的等号 cap 构造 α=κ、q=2⁸。X2d 一致族用有符号 G、"
        "T=2¹⁶、α=2⁻⁵⁰。LR 在各预算点独立模拟两个世界，以不安全侧的经验分位校准。",
        "A_ep 取 γ=1/q，SE 来自逐次签发概率。确定性的有限 q 偏差照样计入 P3，"
        "包括 p=0 时 δ−γ 与 δ 的差。A_knownZ 的截断 Hoeffding 半径按实际均值半库样本量计算。",
        "X2c 保留每个缺口，只对严格正的缺口取对数拟合。冻结后没有更改配置。",
    ]
    return {"P3": p3, "P4": p4, "P5": p5, "P6": p6}, notes
