"""X1: lower and upper budget frontiers and the adversary's weight level."""

from collections import Counter
import math

import matplotlib.pyplot as plt
import numpy as np

from .bench import Bank, evaluate_audit, minimum_lr, save_figure, slope
from .families import ThreeLayers, calibrated_pair, tail_profile
from .run_io import Finding


def _pair(config, K, psi, epsilon):
    family = ThreeLayers(config["J"], K, psi / config["mass_divisor"], config["bulk_mass"])
    worlds = calibrated_pair(family, config["b"], config["t"], config["k"], epsilon,
                             max_tilt=config["max_tilt"], tilt_tolerance=config["tilt_tolerance"])
    return worlds, family.m


def _lr(run, config, rng, section, worlds, meta):
    rho = meta["rho"]
    p, result, status = minimum_lr(
        run, rng, section, worlds.unsafe, worlds.safe, lambda p: rho * p,
        config["criteria"]["P1"]["power_target"], config["lr_reps"], config["delta"],
        max_p=config["q_cap"] // rho, normal_threshold=config["normal_expected_threshold"],
        **meta,
    )
    run.record(section, kind="minimum", status=status,
               p=p, q=rho * p, power=result.power, se=result.se, **meta)
    return p, status


def _clip(run, config, rng, worlds, meta):
    rho, psi, eps = (meta[key] for key in ("rho", "psi", "epsilon"))
    delta = config["delta"]
    base_p = psi * math.log(1 / delta) / eps**2
    safe = Bank(worlds.safe, config["b"], config["t"], config["k"])
    unsafe = Bank(worlds.unsafe, config["b"], config["t"], config["k"])
    attained = None
    skipped = 0
    for c in config["c_grid"]:
        p, q = math.ceil(c * base_p), rho * math.ceil(c * base_p)
        if q > config["q_cap"]:
            skipped += 1
            run.record("X1b", kind="skipped", status="q_cap", c=c, p=p, q=q, **meta)
            continue
        if attained is not None:
            continue
        est = evaluate_audit(
            run, rng, "X1b", safe, "A_clip", p, q, config["reps"], delta,
            normal_threshold=config["normal_expected_threshold"], c0=config["c0"], c=c, **meta,
        )
        evaluate_audit(
            run, rng, "X1b", unsafe, "A_clip", p, q, config["reps"], delta, unsafe=True,
            normal_threshold=config["normal_expected_threshold"], c0=config["c0"], c=c, **meta,
        )
        if est.power >= config["criteria"]["P1"]["power_target"]:
            attained = (c, p, q, est)
    if attained is None:
        status = "q_cap" if skipped else "c_grid_exhausted"
        run.record("X1b", kind="minimum", status=status, **meta)
        return None, status, skipped
    c, p, q, est = attained
    run.record("X1b", kind="minimum", status="reached", c=c, p=p, q=q,
               power=est.power, se=est.se, **meta)
    return (c, p, q), "reached", skipped


def _frontier(run, config, rng):
    count_lr, count_clip = Counter(), Counter()
    curves, broken = {}, []
    skipped_candidates = 0
    for profile in config["profiles"]:
        name = profile["name"]
        print(f"X1a/b profile {name}", flush=True)
        values = []
        for rho in config["rho_grid"]:
            psi = tail_profile(rho, config["J"], profile["r"],
                               knee=profile.get("knee"), right_r=profile.get("right_r"))
            meta = dict(profile=name, rho=rho, K=rho, psi=psi, epsilon=psi / config["eps_divisor"])
            if rho < config["J"] or psi > config["psi_max"]:
                run.record("grid", status="outside_theorem_band", **meta)
                continue
            run.guard.check()
            try:
                worlds, m = _pair(config, rho, psi, meta["epsilon"])
            except ValueError as error:
                run.record("grid", status="calibration_failed", error_message=str(error), **meta)
                count_lr["calibration_failed"] += 1
                count_clip["calibration_failed"] += 1
                continue
            run.record("grid", status="calibrated", m=m, beta=worlds.beta, s=worlds.s,
                       C_tilt=worlds.s * m / meta["epsilon"],
                       G_safe=worlds.g_safe, G_unsafe=worlds.g_unsafe, bracket=worlds.bracket,
                       **meta)
            p, status = _lr(run, config, rng, "X1a", worlds, meta)
            count_lr[status] += 1
            normalized = (p * meta["epsilon"]**2 / math.log(1 / config["delta"]) / psi
                          if status == "reached" else None)
            if normalized is not None:
                run.record("X1a_normalized", p_min=p,
                           y=p * meta["epsilon"]**2 / math.log(1 / config["delta"]),
                           y_over_psi=normalized, **meta)
            clip, status, skipped = _clip(run, config, rng, worlds, meta)
            skipped_candidates += skipped
            count_clip[status] += 1
            values.append((rho, normalized, clip[0] if clip else None))
            if profile.get("knee") and clip:
                broken.append((rho, clip[1], clip[2]))
        curves[name] = values
    return curves, broken, count_lr, count_clip, skipped_candidates


def _frontier_figures(run, curves):
    for i, name, ylabel, title in (
        (1, "X1a_lr_frontier.png", "p epsilon² / (log(1/delta) Psi)", "X1a. LR frontier"),
        (2, "X1b_clip_frontier.png", "Minimum grid constant c*", "X1b. Clipped-certificate frontier"),
    ):
        fig, ax = plt.subplots(figsize=(7.8, 4.7))
        for profile, values in curves.items():
            ax.loglog([row[0] for row in values], [row[i] for row in values], "o-",
                      markersize=3, label=profile)
        ax.set(xlabel="Budget ratio q/p", ylabel=ylabel, title=title)
        ax.legend()
        save_figure(run, fig, name)


def _kink(run, config, broken):
    knee = next(profile["knee"] for profile in config["profiles"] if profile.get("knee"))
    criterion = config["criteria"]["P2"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fits, sizes = [], []
    for key, points, target in (
        ("left", [row for row in broken if row[0] < knee], criterion["slopes"][0]),
        ("right", [row for row in broken if row[0] > knee], criterion["slopes"][1]),
    ):
        size = len(points)
        sizes.append(size)
        ax.loglog([row[2] for row in points], [row[1] for row in points], "o-", label=key)
        if size < criterion["min_points_each_side"]:
            run.record("X1c_fit", side=key, status="insufficient_points", points=size, target=target)
            continue
        estimate = slope([row[2] for row in points], [row[1] for row in points],
                         ci_level=criterion["ci_level"])
        passed = estimate["ci_lo"] <= target <= estimate["ci_hi"]
        fits.append((passed, estimate))
        run.record("X1c_fit", side=key, status="fit", target=target, passed=passed, **estimate)
    at_knee = [row for row in broken if row[0] == knee]
    if at_knee:
        ax.loglog([row[2] for row in at_knee], [row[1] for row in at_knee], "kD", label="kink")
    ax.set(xlabel="q at minimum p", ylabel="Minimum p", title="X1c. Broken profile")
    ax.legend()
    save_figure(run, fig, "X1c_kink.png")
    if len(fits) < 2:
        return Finding("未运行", f"拐点两侧分别有 {sizes[0]} 和 {sizes[1]} 个可拟合点，"
                       "固定判据要求每侧至少 4 点。", ("X1c_kink.png",))
    detail = "、".join(f"{fit['slope']:.3f}（95% CI [{fit['ci_lo']:.3f}, {fit['ci_hi']:.3f}]）"
                       for _, fit in fits)
    return Finding("通过" if all(passed for passed, _ in fits) else "未通过",
                   f"两段斜率为 {detail}。", ("X1c_kink.png",))


def _adversary(run, config, rng):
    design = config["adversary"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    maxima = []
    for rho in design["rho_grid"]:
        print(f"X1d rho={rho}", flush=True)
        epsilon = tail_profile(rho, config["J"], design["r"]) / config["eps_divisor"]
        x, y = [], []
        for expo in design["K_over_rho_exponents"]:
            K = rho * 2**expo if expo >= 0 else rho // 2**(-expo)
            psi = tail_profile(K, config["J"], design["r"])
            meta = dict(profile="r=.5", rho=rho, K=K, K_over_rho=K / rho, psi=psi, epsilon=epsilon)
            if K < config["J"] or psi > config["psi_max"]:
                run.record("X1d", kind="skipped", status="outside_theorem_band", **meta)
                continue
            run.guard.check()
            try:
                worlds, m = _pair(config, K, psi, epsilon)
            except ValueError as error:
                run.record("X1d", kind="skipped", status="calibration_failed",
                           error_message=str(error), **meta)
                continue
            run.record("X1d", kind="calibration", m=m, beta=worlds.beta, s=worlds.s,
                       C_tilt=worlds.s * m / epsilon, G_safe=worlds.g_safe,
                       G_unsafe=worlds.g_unsafe, **meta)
            p, status = _lr(run, config, rng, "X1d", worlds, meta)
            x.append(K / rho)
            y.append(p if status == "reached" else np.nan)
        if np.any(np.isfinite(y)):
            maxima.append(f"ρ=2^{rho.bit_length() - 1} 时 K/ρ={x[int(np.nanargmax(y))]:g}")
        ax.loglog(x, y, "o-", markersize=3, label=f"rho=2^{rho.bit_length() - 1}")
    ax.set(xlabel="Adversary K / rho", ylabel="Minimum p (LR)", title="X1d. Adversary weight")
    ax.legend()
    save_figure(run, fig, "X1d_adversary_K.png")
    return "，".join(maxima)


def _ratio(numbers):
    return max(numbers) / min(numbers) if numbers else None


def run(run, config):
    rng = np.random.default_rng(config["seed"])
    curves, broken, lr_counts, clip_counts, skipped = _frontier(run, config, rng)
    _frontier_figures(run, curves)
    p2 = _kink(run, config, broken)
    maximum = _adversary(run, config, rng)
    lr_values = [row[1] for group in curves.values() for row in group if row[1] is not None]
    clip_values = [row[2] for group in curves.values() for row in group if row[2] is not None]
    lr_ratio, clip_ratio = _ratio(lr_values), _ratio(clip_values)
    lr_total, clip_total = sum(lr_counts.values()), sum(clip_counts.values())
    criterion = config["criteria"]["P1"]
    p1_passed = (lr_ratio is not None and lr_ratio <= criterion["normalized_lr_ratio_max"]
                 and clip_ratio is not None and clip_ratio <= criterion["clip_ratio_max"]
                 and len(lr_values) == lr_total and len(clip_values) == clip_total)
    run.record("P1_criterion", lr_reached=len(lr_values), lr_total=lr_total,
               clip_reached=len(clip_values), clip_total=clip_total,
               lr_ratio=lr_ratio, clip_ratio=clip_ratio,
               clip_c_min=min(clip_values, default=None), clip_c_max=max(clip_values, default=None),
               passed=p1_passed)
    detail = (f"X1a 在 {len(lr_values)}/{lr_total} 个理论区间点测得最小预算，"
              f"X1b 在 {len(clip_values)}/{clip_total} 点测得 c*。")
    if lr_values:
        detail += f"已测 X1a 归一化比值 {lr_ratio:.3f}（阈值 4）。"
    if clip_values:
        detail += f"已测 c* 范围 {min(clip_values)}–{max(clip_values)}，比值 {clip_ratio:.3f}（阈值 8）。"
    if lr_counts["q_cap"] or clip_counts["q_cap"]:
        detail += (f"q 上限截断 X1a 的 {lr_counts['q_cap']} 点和 X1b 的 {clip_counts['q_cap']} 点，"
                   "全网格判据缺少这些点的最小预算。")
    if clip_counts["c_grid_exhausted"]:
        detail += f"{clip_counts['c_grid_exhausted']} 个点在整个 c 网格内都未达 0.95。"
    p1 = Finding("通过" if p1_passed else "未通过", detail,
                 ("X1a_lr_frontier.png", "X1b_clip_frontier.png"))
    unsafe_rows = [row for row in run.rows if row.get("world") == "unsafe"]
    unsafe_n = sum(row["reps"] for row in unsafe_rows)
    unsafe_issues = sum(row["observed_issues"] for row in unsafe_rows)
    notes = [
        f"q>2⁶⁰ 的 X1b 候选预算共有 {skipped} 个，逐点记录在 raw.csv。"
        "超过 10¹² 期望计数的边际采用条件正态近似，整数总量守恒，稀有格保留二项抽样。",
        "X1b 用独立的 2,000 次重复评估每个运行预算。裁剪审计用 C₀=16，"
        "路径与四个中心矩分别使用一半误差预算。LR 用每世界 20,000 次重复和经验随机化阈值。",
        f"X1d 按固定 ρ 的 ε 比较 r=.5 的同族世界。各组已测最大预算位于 {maximum or '无可测点'}。"
        " [X1d_adversary_K.png](figs/X1d_adversary_K.png)",
        f"为 X6 保存 A_clip 不安全试验 {unsafe_n:,} 次和实际签发 {unsafe_issues:,} 次。",
        "按冻结口径，P1 单列可测子集比值。任何受截断的最小预算使全网格通过条件无法核验。"
        "P2 排除拐点本身并保留原网格。冻结后没有更改配置。",
    ]
    return {"P1": p1, "P2": p2}, notes
