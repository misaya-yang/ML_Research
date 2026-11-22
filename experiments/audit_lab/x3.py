"""X3: exact finite-window calibration at each prespecified k."""

import math

import matplotlib.pyplot as plt
import numpy as np

from .bench import save_figure
from .families import ThreeLayers, calibrated_pair, signed_limit_root
from .run_io import Finding


def run(run, config):
    records = []
    for k in config["k_grid"]:
        run.guard.check()
        t = round(2 * k)
        J = 128 * k * (config["b"] + t)
        K = config["K_over_J"] * J
        psi = (J / K)**config["profile_r"]
        epsilon = psi / config["eps_divisor"]
        limit = (k + 1)**(-.5)
        root = signed_limit_root(k, config["floor_t"])
        meta = dict(k=k, t=t, J=J, K=K, psi=psi, epsilon=epsilon, center_root=root,
                    limit_root=limit, root_abs_error=abs(root - limit))
        family = ThreeLayers(J, K, psi / config["mass_divisor"], config["bulk_mass"])
        try:
            worlds = calibrated_pair(family, config["b"], t, k, epsilon,
                                     max_tilt=config["max_tilt"],
                                     tilt_tolerance=config["tilt_tolerance"])
        except ValueError as error:
            run.record("X3", passed=False, error_message=str(error), **meta)
            continue
        passed = worlds.g_safe <= -epsilon and worlds.g_unsafe >= epsilon
        scaled = worlds.bracket * math.sqrt(k)
        run.record("X3", beta=worlds.beta, s=worlds.s, m=family.m,
                   C_tilt=worlds.s * family.m / epsilon, bracket=worlds.bracket,
                   bracket_times_sqrt_k=scaled, G_safe=worlds.g_safe,
                   G_unsafe=worlds.g_unsafe, passed=passed, **meta)
        records.append((k, worlds.g_safe / epsilon, worlds.g_unsafe / epsilon, scaled, passed))

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.4))
    for index, label, color in ((1, "Safe", "C0"), (2, "Unsafe", "C1")):
        axes[0].plot([row[0] for row in records], [row[index] for row in records],
                     "o-", color=color, label=label)
    for line in (-1, 1):
        axes[0].axhline(line, color=".5", linestyle=":", linewidth=1)
    axes[0].legend()
    axes[0].set(xlabel="k", ylabel="Exact G / epsilon", title="Gate crossings")
    axes[1].plot([row[0] for row in records], [row[3] for row in records], "o-")
    axes[1].set(xlabel="k", ylabel="Bracket × sqrt(k)", title="Scaled bracket")
    save_figure(run, fig, "X3_k.png")

    count = sum(bool(row[-1]) for row in records)
    errors = [row["root_abs_error"] for row in run.rows]
    detail = (f"{count}/{len(config['k_grid'])} 个 k 的两个世界均跨过 ±ε，"
              f"中心根与极限最大差 {max(errors):.3g}。")
    if records:
        scaled = np.asarray([row[3] for row in records])
        detail += f"括号项乘 √k 落在 [{scaled.min():.4f}, {scaled.max():.4f}]。"
    notes = [
        "三层族取 K=16J、Ψ=(J/K)^0.5、m=Ψ/12、ε=Ψ/2000。β 在 [0.5,1] 内标定，"
        "在证明的 s≤0.072 区间内二分到跨阈值的最小上界。括号项按安全到不安全世界的方向计算。",
        "中心根使用 T=2¹⁶ 的有限 T 公式。P7 只在冻结的 k 网格上判定。冻结后没有更改配置。",
    ]
    state = "通过" if count == len(config["k_grid"]) else "未通过"
    return {"P7": Finding(state, detail, ("X3_k.png",))}, notes
