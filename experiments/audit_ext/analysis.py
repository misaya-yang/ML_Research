"""Per-experiment task lists, analysis, figures and summaries."""

import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from experiments.audit_lab.run_io import Finding, cost_lines


def seed_of(base, *parts):
    return int(np.random.SeedSequence([base, *parts]).generate_state(1)[0])


def save_rows(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (path / "raw.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["section"])
        writer.writeheader()
        writer.writerows(rows)


def fig(path, figure, name):
    figure.tight_layout()
    figure.savefig(path / "figs" / name, dpi=150)
    plt.close(figure)


def ols(x, y):
    fit = linregress(np.log(x), np.log(y))
    return float(fit.slope), float(fit.stderr), len(x)


# ---------------------------------------------------------------- X8
def x8_tasks(config):
    return [dict(key=f"eta0={e}|r={r}|t={t}", eta0=e, r=r, t=t, reps=config["reps"],
                 seed=seed_of(config["seed"], i))
            for i, (e, r, t) in enumerate((e, r, t) for e in config["eta0_grid"]
                                         for r in config["r_grid"] for t in config["t_grid"])]


def x8_analyze(path, config, results):
    rows = [dict(section="X8", **res) for res in results.values()]
    fits, lines = [], []
    figure, ax = plt.subplots(figsize=(7.6, 4.8))
    for e in config["eta0_grid"]:
        for r in config["r_grid"]:
            cell = sorted((row for row in rows if row["eta0"] == e and row["r"] == r), key=lambda z: z["x"])
            x = np.array([c["x"] for c in cell])
            y = np.array([c["mean_cost"] for c in cell])
            ax.loglog(1 + x, y, "o-", ms=3, label=f"eta0=2^{int(math.log2(e))}, r={r}")
            use = (1 + x) <= 1 / (4 * e)
            slope, se, n = ols(1 + x[use], y[use]) if use.sum() >= 3 else (float("nan"), float("nan"), int(use.sum()))
            passed = 0.8 <= slope <= 1.2
            fits.append((e, r, slope, se, n, passed, float(y[use].min()) if n else None,
                         float(y[use].max()) if n else None))
            rows.append(dict(section="X8_fit", eta0=e, r=r, slope=slope, slope_se=se, points=n, passed=passed))
    xs = np.array([1 + row["x"] for row in rows if row["section"] == "X8" and 1 + row["x"] <= 1 / (4 * row["eta0"])])
    ys = np.array([row["mean_cost"] for row in rows if row["section"] == "X8" and 1 + row["x"] <= 1 / (4 * row["eta0"])])
    pooled = ols(xs, ys)
    rows.append(dict(section="X8_fit_pooled", slope=pooled[0], slope_se=pooled[1], points=pooled[2]))
    grid = np.geomspace(1, 2e3, 50)
    ax.loglog(grid, grid, "k:", lw=1, label="slope 1")
    ax.set(xlabel="1 + D_half/(eta0 t)", ylabel="mean target draws per two-time replicate",
           title="X8. P-jump cost")
    ax.legend(fontsize=7)
    fig(path, figure, "X8_cost.png")
    ok = all(f[5] for f in fits)
    detail = "四组斜率 " + "，".join(
        f"η₀=2^{int(math.log2(e))}, r={r}: {s:.3f}±{se:.3f}（{n} 点，成本 {lo:.1f}–{hi:.1f}）"
        for e, r, s, se, n, _, lo, hi in fits) + f"。合并斜率 {pooled[0]:.3f}（描述性）。判据 [0.8,1.2]，拟合区间 1+x≤1/(4η₀)。"
    return {"P10": Finding("通过" if ok else "未通过", detail, ("X8_cost.png",))}, rows, []


# ---------------------------------------------------------------- X9a
def x9a_tasks(config):
    specs = []
    for kind in config["random_kinds"]:
        for i in range(config["random_per_kind"]):
            specs.append(dict(family="random", kind=kind, name=f"random{kind}-{i}"))
    tl = config["three_layer"]
    for rho in tl["rho_grid"]:
        for r in tl["r_grid"]:
            for d in tl["dark_eta0"]:
                specs.append(dict(family="three_layer", rho=rho, r=r, dark_eta0=d,
                                  name=f"three_layer rho=2^{int(math.log2(rho))} r={r} dark={d}"))
    for e in config["pareto_low"]["eta0_grid"]:
        for r in config["pareto_low"]["r_grid"]:
            specs.append(dict(family="pareto_low", eta0=e, r=r, name=f"pareto_low eta0={e} r={r}"))
    tasks = []
    for i, spec in enumerate(specs):
        spec["seed"] = seed_of(config["seed"], i)
        tasks.append(dict(key=spec["name"], spec=spec, t_grid=config["t_grid"], b_rules=config["b_rules"]))
    return tasks


def x9a_analyze(path, config, results):
    rows = []
    for key, res in results.items():
        for row in res["rows"]:
            rows.append(dict(section="X9a", instance=key, family=key.split()[0].rstrip("0123456789-"), **row))
    rg = np.array([row["max_R_over_Gamma"] for row in rows])
    rx = np.array([row["max_R_over_Xi"] for row in rows])
    gx = np.array([row["Gamma_over_Xi"] for row in rows])
    worst_g = rows[int(rg.argmax())]
    worst_x = rows[int(rx.argmax())]
    figure, ax = plt.subplots(figsize=(6.4, 5.4))
    G = np.array([row["Gamma"] for row in rows])
    R = np.array([row["max_R"] for row in rows])
    for fam in sorted({row["family"] for row in rows}):
        m = np.array([row["family"] == fam for row in rows])
        ax.loglog(G[m], np.maximum(R[m], 1e-18), ".", ms=2, alpha=.5, label=fam)
    line = np.geomspace(G.min(), G.max(), 10)
    ax.loglog(line, 44 * line, "r-", lw=1, label="y=44x")
    ax.loglog(line, line, "k--", lw=1, label="y=x")
    ax.set(xlabel="Gamma_t", ylabel="max_h R_Q(b,t)", title="X9a. Cold risk vs Gamma_t")
    ax.legend(fontsize=7, markerscale=4)
    fig(path, figure, "X9a_R_vs_Gamma.png")
    passed = rg.max() <= 44 and rx.max() <= 1520
    by_family = {}
    for row in rows:
        f = row["family"]
        by_family[f] = max(by_family.get(f, 0), row["max_R_over_Gamma"])
    detail = (f"{len(rows)} 个 (实例,b,t) 情形。max_h R/Γ_t 上确界 {rg.max():.3f}"
              f"（{worst_g['instance']}, t={worst_g['t']}, b={worst_g['b']}），中位数 {np.median(rg):.3f}；"
              f"max_h R/Ξ 上确界 {rx.max():.3f}（{worst_x['instance']}, t={worst_x['t']}, b={worst_x['b']}），"
              f"中位数 {np.median(rx):.3f}。Γ_t/Ξ 最大 {gx.max():.1f}。无情形超过 44 或 1520。"
              if passed else
              f"{len(rows)} 个情形中有超过硬判据的情形：max R/Γ_t={rg.max():.3f}，max R/Ξ={rx.max():.3f}。")
    notes = ["各族 max R/Γ_t：" + "，".join(f"{k} {v:.3f}" for k, v in sorted(by_family.items())) + "。",
             "X4 样本库未建，样本库部分未运行（阶段 3）。"]
    with (path / "x9a.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"P11": Finding("通过" if passed else "未通过", detail, ("X9a_R_vs_Gamma.png",))}, rows, notes


# ---------------------------------------------------------------- X9b/X9c
def x9b_tasks(config):
    tasks = []
    for i, spec in enumerate(config["families"]):
        for t in config["t_grid"]:
            tasks.append(dict(key=f"{spec['name']}|t={t}", spec=spec, t=t, cfg=config,
                              seed=seed_of(config["seed"], i, t)))
    return tasks


def x9b_analyze(path, config, results):
    from scipy.stats import beta as beta_dist
    rows = [dict(section="X9b", **row) for res in results.values() for row in res["rows"]]
    cert = [row for row in rows if row["auditor"] == "A_gamma"]
    clip = [row for row in rows if row["auditor"] == "A_clip_paths"]
    delta = config["delta"]
    # X6 validity
    unsafe = [row for row in cert if row["unsafe"]]
    n_unsafe = sum(row["reps"] for row in unsafe)
    false_issues = sum(row["issues"] for row in unsafe)
    up = beta_dist.ppf(.975, false_issues + 1, n_unsafe - false_issues) if n_unsafe else float("nan")
    clip_unsafe = [row for row in clip if row["unsafe"]]
    clip_n = sum(row["reps"] for row in clip_unsafe)
    clip_false = sum(row["issues"] for row in clip_unsafe)
    explore_unsafe_rate = (np.mean([row["rate_C0_explore"] for row in unsafe]) if unsafe else float("nan"))
    # X9b: issue rate at max budget vs x
    pmax, qmax = max(config["p_grid"]), max(config["q_grid"])
    top = [row for row in cert if row["p"] == pmax and row["q"] == qmax]
    x = np.array([row["x"] for row in top])
    rate = np.array([row["rate"] for row in top])
    rate2 = np.array([row["rate_C0_explore"] for row in top])
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogx(x, rate, "o", ms=3, label="C0=44")
    ax.semilogx(x, rate2, "x", ms=3, alpha=.6, label="C0=2 (exploratory, not valid)")
    ax.axvline(44, color=".5", ls=":")
    ax.set(xlabel="Var_P(h) / (k Gamma_t)", ylabel=f"issue rate (p=2^{int(math.log2(pmax))}, q=2^{int(math.log2(qmax))})",
           title="X9b. Gamma-certificate issuance")
    ax.legend(fontsize=8)
    fig(path, figure, "X9b_issue_vs_x.png")
    xmax = float(x.max())
    above = x[rate >= .5]
    below = x[rate < .5]
    cross = (float(above.min()) if above.size else None, float(below[below < (above.min() if above.size else np.inf)].max())
             if below.size else None)
    # X9c: min p at q max with rate >= .8
    groups = {}
    for row in cert:
        if row["q"] != qmax:
            continue
        groups.setdefault((row["family"], row["t"], row["k"], row["h"]), []).append(row)
    mins = []
    for key, items in groups.items():
        items.sort(key=lambda z: z["p"])
        reached = [z for z in items if z["rate"] >= .8]
        if reached:
            z = reached[0]
            mins.append(dict(section="X9c_min", family=key[0], t=key[1], k=key[2], h=key[3], x=z["x"],
                             var=z["var"], p_min=z["p"], kl_over_var=z["k"] * math.log(1 / delta) / z["var"]))
    rows.extend(mins)
    fit_set = [m for m in mins if m["x"] >= 1e3]
    if len(fit_set) >= 3:
        s, se, n = ols([m["kl_over_var"] for m in fit_set], [m["p_min"] for m in fit_set])
        p12_slope = (0.8 <= s <= 1.2, f"x≥10³ 的 {n} 个实例斜率 {s:.3f}±{se:.3f}。")
    else:
        p12_slope = (None, f"x≥10³ 的实例数为 {len(fit_set)}，冻结判据无法计算。")
    desc = ""
    if len(mins) >= 3:
        s, se, n = ols([m["kl_over_var"] for m in mins], [m["p_min"] for m in mins])
        desc = f"描述性拟合（全部 {n} 个达到 80% 的实例，x 范围 {min(m['x'] for m in mins):.0f}–{max(m['x'] for m in mins):.0f}）斜率 {s:.3f}±{se:.3f}。"
        rows.append(dict(section="X9c_fit_descriptive", slope=s, slope_se=se, points=n))
    figure, ax = plt.subplots(figsize=(6.4, 4.6))
    if mins:
        ax.loglog([m["kl_over_var"] for m in mins], [m["p_min"] for m in mins], "o", ms=4)
    ax.set(xlabel="k log(1/delta) / Var_P(h)", ylabel="minimum p for 80% issue (q=2^18)", title="X9c. Target cost")
    fig(path, figure, "X9c_min_p.png")
    # comparator: min total samples to 80%
    comp = []
    for auditor, subset, qkey in (("A_gamma", [r for r in cert if r["k"] == config["compare"]["k"] and r["t"] in config["compare"]["t_grid"]], "q"),
                                  ("A_clip_paths", clip, "q_actual")):
        best = {}
        for row in subset:
            if row["rate"] >= .8:
                key = (row["family"], row["t"], row["h"])
                total = row["p"] + row[qkey]
                best[key] = min(best.get(key, math.inf), total)
        comp.append((auditor, best))
    common = set(comp[0][1]) & set(comp[1][1])
    ratio = [comp[1][1][key] / comp[0][1][key] for key in common]
    for key in sorted(common):
        rows.append(dict(section="X9_compare", family=key[0], t=key[1], h=key[2],
                         total_A_gamma=comp[0][1][key], total_A_clip_paths=comp[1][1][key]))
    validity_ok = false_issues == 0
    detail = (f"有效性：不安全 (h,k,t) 试验 {n_unsafe:,} 次，A_gamma 误发 {false_issues} 次"
              f"（95% CP 上界 {up:.2e}），记入 X6。")
    if p12_slope[0] is None:
        status = "未运行"
        detail += ("斜率判据未运行：" + p12_slope[1] +
                   f"本网格上 x=Var/(kΓ_t) 实测最大 {xmax:.1f}。原因：t≥2 时 E_P min(W,t)≥1，"
                   "故 Γ_t≥1/t、x≤t/(4k)≤512（t≤2¹²），计划设想的 [1,10⁴] 覆盖与 x≥10³ 拟合集在冻结网格上不可达。")
    else:
        status = "通过" if (p12_slope[0] and validity_ok) else "未通过"
        detail += p12_slope[1]
    detail += desc
    if cross[0] is not None:
        detail += f"最大预算下签发率≥0.5 的最小 x 为 {cross[0]:.1f}。"
    else:
        detail += "最大预算下没有实例的签发率达到 0.5。"
    notes = [f"A_clip 加冷路径对照：不安全试验 {clip_n:,} 次误发 {clip_false} 次。两者都在 80% 签发的共同实例 {len(common)} 个，"
             + (f"A_clip 总抽样数 / A_gamma 总抽样数的中位数 {np.median(ratio):.3g}（范围 {min(ratio):.3g}–{max(ratio):.3g}）。" if ratio else "无共同实例。")
             + " 只描述。",
             f"探索变体 C0=2（理论上不保证有效，不进 X6）：不安全格平均签发率 {explore_unsafe_rate:.4f}。",
             "X4 样本库未建，样本库部分未运行（阶段 3）。"]
    findings = {"P12": Finding(status, detail, ("X9b_issue_vs_x.png", "X9c_min_p.png"))}
    x6 = dict(A_gamma_unsafe_trials=n_unsafe, A_gamma_false=false_issues, A_clip_unsafe_trials=clip_n,
              A_clip_false=clip_false)
    rows.append(dict(section="X6_tally", **x6))
    return findings, rows, notes


# ---------------------------------------------------------------- X10
def x10_tasks(config):
    from .x10 import sigma_K
    a, b = config["x10a"], config["x10b"]
    tasks = []
    for k in a["k_grid"]:
        for beta in a["beta_grid"]:
            for c1, c2 in a["c_grid"]:
                if sigma_K(k, beta, c1, c2) <= 0:
                    continue
                for t in a["t_grid"]:
                    for rho in a["rho_grid"]:
                        tasks.append(dict(key=f"a|k={k}|beta={beta}|c=({c1},{c2})|t={t}|rho={rho}", part="a",
                                          k=k, beta=beta, c1=c1, c2=c2, t=t, rho=rho, cfg=a))
    for i, t in enumerate(b["t_grid"]):
        for j, rho in enumerate(b["rho_grid"]):
            tasks.append(dict(key=f"b|t={t}|rho={rho}", part="b", k=b["k"], beta=b["beta"], c1=b["c"][0],
                              c2=b["c"][1], t=t, rho=rho, cfg=b, seed=seed_of(config["seed"], i, j)))
    return tasks


def x10_analyze(path, config, results):
    from .x10 import sigma_K
    a_cfg = config["x10a"]
    rows_a = [dict(section="X10a", **res["row"]) for key, res in results.items() if key.startswith("a|")]
    rows_b = [dict(section="X10b", **res["row"]) for key, res in results.items() if key.startswith("b|")]
    kept = sorted({(r["k"], r["beta"], r["c1"], r["c2"]) for r in rows_a})
    excluded = [(k, beta, c1, c2) for k in a_cfg["k_grid"] for beta in a_cfg["beta_grid"]
                for c1, c2 in a_cfg["c_grid"] if sigma_K(k, beta, c1, c2) <= 0]
    bad = []
    for r in rows_a:
        fails = []
        if r["status"] != "ok":
            fails.append(r["status"])
        else:
            if max(r["inv_W_law"], r["inv_Ph"], r["inv_Qh"]) > 1e-12:
                fails.append("invariant")
            if not (r["G_safe_over_eps"] <= -1 and r["G_unsafe_over_eps"] >= 1):
                fails.append("crossing")
            if r["quad_rel"] > 1e-10:
                fails.append("quadratic")
        r["fails"] = ",".join(fails)
        if fails:
            bad.append(r)
    ok_rows = [r for r in rows_a if r["status"] == "ok"]
    a_pass = not bad
    by_reason = {}
    for r in bad:
        for reason in r["fails"].split(","):
            by_reason[reason] = by_reason.get(reason, 0) + 1
    # X10b slopes
    good_b = [r for r in rows_b if r.get("status") == "ok" and r.get("q_min") and r.get("p_min")]
    b_pass, b_detail = False, "X10b 无可用点。"
    if len(good_b) >= 4:
        lr, lt = np.log([r["rho"] for r in good_b]), np.log([r["t"] for r in good_b])
        X = np.column_stack([np.ones(len(good_b)), lr, lt])
        coef_q, *_ = np.linalg.lstsq(X, np.log([r["q_min"] for r in good_b]), rcond=None)
        sp, _, _ = ols([r["rho"] for r in good_b], [r["p_min"] for r in good_b])
        b_pass = all(0.8 <= v <= 1.2 for v in (coef_q[1], coef_q[2], sp)) and len(good_b) == len(rows_b)
        b_detail = (f"X10b {len(good_b)}/{len(rows_b)} 点：最小 q 对 ρ 斜率 {coef_q[1]:.3f}、对 t 斜率 {coef_q[2]:.3f}；"
                    f"最小 p 对 ρ 斜率 {sp:.3f}。")
        rows_b.append(dict(section="X10b_fit", q_rho_slope=coef_q[1], q_t_slope=coef_q[2], p_rho_slope=sp))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for r in ok_rows:
        axes[0].plot(r["rho"], r["G_safe_over_eps"], "b.", ms=3)
        axes[0].plot(r["rho"], r["G_unsafe_over_eps"], "r.", ms=3)
    axes[0].axhline(1, color=".5", ls=":"); axes[0].axhline(-1, color=".5", ls=":")
    axes[0].set(xscale="log", xlabel="rho", ylabel="G / eps", title="X10a crossings")
    axes[1].semilogy([r["t"] for r in ok_rows], [max(r["quad_rel"], 1e-18) for r in ok_rows], ".", ms=3)
    axes[1].axhline(1e-10, color="r", ls=":")
    axes[1].set(xscale="log", xlabel="t", ylabel="quadratic residual / max|G|", title="X10a quadratic structure")
    for t in sorted({r["t"] for r in good_b}):
        pts = sorted((r for r in good_b if r["t"] == t), key=lambda z: z["rho"])
        axes[2].loglog([z["rho"] for z in pts], [z["q_min"] / t for z in pts], "o-", ms=3, label=f"q/t, t=2^{int(math.log2(t))}")
    pts = sorted((r for r in good_b if r["t"] == min(r2["t"] for r2 in good_b)), key=lambda z: z["rho"]) if good_b else []
    axes[2].loglog([z["rho"] for z in pts], [z["p_min"] for z in pts], "ks--", ms=4, label="p (q=0)")
    axes[2].set(xlabel="rho", ylabel="minimum budget", title="X10b LR budgets")
    axes[2].legend(fontsize=6)
    fig(path, figure, "X10.png")
    if ok_rows:
        kp = [r["KL_P_const"] for r in ok_rows]; kq = [r["KL_Q_const"] for r in ok_rows]
        stats = (f"X10a 保留 {len(kept)} 个 (k,β,c) 组合（条件 K 排除 {len(excluded)} 个），共 {len(rows_a)} 个格点，"
                 f"未过 {len(bad)} 个" + (f"（{', '.join(f'{k} {v}' for k, v in by_reason.items())}）" if bad else "") +
                 f"。不变量最大差 {max(max(r['inv_W_law'], r['inv_Ph'], r['inv_Qh']) for r in ok_rows):.1e}；"
                 f"G/ε 安全侧最大 {max(r['G_safe_over_eps'] for r in ok_rows):.3f}、不安全侧最小 {min(r['G_unsafe_over_eps'] for r in ok_rows):.3f}；"
                 f"二次残差相对值最大 {max(r['quad_rel'] for r in ok_rows):.2e}；R/ρ 对 Φ̄/2 的比值范围 "
                 f"{min(r['R_over_rho'] / r['Phibar_half'] for r in ok_rows):.3f}–{max(r['R_over_rho'] / r['Phibar_half'] for r in ok_rows):.3f}；"
                 f"KL_P 常数 {min(kp):.0f}–{max(kp):.0f}，KL_Q 常数 {min(kq):.0f}–{max(kq):.0f}。")
    else:
        stats = "X10a 无可用点。"
    status = "通过" if (a_pass and b_pass) else "未通过"
    finding = Finding(status, stats + b_detail, ("X10.png",))
    notes = ["条件 K 排除的组合：" + ("；".join(f"k={k},β={b},c=({c1},{c2})" for k, b, c1, c2 in excluded) or "无") + "。",
             "θ* 由 9 点精确二次拟合的根加 3 步精确牛顿修正得到；交叉检验在 θ*∓Δ 处精确计算。"]
    if bad:
        notes.append("未过格点（前 20 个）：" + "；".join(
            f"k={r['k']},β={r['beta']},c=({r['c1']},{r['c2']}),t={r['t']},ρ={r['rho']}: {r['fails']}"
            + (f" G/ε=({r.get('G_safe_over_eps', float('nan')):.3f},{r.get('G_unsafe_over_eps', float('nan')):.3f})"
               if r["status"] == "ok" else "") for r in bad[:20]))
    return {"P13": finding}, rows_a + rows_b, notes


# ---------------------------------------------------------------- X1e
def x1e_tasks(config):
    lo, sat = config["lowoverlap"], config["saturation"]
    tasks = []
    i = 0
    for r in lo["r_grid"]:
        for rho in lo["rho_grid"]:
            for e in lo["eta0_grid"]:
                tasks.append(dict(key=f"low|r={r}|rho={rho}|eta0={e}", part="low", r=r, rho=rho, eta0=e,
                                  cfg=lo, seed=seed_of(config["seed"], 1, i)))
                i += 1
    for j, rho in enumerate(sat["rho_grid"]):
        tasks.append(dict(key=f"sat|rho={rho}", part="sat", rho=rho, cfg=sat, seed=seed_of(config["seed"], 2, j)))
    return tasks


def x1e_analyze(path, config, results):
    lo, sat = config["lowoverlap"], config["saturation"]
    rows = []
    for key, res in results.items():
        for row in res["rows"]:
            rows.append(dict(section="X1e_" + key.split("|")[0], **row))
    mins = [r for r in rows if r["section"] == "X1e_low" and r["kind"] == "minimum"]
    calfail = [r for r in rows if r["section"] == "X1e_low" and r["kind"] == "calibration_failed"]
    point_lines, pass_points, all_points = [], 0, 0
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in lo["r_grid"]:
        for rho in lo["rho_grid"]:
            scope = all(lo["J"] / e <= rho / 2 for e in lo["eta0_grid"] if e < 1)
            cs = {}
            for m in mins:
                if m["r"] == r and m["rho"] == rho:
                    cs[(m["auditor"], m["eta0"])] = m["c_star"]
            small = [cs.get(("A_clip_grid", e)) for e in lo["eta0_grid"] if e < 1]
            ok = all(c is not None for c in small) and max(small) / min(small) <= 4
            if scope:
                all_points += 1
                pass_points += ok
            fmt = lambda c: "截断" if c is None else f"{c:g}"
            point_lines.append(f"r={r}, ρ=2^{int(math.log2(rho))}: grid c*=" +
                               "/".join(fmt(cs.get(("A_clip_grid", e))) for e in lo["eta0_grid"]) +
                               "，A_clip c*=" + "/".join(fmt(cs.get(("A_clip", e))) for e in lo["eta0_grid"]))
            rows.append(dict(section="X1e_point", r=r, rho=rho, in_scope=scope, passed=ok,
                             **{f"{a}_eta0={e}": cs.get((a, e)) for a in ("A_clip_grid", "A_clip") for e in lo["eta0_grid"]}))
    for idx, aud in enumerate(("A_clip_grid", "A_clip")):
        for e in lo["eta0_grid"]:
            pts = sorted((m for m in mins if m["auditor"] == aud and m["eta0"] == e and m["c_star"]),
                         key=lambda z: (z["r"], z["rho"]))
            axes[idx].plot(range(len(pts)), [m["c_star"] for m in pts], "o-", ms=4,
                           label=f"eta0=2^{int(math.log2(e))}" if e < 1 else "eta0=1")
        axes[idx].set(yscale="log", xlabel="design point (r, rho) index", ylabel="c*", title=f"X1e {aud}")
        axes[idx].legend(fontsize=7)
    fig(path, figure, "X1e_lowoverlap.png")
    low_pass = all_points > 0 and pass_points == all_points and not calfail
    # saturation
    smins = sorted((r for r in rows if r["section"] == "X1e_sat" and r["kind"] == "minimum"), key=lambda z: z["rho"])
    sfail = [r for r in rows if r["section"] == "X1e_sat" and r["kind"] == "calibration_failed"]
    eps = sat["epsilon"]
    tail = [r for r in smins if r["psi"] <= eps / 4]
    sat_pass, sat_detail = False, ""
    if tail and all(r["status"] == "reached" for r in tail) and len(tail) >= 3:
        vals = [r["p_eps_over_log"] for r in tail]
        s, se, n = ols([r["psi"] for r in tail], [r["p_min"] for r in tail])
        sat_pass = max(vals) / min(vals) <= 2 and -0.2 <= s <= 0.2
        sat_detail = (f"饱和：Ψ≤ε/4 的 {n} 点 p_min·ε/log(1/δ) 在 {min(vals):.2f}–{max(vals):.2f}，"
                      f"对 log Ψ 斜率 {s:.3f}±{se:.3f}。")
    else:
        sat_detail = f"饱和：Ψ≤ε/4 的点 {len(tail)} 个，达到者 {sum(r['status'] == 'reached' for r in tail)} 个。"
    figure, ax = plt.subplots(figsize=(6.8, 4.6))
    reached = [r for r in smins if r["status"] == "reached"]
    ax.loglog([r["psi_over_eps"] for r in reached], [r["p_eps_over_log"] for r in reached], "o-", label="A_clip_grid p_min eps/log(1/delta)")
    ax.loglog([r["psi_over_eps"] for r in reached], [r["A_psi"] * eps / math.log(1 / sat["delta"]) for r in reached], "k:", label="Psi/eps (A Psi scale)")
    ax.axvline(.25, color=".5", ls=":")
    ax.set(xlabel="Psi(rho)/eps", ylabel="p eps / log(1/delta)", title="X1e saturation (r=.5)")
    ax.legend(fontsize=7)
    fig(path, figure, "X1e_saturation.png")
    unsafe_issues = sum(r.get("unsafe_issues", 0) or 0 for r in rows if r.get("kind") == "probe")
    unsafe_trials = sum(r.get("reps", 0) or 0 for r in rows if r.get("kind") == "probe")
    detail = (f"小重叠：判据范围内 {pass_points}/{all_points} 个设计点 A_clip_grid 的 c*（η₀=2⁻⁸/2⁻¹²）比值≤4。"
              + ("；".join(point_lines) + "（顺序 η₀=1/2⁻⁸/2⁻¹²）。")
              + (f"标定失败 {len(calfail)} 点。" if calfail else "") + sat_detail
              + (f"饱和标定失败 {len(sfail)} 点。" if sfail else ""))
    status = "通过" if (low_pass and sat_pass) else "未通过"
    notes = [f"不安全世界试验 {unsafe_trials:,} 次（两个审计器合计），签发 {unsafe_issues} 次，记入 X6。",
             "c* 以原族 Ψ(ρ) 归一：p=c·Ψ(ρ)·log(1/δ)/ε²，ε=Ψ(ρ)/2000，与 X1b 相同。",
             "饱和部分 ε=2⁻⁶ 固定，世界在 θ=1/2 调 β 使精确 G_k=∓ε，c 网格步长 2^{1/4}。"]
    return {"P9": Finding(status, detail, ("X1e_lowoverlap.png", "X1e_saturation.png"))}, rows, notes


# ---------------------------------------------------------------- X11
def x11_tasks(config):
    a, b = config["x11a"], config["x11b"]
    tasks = []
    i = 0
    for c in a["c_grid"]:
        for t in a["t_grid"]:
            for rho in a["rho_grid"]:
                for fr in a["fr_grid"]:
                    tasks.append(dict(key=f"c={tuple(c)}|t={t}|rho={rho}|fr={fr}", c=tuple(c), t=t, rho=rho, fr=fr,
                                      cfg=a, x11b=b, seed=seed_of(config["seed"], i)))
                    i += 1
    # costliest (largest t) first
    tasks.sort(key=lambda z: -z["t"])
    return tasks


def x11_analyze(path, config, results):
    rows = [dict(section="X11a", **row) for res in results.values() for row in res["rows"]]
    ok = [r for r in rows if r["status"] == "ok"]
    na = [r for r in rows if r["status"] == "n/a"]
    infeasible = [r for r in rows if r["status"] == "tilt_infeasible"]
    hard_bad = [r for r in ok if not (r["inv"] <= 1e-12 and r["R_rel_diff"] <= 1e-12 and r["cs_ratio"] <= 1
                                      and r["excess_over_elow"] <= 1)]
    soft = [r for r in ok if r["soft_row"]]

    def soft_fail(r):
        f = []
        if not .85 <= r["a_over_pred"] <= 1.15:
            f.append("a_U/pred")
        for pre in ("S", "V"):
            gm, gp = r[f"{pre}_Gm_over_eps"], r[f"{pre}_Gp_over_eps"]
            if not (gm * gp < 0 and min(abs(gm), abs(gp)) >= 1 - 1e-12):
                f.append(f"{pre}-crossing")
        for key in ("KL_P_d_over_D2", "KL_Q_d_over_D2"):
            if not 8.7 <= r[key] <= 9.2:
                f.append(key)
        return f

    soft_bad = [(r, soft_fail(r)) for r in soft if soft_fail(r)]
    groups = {}
    for r in soft:
        groups.setdefault((r["k"], r["c1"], r["c2"], r["rho"], r["rB_over_rho"]), []).append(r)
    slopes = []
    for key, items in sorted(groups.items()):
        items = [z for z in items if z["K"] > 0]
        if len(items) >= 3:
            s, se, n = ols([z["t"] for z in items], [z["K"] for z in items])
            slopes.append((key, s, n))
            rows.append(dict(section="X11a_K_slope", k=key[0], c1=key[1], c2=key[2], rho=key[3], rB_over_rho=key[4],
                             slope=s, slope_se=se, points=n, passed=.8 <= s <= 1.2))
    slope_bad = [x for x in slopes if not .8 <= x[1] <= 1.2]
    hard_pass = not hard_bad and not infeasible
    soft_pass = not soft_bad and not slope_bad and bool(slopes)
    # X11b
    xb = [r for r in ok if "x11b_n_S" in r]
    xb_detail = ""
    if xb:
        ratios = [r.get("x11b_ratio_over_K") for r in xb]
        within = all(v is not None and .5 <= v <= 2 for v in ratios)
        sl = []
        for fr in config["x11b"]["fr_grid"]:
            pts = sorted((r for r in xb if r["rB_over_rho"] == fr and r.get("x11b_ratio")), key=lambda z: z["t"])
            if len(pts) >= 3:
                sl.append((fr, ols([z["t"] for z in pts], [z["x11b_ratio"] for z in pts])[0]))
        xb_ok = within and sl and all(.8 <= s <= 1.2 for _, s in sl)
        xb_detail = (f"X11b（可选）{len(xb)} 点：比值/K 范围 {min(v for v in ratios if v):.2f}–{max(v for v in ratios if v):.2f}；"
                     + "，".join(f"ρ_B/ρ={fr:g} 时比值对 t 斜率 {s:.3f}" for fr, s in sl)
                     + f"；X11b 部分{'通过' if xb_ok else '未通过'}。")
    # figure
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for key, items in sorted(groups.items()):
        items = sorted(items, key=lambda z: z["t"])
        axes[0].loglog([z["t"] for z in items], [max(z["K"], 1e-6) for z in items], "o-", ms=3,
                       label=f"k={key[0]} c=({key[1]},{key[2]}) rho={key[3]} rB/rho={key[4]:g}")
    axes[0].set(xlabel="t", ylabel="gain K", title="X11a gain K (soft rows)")
    axes[0].legend(fontsize=5)
    axes[1].plot([r["t"] for r in ok], [r["a_over_pred"] for r in ok], ".", ms=4)
    axes[1].axhspan(.85, 1.15, color=".9")
    axes[1].set(xscale="log", xlabel="t", ylabel="a_U / pred", title="X11a startup slope")
    fig(path, figure, "X11.png")
    desc = [r for r in ok if abs(r["rB_over_rho"] - 1 / 16) < 1e-12]
    detail = (f"共 {len(rows) - len(slopes)} 行：ok {len(ok)}，n/a {len(na)}（不算失败），倾斜越界 {len(infeasible)}。"
              f"硬判据 {'全部通过' if hard_pass else f'未过 {len(hard_bad)} 行'}：不变量最大差 "
              f"{max((r['inv'] for r in ok), default=float('nan')):.1e}，R 相对差最大 {max((r['R_rel_diff'] for r in ok), default=float('nan')):.1e}，"
              f"CS 比值最大 {max((r['cs_ratio'] for r in ok), default=float('nan')):.3f}，低层超额/e_b 最大 "
              f"{max((r['excess_over_elow'] for r in ok), default=float('nan')):.3f}。"
              f"软判据（ρ_B≥ρ/2 的 {len(soft)} 行）未过 {len(soft_bad)} 行；"
              + (f"a_U/pred 范围 {min(r['a_over_pred'] for r in soft):.3f}–{max(r['a_over_pred'] for r in soft):.3f}，"
                 f"KL_P·d/Δ² {min(r['KL_P_d_over_D2'] for r in soft):.2f}–{max(r['KL_P_d_over_D2'] for r in soft):.2f}，"
                 f"KL_Q·d/Δ² {min(r['KL_Q_d_over_D2'] for r in soft):.2f}–{max(r['KL_Q_d_over_D2'] for r in soft):.2f}；" if soft else "")
              + (f"K 对 t 斜率 {len(slopes)} 组，范围 {min(s for _, s, _ in slopes):.3f}–{max(s for _, s, _ in slopes):.3f}，未过 {len(slope_bad)} 组。" if slopes else "K 斜率无可拟合组。")
              + xb_detail)
    reasons = {}
    for r, f in soft_bad:
        for x in f:
            reasons[x] = reasons.get(x, 0) + 1
    notes = [f"ρ_B=ρ/16 描述行 {len(desc)} 个：K 范围 " +
             (f"{min(r['K'] for r in desc):.3g}–{max(r['K'] for r in desc):.3g}，a_U/pred {min(r['a_over_pred'] for r in desc):.2f}–{max(r['a_over_pred'] for r in desc):.2f}。" if desc else "无。"),
             "软判据未过原因计数：" + ("，".join(f"{k} {v}" for k, v in reasons.items()) or "无") + "。",
             "n/a 原因：" + "，".join(f"{k} {v}" for k, v in
                                     {r['reason']: sum(z['reason'] == r['reason'] for z in na) for r in na}.items()) + "。"
             if na else "无 n/a 行。"]
    if hard_bad:
        notes.append("硬判据未过（前 15 行）：" + "；".join(
            f"k={r['k']},c=({r['c1']},{r['c2']}),t={r['t']},ρ={r['rho']},ρ_B/ρ={r['rB_over_rho']:g}: inv={r['inv']:.1e}, "
            f"Rrel={r['R_rel_diff']:.1e}, CS={r['cs_ratio']:.3f}, exc={r['excess_over_elow']:.3f}" for r in hard_bad[:15]))
    if soft_bad:
        notes.append("软判据未过（前 15 行）：" + "；".join(
            f"k={r['k']},c=({r['c1']},{r['c2']}),t={r['t']},ρ={r['rho']},ρ_B/ρ={r['rB_over_rho']:g}: {','.join(f)}"
            for r, f in soft_bad[:15]))
    if slope_bad:
        notes.append("K 斜率未过：" + "；".join(f"k={k[0]},c=({k[1]},{k[2]}),ρ={k[3]},ρ_B/ρ={k[4]:g}: {s:.3f}（{n} 点）"
                                            for k, s, n in slope_bad))
    status = "通过" if hard_pass and soft_pass else "未通过"
    return {"P14": Finding(status, detail, ("X11.png",))}, rows, notes
