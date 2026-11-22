# 实验计划（交给实验 agent 执行）

2026-11-22，Claude 根据理论笔记编写。每项实验都对应一条理论预测。代码只给伪代码，由实验 agent 自行实现。运行规则以 [EXPERIMENTS_README.md](../EXPERIMENTS_README.md) 为准，本文件只补充实验内容。

理论来源：
- [Theorem A 尾部轮廓前沿](../theory/proofs/tail_profile_frontier.md)
- [固定 p 扩展](../theory/proofs/fixed_target_extensions.md)
- [三通道总览](../theory/proofs/three_channels.md)
- [Pro 矩前沿](../theory/proofs/pro_moment_frontier_original.md)
- [固定 p 原定理](../theory/proofs/fixed_target_power.md)

## 0. 规则

1. **先冻结后运行。** 每个实验先写 `config.json`，包含参数网格、种子、成功判据，以及该文件的 sha256。冻结后不得改网格或判据。确需修改时另起一个 run，并在 summary 里写明原因。
2. **输出位置。** 输出只写 `results/local_runs/<实验号>-<YYYYMMDD>-XXXX/`，内容为 `config.json`、`raw.csv`、`summary.md`、`figs/`。不得覆盖 `results/` 下已冻结的目录。
3. **结果照实报告。** 失败、与理论不符、跑不完的部分都要报告。不许为了让判据通过而调参。不主张"方法更优"。
4. **成本记账。** 目标样本、提议样本、点名评分分开计数。LLM 部分额外记录生成条数、拒绝采样尝试次数和 GPU 时长。
5. **本机资源。** 一次只跑一个作业；线程数（OPENBLAS/OMP/MKL/torch）≤2；用 `nice -n 19`；保留 ≥6GB 空闲内存。
6. **Kaggle。** 按 `~/kaggle_t4_template/README.md` 操作：必须私有，先 SMOKE 再全量，每 10 分钟写 checkpoint。每周额度 30h，每次会话 ≤9h，本计划 GPU 总预算 ≤12h。
7. 不提交、不推送 git；不安装新包，除非用户要求。Kaggle 镜像自带 torch/transformers。

## 1. 理论预测与判据

| 编号 | 预测 | 来源 | 实验 | 通过判据（冻结） |
|---|---|---|---|---|
| P1 | 中心通道前沿 p≍AΨ(q/p)，同一审计器对任何 Ψ 都成立 | Theorem A | X1 | 见 X1 |
| P2 | 前沿在尾部拐点处弯折，斜率 −r/(1−r) | Theorem A 推论 3 | X1c | 两段拟合斜率的 95% 置信区间分别覆盖 −1/3 和 −3 |
| P3 | 固定 p 时功效上限 δ/(1−κ)^p，e_p 程序可达 | B1 | X2a、X2b | q 最大时 \|功效−Π_p\| ≤ 3 个 MC 标准误 |
| P4 | 逐点 ε⁻¹ 对一致 ε⁻² | B2 | X2d | 两条线的对数斜率分别落在 [−1.15,−0.85] 和 [−2.3,−1.7] |
| P5 | 已知 Z 时 p=0 也能让功效趋于 1；Z 未知时 p=0 的功效 ≤δ | B3 | X2e、X4a | 已知 Z 时功效 ≥0.9（q 最大处）；未知 Z 时任何 q 下功效 ≤δ+3SE |
| P6 | 逼近上限的速率：缺口 ≤C(M/q^r)^{1/(1+r)} | B4 | X2c | 拟合斜率 ≤ −r/(1+r)+0.1（定理只给上界，实测可以更陡） |
| P7 | 一般 k 的构造成立（中心根 μ_*=(k+1)^{−1/2}、三层族标定、括号项 ≥c/√k） | Theorem A、B2 | X3 | 每个 k 的两个世界都跨过 ±ε |
| P8 | 有效性：误发率 ≤δ | 所有审计器 | X6 | 合并误发率的 95% 上界 ≤δ |
| P9 | 小重叠下中心前沿不变（网格证书）；q 很大时前沿饱和到 ε⁻¹ | E1、E1′（small_overlap_joint_frontier.md） | X1e | 见 X1e |
| P10 | P 跳跃的单次成本 ≈C·min{1/η₀, 1+D_{1/2}/(η₀t)}；β 扫描跨越 D_{1/2}≈η₀t 时成本骤降 | E4 | X8 | 实测成本对 1+D_{1/2}/(η₀t) 的对数斜率在 [0.8,1.2] 内（在 1/η₀ 截断之前） |
| P11 | 冷启动风险上界：对所有 h，R_Q(b,t)≤44Γ_t（Γ_t=E_P min(W,t)/t）；精化版 R≤1520·Ξ_{b,t}（Ξ=E_P min(W/t,1,L/W)+τ_P(L)²） | Lemma R、R′（cold_risk_bound.md） | X9a | 硬判据：没有任何情形超过 44 或 1520。另报实测上确界（本机已测约 1.4） |
| P12 | Γ-证书处处有效且不跑链；Var_P(h)/(kΓ_t) 足够大时签发；达到 80% 功效所需的 p 与 k/Var 成正比 | 推论 R2 | X9b、X9c | 误发率计入 X6（判据 P8）；最小 p 对 k/Var 的对数斜率在 [0.8,1.2] 内（只用 Var/(kΓ_t)≥10³ 的实例拟合；转折点预计在 10²–10³ 之间，因为 C₀=44 且置信界有放大） |
| P13 | 轻窗口动力学下界族：两个世界的 W 分布、h 边际、μ、Var 完全相同，G_k 分别为 ∓ε；G_k(θ) 恰为二次多项式；最优检验所需 q ∝ ρt/ε²、p ∝ ρ/ε² | Theorem V（variance_scaled_floor.md） | X10a、X10b | 不变量差 ≤1e-12；\|G\|≥ε；二次残差 ≤1e-10×\|G\| 的尺度；最小 q 对 ρ、对 t 的对数斜率，以及最小 p 对 ρ 的对数斜率，都在 [0.8,1.2] 内 |
| P14 | 小重叠下界：(i) 同一 W 分布、同一尾类、η=(2−ρ)/d→0 时两个世界的 μ、Q(h=1) 不变，G_k 跨门槛；(ii) 启动层标签见证 S 的斜率 a_U 符合闭式；(iii) Proposition N 的两步不等式在随机补偿标签对上成立；(iv) 增益 K 随 t 线性增长 | variance_scaled_floor.md「Small overlap」节：Lemma F、Theorem V_η、Proposition N、Theorem S | X11a、X11b | 硬判据：不变量差 ≤1e-12；二次型 R 与显式格子链相对差 ≤1e-12；CS 比值 ≤1；低层超额/e_b ≤1。软判据（只在 ρ_B≥ρ/2 且能校准的行）：a_U/pred∈[0.85,1.15]；G∓/ε 异号且绝对值 ≥1；KL·d/Δ²∈[8.7,9.2]（P、Q 两侧）；固定 (ρ,ρ_B/ρ) 时 K 对 t 的对数斜率∈[0.8,1.2] |

## 2. 公共组件（伪代码）

### 2.1 有限支撑 IMH 的精确路径矩

适用于离散族和 LLM 样本库（bank）。转移只依赖权重，所以按 w 排序后，用前缀和就能在 O(N) 时间内做一步。

```python
def exact_path_moments(w, qm, h, b, t):
    # w: 归一化权重 dP/dQ（只差常数倍也可以，接受率只用比值）；qm: Q 质量；h ∈ [0,1]
    o = argsort(w); w, qm, h = w[o], qm[o], h[o]
    alpha = Qmass(w_y >= w_x) + (1/w_x) * sum_{w_y < w_x} qm_y w_y      # 用 searchsorted 处理并列
    def K(g):    # (Kg)(x) = Σ_y qm_y min(1, w_y/w_x) g_y + (1-alpha_x) g_x
        return suffix_sum(qm*g)[first idx with w_y>=w_x] + prefix_sum(qm*w*g)[w_y<w_x]/w + (1-alpha)*g
    def KT(pi):  # (πK)(y) = qm_y Σ_x π_x min(1, w_y/w_x) + π_y (1-alpha_y)
        return qm*(prefix_sum(pi)[w_x<=w_y] + w*suffix_sum(pi/w)[w_x>w_y]) + pi*(1-alpha)
    pis = [qm];  for i in 1..b+t-1: pis.append(KT(pis[-1]))   # X_i 的分布，冷启动 X_0~Q
    gs  = [h];   for dd in 1..t-1:  gs.append(K(gs[-1]))      # E[h(X_{i+dd}) | X_i = x]
    Eh  = [pi @ h for pi in pis[b:b+t]]
    M   = (stack(pis[b:b+t]) * h) @ stack(gs).T               # M[i,dd] = E h(X_{b+i}) h(X_{b+i+dd})
    a   = mean(Eh)
    d   = (sum_i M[i,0] + 2*sum_{i, dd>=1, i+dd<t} M[i,dd]) / t**2
    P   = qm*w / sum(qm*w);  mu = P@h;  nu = P@(h*h)
    return a, d, mu, nu
def G_k(a, d, mu, nu, k): return d - 2*mu*a + mu**2 - (nu - mu**2)/k
```

单元测试：
- N≤2000 的随机实例，与稠密矩阵 K 的结果相对误差 <1e-10；
- 与 10⁶ 条 MC 链比较，偏差 <4 个标准误；
- 复现 `experiments/finite_state/cap_equality.py` 的 G/x→0.0063138。

### 2.2 闸门多项式与 κ

```python
def roots_Fk(a, d, k):  # 二值 h：F_k(m) = d - (2a+1/k) m + (1+1/k) m^2，安全区间 = 两根之间
    A2, A1, A0 = 1+1/k, -(2*a+1/k), d
    disc = A1**2 - 4*A2*A0;  if disc <= 0: return None
    return ((-A1 - sqrt(disc))/(2*A2), (-A1 + sqrt(disc))/(2*A2))
def kappa(mu0, l, u): return min(1 - l/mu0 if l > 0 else 1, 1 - (1-u)/(1-mu0) if u < 1 else 1)
def Pi_p(delta, kap, p): return min(1, delta/(1-kap)**p)
```

### 2.3 计数模拟（无损）

离散族的状态是若干"格子"（权重层 × h），样本只需记各格计数。

```python
nP = rng.multinomial(p, P_cells); nQ = rng.multinomial(q, Q_cells)   # q 可到 2^40
# 冷路径均值：先用 DP 算出窗口和 Σ_{i=b}^{b+t-1} h(X_i) 的精确分布，状态为 (cell, count)，规模 cells×(t+1)
pmf = window_sum_pmf(cells, b, t);  path_counts = rng.multinomial(n_paths, pmf)
```

### 2.4 e_p 统计量

用 Newton 恒等式，按计数加权计算。

```python
def e_p_ratio(x, y, cnt, p):           # 返回 e_p(y)/e_p(x)，其中 x、y 为各格的值，cnt 为各格计数
    s = sum(cnt*x); x, y = x/s, y/s    # 先归一化，防止溢出
    def e(z):
        P = [sum(cnt*z**i) for i in 1..p]; E = [1]
        for j in 1..p: E.append(sum((-1)**(i-1)*E[j-i]*P[i-1] for i in 1..j)/j)
        return E[p]
    return e(y)/e(x)
# 小规模时与精确 DP（fractions 或 mpmath，逐项 E[j] += z*E[j-1]）对照，相对误差 <1e-9
```

### 2.5 审计器

每个审计器都返回签发概率；随机化审计器返回 min(1,δ'E)，汇总时直接对它取平均。

```python
def A_ep(Prows, Qrows, delta, k, gamma):              # 达到上限的程序（B1）
    Qpool, Qpath = split_half(Qrows)
    a_lo, d_hi = hoeffding(path_avgs(Qpath), gamma)    # 两个量合计误差 gamma
    r = roots_Fk(a_lo, d_hi, k);  if r is None: return 0
    l, v = r
    x = S[Prows ∪ Qpool];  Bp = e_p_ratio(x, x*h, ...);  Bm = e_p_ratio(x, x*(1-h), ...)
    E = min(Bp/l**p, Bm/(1-v)**p);  return min(1, (delta-gamma)*E)

def A_knownZ(Qrows, Z, delta, k):                     # p=0，已知 Z（B3a）
    Qmean, Qpath = split_half(Qrows); W = S[Qmean]/Z; c = len(Qmean)**0.25
    mu_L = mean(min(W*h, c)) - c*sqrt(log(8/delta)/(2q))
    mu_U = 1 - (mean(min(W*(1-h), c)) - c*sqrt(log(8/delta)/(2q)))
    a_lo, d_hi = hoeffding(path_avgs(Qpath), delta/2); r = roots_Fk(a_lo, d_hi, k)
    return 1 if r and r[0] < mu_L and mu_U < r[1] else 0

def A_clip(Prows, Qrows, delta, k):                   # Pro 的截断证书，写法见 pro_moment_frontier_original.md
    Qmean, Qpath = split_half(Qrows); pilot, rest = split(Qmean, frac=1/8)
    H = (len(rest)/p) * quantile(S[pilot], 31/32)
    v = min(1, S_rest/H); g = min(1, H/S_P)
    u1, u0 = mean(v*h), mean(v*(1-h)) over rest;  c1, c0 = mean((1-g)*h), mean((1-g)*(1-h)) over P
    # 四个量用同时成立的经验 Bernstein 区间（Maurer–Pontil），误差预算 delta/C0
    th_lo, th_hi = u1_lo/(u1_lo+u0_hi), u1_hi/(u1_hi+u0_lo)
    F = lambda th, c1, c0: th + (1-th)*c1 - th*c0
    mu_lo, mu_hi = clip01(F(th_lo, c1_lo, c0_hi)), clip01(F(th_hi, c1_hi, c0_lo))
    a_lo, d_hi = hoeffding(path_avgs(Qpath))
    return int(max(d_hi - (2*a_lo + 1/k)*m + (1+1/k)*m**2 for m in (mu_lo, mu_hi)) < 0)

def A_replay(Prows, Qrows, delta, k):                 # 基线：中心只用 P 标签的 Clopper–Pearson，其余同上
    ...

def LR_oracle(cellsP0, cellsP1, cellsQ0, cellsQ1, p, q, delta, reps=20000):
    # 已知两个世界：用计数模拟 log LR；在不安全世界上把第一类错误压到 delta，报告安全世界上的功效
```

## 3. 实验

### X1 中心通道前沿（本机 CPU，约 2–4h）

**实例。** Theorem A 的三层族：
- L=b+t，取 (b,t)=(0,8)，k=2；J=128kL，K=ρ:=q/p，m=Ψ(K)/12；
- 二值 h；bulk 的 h=1 概率 β 用二分法标定，使 θ=1/2 时 G_k=0（用 2.1 精确计算）；
- 两个世界 θ±=1/2±s，s=C·ε/m，C 取能让两个世界跨过 ±ε 的最小值（实测记录）；
- 低层的 u_L 用于补偿，使 Q(h=1)=1/2 不变。

**Ψ 的形状。**
- 幂律 r∈{0.25,0.5,0.75}，取 M=D₀^r；
- 折线：D_b 之前 r₁=0.25，之后 r₂=0.75，衔接处连续。

**网格。** ρ∈2^{12..40}，步长 2；ε=Ψ(ρ)/2000；δ=0.05。
- 只保留落在定理区间内的点：K=ρ≥D₀=2048，且 Ψ(ρ)≤1/2。
- 折线 Ψ 的拐点改为 D_b=2^20，保证两段各有 ≥4 个点。
- 计数可能超过 int64：期望计数 >10^12 的格子改用正态近似 N(nπ, nπ(1−π))；q>2^60 的点不跑，并在 summary 里注明。

**X1a（下界校验）。** 对每个 ρ，二分求 LR_oracle 达到功效 ≥0.95 所需的最小 p，记 y=p_min·ε²/log(1/δ)。判据：y/Ψ(ρ) 在所有 ρ 和 Ψ 形状上的最大/最小比值 ≤4。

**X1b（上界，主结果）。** 在同样的实例上，令 p=c·Ψ(ρ)·A，c∈2^{0..14}，求 A_clip 功效 ≥0.95 的最小 c*(ρ,Ψ)。判据：所有 ρ 和 Ψ 上 c* 的最大/最小比值 ≤8。还要报告 c* 的绝对大小（Pro 的经验是常数很大）。

**X1c（拐点）。** 用折线 Ψ 的 A_clip 结果画 log p_min 对 log q（固定 ρ 网格，q=ρ·p_min）。两段分别拟合斜率，判据见 P2。

**X1d（对手的 K）。** 取 3 个 ρ，令 K/ρ∈2^{−4..4}，看 LR 所需的 p 是否在 K≈ρ 附近最大。只描述，不设判据。

**X1e（小重叠与饱和）。**
- 实现 A_clip_grid：用独立 P 样本的 1/4 分位数作锚点 a，在 H_j=a·(q/p)·2^{−j}（j=−3..⌈log₂(q/p)⌉+1）上各算一次 A_clip 的中心区间，每个区间误差 δ/(4J)，取交集后再做路径检验。
- 低重叠版本：把三层族的提议分布与一个"暗层"混合。暗层占 Q 质量 1−η₀，原始分数取原族最小分数的 10^{−6}，目标质量可以忽略。原族各层的 Q 质量乘以 η₀，于是归一化权重放大约 1/η₀ 倍，实际重叠约为 η₀ 乘以原族重叠，尾部轮廓相应变为 Ψ(η₀D)，隐藏层取 K/η₀。
- 取 η₀∈{2^{−8},2^{−12}}。原 pilot（Q 的 31/32 分位数）会落进暗层，因此预期 A_clip 失效。比较 A_clip 与 A_clip_grid 的 c*。判据：A_clip_grid 的 c* 与 η₀ 无关（最大/最小比值 ≤4）。
- 饱和：固定 Ψ 为幂律 r=0.5，令 ρ 增大到 Ψ(ρ)<ε。判据：A_clip_grid 所需 p 趋于常数 ×log(1/δ)/ε，不再随 Ψ(ρ) 下降。

### X2 固定 p 上限、速率与二分（本机 CPU，约 2h）

**实例。**
- I1：Part II 的公平硬币，Q 均匀，S≡1，t=4，b=0，k=2，κ=1/√6。此时 B₊ 是超几何比值，有闭式。
- I2(κ)：三层族在根附近，κ∈{0.02,0.05,0.1,0.2}。通过调 β 让 μ₀ 离根 Θ(κ)，并记录精确的 κ 和 G。
- I3(r)：重尾族。权重层 w_j=2^j，j=−6..30；Q 质量 ∝2^{−j(1+r+0.05)}（j≥0），低层用来使 E_QW=1；r∈{0.5,1}。h 的概率随层变化，使 μ₀ 在 (0.3,0.7) 内，且实例严格安全。

**X2a。** 用 Part I 的两世界精确 ROC 公式 1−e₀+e₀·min(1,δ/e₁)，其中 e_i=(1−QD)^q(1−P_iD)^p。令 QD→0，画它收敛到 δ/(1−α)^p。纯公式计算，不需要模拟。

**X2b。** 在 I1 和 I2 上取 p∈{0,1,2,4,8,16}，q∈2^{8..30}，用 A_ep 求签发概率的均值（每格 2000 次重复），与 Π_p 比较（P3）。

**X2c。** 在 I3 上记录缺口 Π_p−Power_q 随 q 的变化，p∈{2,8}，在对数坐标拟合斜率（P6）。

**X2d。** 这是 Theorem 2 的主图。
- 逐点：在 I2 型实例上，令 G=−ε，ε∈{0.04,0.02,0.01,0.005,0.0025}，计算 p_min=min{p: δ/(1−κ)^p≥0.8}（公式）。再用 A_ep 在 q=2^30 处验证 3 个点。
- 一致：用长窗口笔记的 center floor 族，α≪1/q，q=2^20。用 LR_oracle 求功效 ≥0.8 所需的 p_min(ε)。
- 两条线画在同一张对数图上（P4）。

**X2e。** 在 I1 和 I3 上取 p=0：A_knownZ 的功效随 q 上升；A_ep 的功效始终 ≤δ（P5）。

### X3 一般 k 的核验（本机 CPU，<1h）

- k∈{1.5,2,3,4,8}。
- center floor：±1 符号，计算 G_k=1/T+(1+1/k)μ²−1/k 的根，与 μ_*=(k+1)^{−1/2} 比较（T=2^16）。
- Theorem A 三层族：取 t=2k、J=128kL，确认标定根存在、两个世界跨过 ±ε；记录括号项随 k 的值，检查它是否 ≳c/√k。
- 判据见 P7。任何失败都要报告，这会直接修正定理的陈述。

### X0 提议头部筛查（Kaggle，≤1h）

目的：确认 LLM 任务不会被"便宜地枚举高概率序列"直接解决（见 `results/disclosure/README.md` 的高阶奖励对照）。

```python
model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb", torch_dtype=float16).cuda()   # π_ref = Q
rm    = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb").half().cuda()
prompts = 按 sha256(review_id) 排序取 IMDB test 前 12 条，前缀 = 前 8 个 gpt2 token          # 冻结在 config
for x in prompts:
    head = beam_search(x, num_beams=512, num_return_sequences=512, max_new_tokens=32)        # 近似最可能的 512 条
    logq_head = seq_logprob(model, x, head); r_head = clip(rm_logit_diff(x+head), -4, 4)
    bank_small = sample(x, n=8192, top_k=0, top_p=1, temperature=1, max_new_tokens=32)
    Zhat_beta = mean(exp(r_bank/beta))                     # 仅用于筛查
    head_Qmass = sum(exp(logq_head)); head_Pmass(beta) = sum(exp(logq_head + r_head/beta)) / Zhat_beta
    dup_rate = 1 - unique(bank_small)/8192
```

通过条件：对 β∈{0.5,1,2,4}，head_Pmass<0.5 且 dup_rate<0.05。不通过的 prompt 剔除；通过的少于 8 个就停下来汇报。

### X4 LLM 奖励倾斜样本库（Kaggle 建库 ≤8h；审计在本机 CPU）

**建库**，在 X0 通过后进行：
- 每个 prompt 纯采样 N=2^18 条，32 个新 token，top_k=0；
- 每条保存：r=clip(logit_pos−logit_neg,−4,4)（float32）、π_ref 对数概率、文本 hash（uint64）、长度、h 位；
- h 在 config 里冻结，至少 3 个：h1=含否定词（列表冻结）；h2=1{sigmoid(r)>0.95}；h3=1{含 "!"}；
- 每个 prompt 结束写一次 checkpoint。
- 样本库即这个代理问题中的 Q（库上的均匀分布）。P∝exp(r/β) 在库上精确可算，只用于评估。审计器不得读取整张表，只能按预算付费抽样。

**任务冻结。**
- 遍历 β∈{0.5,1,2,4}、h、k∈{2,4}，以及 (b,t)∈{0,4,16,64}×{4,8,16,32,64,128}；
- 用 2.1 计算精确的 G_k/Var、κ、η、ω=min W 和 τ_P(D) 曲线；
- 选 0.03≤|G_k|/Var≤0.3 的任务，安全与不安全各占一半，至少各 60 个；
- 写成 `tasks.json` 并记录 sha256。此后才开始审计。

**审计**（每个任务每种设置重复 200 次）：
- P 样本：在库上按 exp(r/β) 做类别抽样；同时按几何分布模拟拒绝采样尝试次数，成功率 Z·e^{−4/β}，计入成本。
- Q 样本：在库上均匀抽样。
- X4a（P5）：在 q∈2^{10..18} 上，比较 A_ep（Z 未知）与 A_knownZ（给精确 Z）达到功效 0.8 所需的最小 p。预测：A_knownZ 的 p 在大 q 时为 0；A_ep 接近 ⌈log(0.8/δ)/(−log(1−κ))⌉。
- X4b（P1，描述性）：A_clip 与 A_replay 的最小 p 随 q 的曲线，与用库上 τ_P 代入的 Ψ_emp(q/p)·log(1/δ)/G² 作形状比较。报告秩相关，不设通过线。
- X4c（P3）：q=2^18 时 A_ep 的功效随 p∈{0,1,2,4,8,16,32} 的变化，与按库上 κ 算出的 Π_p 比较。
- 所有签发都记入 X6。

### X5 真实生成迁移（可选，Kaggle ≤3h）

- 选 2 个 prompt × β∈{1,2}。
- 不用样本库：Q 来自新生成；P 来自真实拒绝采样（以 π_ref 为提议，接受概率 exp((r−4)/β)）。
- 预算 p≤256，q≤2^16。用一个独立的大样本库（2^18）作为参考真值。
- 报告审计决策与库上决策是否一致，以及真实成本。

### X6 有效性汇总（本机，随 X1–X4 一起累计）

- 汇总所有不安全实例上的签发：合成族至少 200 个，LLM 任务另计。
- 报告误发率及 Clopper–Pearson 95% 上界，按审计器分开。判据见 P8。

### X8 依赖尾部的 P 跳跃成本（本机 CPU，<1h；阶段 1 末尾做）

- 起点是现成的模拟 `experiments/finite_state/jump_cost.py`（单个滞后 k 的测试次数）。在此基础上实现完整的双时点重复：i、j 在窗口 [b,b+t) 上均匀，两次跳跃的测试次数相加。
- 目标律：Pareto（尺度 D₁=1/η₀，指数 r）加上 η₀/2 的低权重质量。网格 η₀∈{2^{−8},2^{−10}}，r∈{0.25,0.5}，t∈2^{8..20}，每格 2000 次重复。
- 记录每次重复的平均测试次数，以及 x=D_{1/2}/(η₀t) 的值，画在对数坐标上（P10）。
- 如果 X4 的样本库已建好，再按 β 计算库上的 D_{1/2}、η 和平均成本，看 β 扫描是否跨越 D_{1/2}≈η₀t。只描述，不设判据。

### X9 冷启动风险上界与 Γ-证书（本机 CPU，合成部分 <1h；样本库部分在阶段 3）

起点：`experiments/finite_state/risk_bound.py` 和 `risk_bound_refined.py`（本机已跑过，日志在 `results/local_runs/theory_checks-20261122/`）。

**X9a（P11，精确核验）。**
- 合成实例：沿用两个脚本的生成器，再加上 X1 的三层族和 X8 的 Pareto 加低权重族。
- t∈2^{3..12}，b∈{0,t/4,4t}。h 取阈值指示 1{W≥v}、1{W≤v}（v 走 W 的 12 个分位点），外加随机 h。
- 每个 (实例,b,t) 记录 max_h R/Γ_t、max_h R/Ξ_{b,t}、Γ_t/Ξ_{b,t}。
- 若 X4 样本库已建好：对每个 β、每个冻结的 h 与 (b,t)，用 2.1 的精确路径矩算 R、Γ_t、Ξ，同样记录三个比值。
- 输出 `x9a.csv`，再画 R 对 Γ_t 的散点图（对数坐标，配 y=44x 和 y=x 两条线）。

**X9b、X9c（P12，Γ-证书）。** 伪代码：

```python
def A_gamma(Prows, Qrows, t, k, delta, C0=44):        # 推论 R2；三个库独立，各分 delta/3
    P0, P1, P2 = split(Prows, [ceil(32*log(6/delta)), half, half])
    c  = quantile(S[P0], 5/8)                          # 估计 Z·d 的上方
    ZL = c * eb_lcb(minimum(S[Qrows], c)/c, delta/3)   # Maurer–Pontil 下界；E_Q min(S,c)<=Z 恒成立
    if ZL <= 0: return 0
    UG = eb_ucb(minimum(S[P1]/ZL, t)/t, delta/3)       # Γ_t 的上界（对 Z 单调）
    VL = var_lcb(h[P2], delta/3)                        # 配对 U 统计量 (h_i-h_j)^2/2 加 Bernstein
    return int(C0*UG < VL/k)
```

- 实例：合成族（lognormal σ∈{1,2,3}；Pareto r∈{0.5,1,1.5} 加 η₀/2 低权重质量），t∈2^{4..12}，k∈{2,4}，h 取阈值指示，使 Var/(kΓ_t) 覆盖 [1,10^4]。
- 预算网格：p∈2^{6..16}，q∈2^{8..18}；每格重复 200 次，δ=0.05。
- X9b：签发概率对 x=Var_P(h)/(kΓ_t) 作图（各实例合并）。所有签发同时用精确 G_k 判对错，记入 X6，预测误发数为 0。
- X9c：q 取最大时，达到 80% 签发率所需的最小 p 对 k·log(1/δ)/Var 作图，拟合对数斜率（P12），只用 x≥10³ 的实例。同时报告签发率越过 50% 时 x 的位置。
- 对照：同一批实例上，用 A_clip 加冷路径重复做门槛审计，按总抽样数（P 样本数 + Q 样本数，每条路径计 L 个）比较。只描述，不设判据。
- 探索变体：C0=2（实测常数），明确标注"理论上不保证有效"，只用来展示常数损失有多大，不进入 X6 的有效性结论。
- 若 X4 样本库已建好：每个 β 上对冻结任务跑 A_gamma，报告有多少任务能在 p≤2^12 内签发，以及它们的 G_k/Var 分布。

### X10 方差缩放的动力学下界族（本机 CPU，<1h；阶段 1 末尾做）

起点：`experiments/finite_state/variance_floor.py`（本机已跑，日志在 `results/local_runs/theory_checks-20261122/variance_floor.log`）。族的定义见 variance_scaled_floor.md：8 个格子（B±、D±、M₁±、M₂±），两个世界只差参数 θ。

**X10a（P13，精确核验，扩网格）。**
- k∈{1.5,2,3,4,8}；β∈{0,1,2,4}；(c₁,c₂)∈{(1,4),(0.5,2),(3,10)}；只保留满足条件 (K) 的组合，σ=(kΦ̄−1)/2−0.0111>0。
- t∈2^{10..16}，ρ∈{0.005,0.01,0.02,0.05}，ε=ρ|Φ₁−Φ₂|/16（倾斜 Δ=1/4）。
- 每个组合记录：两个世界的不变量差（W 在 P 和 Q 下的分布、P(h=1)、Q(h=1)）；G(θ*∓Δ)/ε；G(θ) 的二次拟合残差；R/ρ 对 Φ̄/2；两个 KL 常数。

**X10b（P13，最优检验的样本量标度）。**
- 两个世界的单次抽样分布都已知：P 抽样落在 8 个格子上，Q 抽样同样。所以似然比检验只依赖格子计数，直接模拟多项分布即可。
- 对 (ρ,t) 网格，找使两类错误都 ≤δ=0.05 的最小 q（p=0）和最小 p（q=0），每点 20000 次重复。ε 固定为 ρ_min|Φ₁−Φ₂|/16；每换一个 ρ，都重新求 θ* 和 Δ=4ε/(ρ|Φ₁−Φ₂|)。
- 预测：最小 q ∝ ρt/ε²，最小 p ∝ ρ/ε²。

```python
def min_budget(cellsA, cellsB, delta, which, reps=20000):   # which='P' or 'Q'
    llr = log(cellsA/cellsB)                                # 每个格子的对数似然比
    for n in geometric_grid():                              # 二分最小样本量
        LA = multinomial(n, cellsA, reps) @ llr;  LB = multinomial(n, cellsB, reps) @ llr
        thr = quantile(LB, 1-delta)                         # B 世界误发率 <= delta
        if mean(LA <= thr) <= delta: return n
```

### X11 小重叠下界的精确核验（本机 CPU，<1h；阶段 1 末尾、X10 之后做）

起点：`experiments/finite_state/small_overlap_floor.py`（本机已跑，日志在 `results/local_runs/theory_checks-20261122/small_overlap_floor.log`）。底座 B(t,d,ρ;c₁,c₂) 有 5 层（暗层 D、启动层 U、主体层 B、粘滞层 M₁、M₂），每层两种标签，共 10 个格子。定义见 variance_scaled_floor.md「Small overlap」节。b=0。

**X11a（P14 (i)–(iv)，精确核验，扩网格）。**
- k∈{4,6,8}；(c₁,c₂)∈{(3,10),(2,6)}；只保留 kΦ̄₀>1 的组合。
- t∈2^{12..20}，ρ∈{0.02,0.05,0.1}，ρ_B/ρ∈{1/16,1/4,1/2,1}，同时要求 2kρ_B<1、dρ≥5；不满足的组合记为 n/a，不算失败。
- 倾斜 Δ=1/4；ε 取诚实余量 min|G(±)|。
- 每个组合记录：
  - 二次型 R 与显式格子链的 R；
  - 不变量差（μ、Q(h=1)）；
  - S 对：a_U/pred、G∓/ε、KL_P·d/Δ²、KL_Q·d/Δ²；
  - V_η 对：G∓/ε、KL_P/ε²；
  - 增益 K；
  - 200 组随机补偿标签对上的 CS 比值最大值，以及低层超额/e_b 的最大值。
- 另报 ρ_B=ρ/16 一行：a_U 靠近相消点，预计 K≈0。这一行只描述现象，不设判据。

**X11b（可选，P14 (iv)，最优检验的样本量）。**
- 两个世界的单次抽样分布都在 10 个格子上已知，所以直接模拟多项分布计数的似然比检验。复用 X10b 的 `min_budget`。
- 固定 ρ=0.05、ρ_B∈{ρ/2,ρ}，t∈2^{12..18}，δ=0.05，每点 20000 次重复。
- 分别对 S 对和 V_η 对求最小 p（q=0），记 n_S·ε_S² 与 n_V·ε_V² 的比值。
- 预测：比值与 K 相差不超过 2 倍，且对 t 的对数斜率∈[0.8,1.2]。

```python
for t, rho, rB in grid:
    d = rB*t; w, q = base(t, d, rho)                 # 5 层，Σq=1，Σqw=1 精确成立
    m = moments(w, q, t)                             # 反向递推得 C、S（O(n^3 t)）
    ref = calibrate(m, th0, k)                       # 用主体层标签把 G_k 调到 0；无根则 n/a
    S_pm = [tilt(m, ref, 1, s*0.25, k) for s in (+1, -1)]          # 启动层标签 ±Δ，暗层和主体层补偿 μ、Q(h=1)
    V_pm = [tilt(m, ref, 3, s*0.25, k) for s in (+1, -1)]          # 粘滞层符号配对 θ / 1−θ
    eps = min(abs(gate(m, x, k)) for x in S_pm)
    log(a_over_pred, G/eps, kl(m, *S_pm, 'P')*d/D**2, kl(m, *S_pm, 'Q')*d/D**2, K, inv, cs_ratio, excess_over_eb)
```

### X7 审计器常数改进（可选，最后做）

- A_clip 的四个区间改用 betting CI（Waudby-Smith & Ramdas, 2024）。
- θ 改为对"被接受的 Q 样本"（以概率 v 接受）中的 h 用条件二项 Clopper–Pearson 区间。
- 在 X1b 和 X4b 上比较 c* 和最小 p 的改善。不改变理论判据。

## 4. 顺序、预算与停点

| 阶段 | 内容 | 资源 | 停点 |
|---|---|---|---|
| 1 | 公共组件和单元测试 → X2 → X1（含 X1e）→ X3 → X8 → X9（合成部分）→ X10 → X11 | 本机 CPU | 写阶段汇报，等用户确认后再用 GPU |
| 2 | X0（SMOKE→全量）→ X4 建库 | Kaggle ≤9h | X0 不通过就停 |
| 3 | 冻结任务 → X4 审计 → X9（样本库部分）→ X6 | 本机 CPU | 汇报 |
| 4 | X5、X7 | 可选 | 用户决定 |

## 5. 汇报格式

每个阶段结束写 `results/local_runs/STATUS_zh.md`，原地更新，内容包括：
- 每条预测（P1–P8）的状态：通过 / 未通过 / 未运行，附关键数字和图的路径；
- 与理论不符的地方，原样描述；
- 成本：CPU 小时、GPU 小时、样本计数；
- 偏离冻结 config 的地方和原因。
