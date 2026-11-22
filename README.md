# 冷启动 IMH 审计：理论与 CPU 实验底稿
2026-11-22。研究备份，不是投稿终稿；没有提交或发布，也没有在 Mac 跑研究实验。

## 现在最值得保留的两条结果
1. **不改变采样器，公开归一化尺度也能改变审计可行性。** 冷启动独立 Metropolis–Hastings 的接受率只用分数比值，乘同一个常数不会改变链。审计器的信息却不同：对支持随预算增长的黑盒分布类，长窗口、固定绝对误差下，已知尺度且能买足目标样本时可以零提议侧查询；只见原始评分时，任意有限的目标样本数也不能移除最坏情形的提议侧下界。下界可让两个世界精确共享目标均值和方差，并跨真实的半方差门槛。
2. **Pro 的互补结果：相同总评分数，来源配错也会失败。** 在明确的高阶矩、重叠和窗口区间，目标与提议配额的 (p^{1-r}q^r) 给出匹配边界。其难例靠改变未知目标均值，不能等同于第 1 条的归一化困难；也别把两条定理的适用区间拼成一个无条件结论。

## 先读
- [核心中文说明](paper/research_notes/research_brief_zh.md)：问题、主要推导、关键 CPU 结果、局限。
- [固定半方差门槛证明](theory/proofs/normalization_disclosure_half_variance.md)；[长窗口双源边界](theory/proofs/long_horizon_two_source_frontier.md)；[单独估归一化常数的黑盒推论](theory/proofs/normalizer_estimation_black_box.md)。
- [Pro 定理原文](theory/proofs/pro_moment_frontier_original.md)、[Pro 结果和对话链接](paper/research_notes/pro_result_zh.md)、[原始完整复现包与数据下载](paper/sources/SOURCE_LINKS.md)。
- [复跑说明](experiments/REPRODUCE_zh.md)。CPU 小例、六个 EWT 任务的逐例表和强基线都保留。高标记法在真实小面板不占优；固定半方差分离不是应用提速结果。

清单：在本目录运行 `shasum -a 256 -c SHA256SUMS` 可核对文件。EWT 数据沿用原 Universal Dependencies 快照，许可证与归属在 `experiments/ewt/data/`。

## 文件分类
- [theory/](theory/)：证明和相关工作分开。
- [experiments/](experiments/)：可运行代码、输入与使用说明。复跑只写入新建的运行目录，不覆盖已归档结果。
- [results/](results/)：有限状态、EWT、Pro 合并账本及诊断。这里记录已跑结果，不把 Pro 汇总账本冒充原始 GUM 逐例数据。
- [paper/](paper/)：中文研究备忘、材料来源和[手稿版本说明](paper/manuscripts/README.md)。

# ML_Research
