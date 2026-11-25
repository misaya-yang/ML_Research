# ICML 2027：研究证据索引

本页是入口，不是投稿稿件。仓库保留两类彼此不同的结果：**固定策略的精确首达加速**和**冻结模型的解码提质**。前者的同轨迹、省算证据，不能当成后者的准确率证据。

当前主线见[前缀分叉与重抽](RESEARCH_QUESTION.md)。下列首达结果保留为旧路线证据。

## 保留并复算

- **首达与预算。** [原研究问题](paper/research_notes/first_hit_research_question_20261123.md)、[首达核心证明](theory/proofs/first_passage_core.md)、[联合预算结果](results/first_passage/README.md)。14× 的确认来自固定归档图，并容许一处额外假阳性。旧 9×／95% 结论已因错误标签撤回。真实 Qwen 窄载荷减少 14.90% 额外 logits 行，低于原定 15% 门槛，不是墙钟或同预算正确题增益。
- **MBPP56 祖先证书。** [源码／回执／一键复算](results/first_passage/mbpp56_audit/README.md)。完整 K4 负载报告同结果下 391.333 → 323.276 秒；其中 32 道题、14,058 个解码位置和 556 次调用的历史省略发生在读出之前。必须保留第 468 题这个“省位置但不省时间”的反例。原生单个加权 Power-SMC 尚未由该结果确认。
- **冷启动 IMH。** [旧研究简报](paper/research_notes/research_brief_zh.md)、[三通道证明](theory/proofs/three_channels.md)、[披露对照](results/disclosure/README.md)和[有限状态复现](experiments/REPRODUCE_zh.md)。属于独立理论线。旧 [PDF](paper/manuscripts/README.md)不作为现行稿件。
- **零训练、单次生成。** [采样证据](results/evidence/single_pass.md)。不以验证器省调用、归档选择器或大样本极限代替现实预算下实际生成的正确率。

## 有价值的负结果

| 证据 | 正确结论 |
| --- | --- |
| 完整 P-GCD 16 题确认 | 同设备路径核验通过，但候选实现总平均请求时间 CPU 增加约 7.06%、M4 增加约 42.73%。不要引用前 13 题的正面阶段结论。 |
| MATH 旧截断前缀续写 | 直接低温和已选 SMC 各 9 个合资格前缀，分别新增 4、5 个正确端点；更严格端点口径是 3、5。不是长预算原生 SMC，不合并成总体准确率。 |
| [真实投机 MH](paper/research_notes/speculative_first_passage_20261123.md) | 10 题 PowerMH 5/10 用 583.2s，单次低温 5/10 用 173.7s且同一胜例集合，四次低温投票 6/10 用 547s。这批题上原策略价值门槛未过。 |
| 普通推理解码提议 | 四题中额外 transport 相对直接缓存只把调用从 750 减至 748；目标执行位置从 2,088 减至 2,072。不是主要贡献，不与原生人口加权逻辑混合。 |
| 71,400 条公开代码候选 | 714 题的旧轨迹分析可核验排序/读出，不是新推理；已测小型排序器对最先自然完成的优势很小，不能把 oracle 覆盖写成方法表现。 |

P-GCD 和续写的只读摘要及原始文件哈希见 [local_screening_snapshot.json](results/evidence/local_screening_snapshot.json)。生成命令：`python3 experiments/summarize_local_evidence.py --compare results/evidence/local_screening_snapshot.json`，需原机的 `results/local_runs/`。

## 只留本机的数据

- 机制笔记本：`results/local_runs/research-mechanism-audit-20261124/Findings.ipynb`。主审计 134 份哈希及候选键已核验。应运行只读核验单元，而非 Run All。来源与原始回执仍在同目录。
- 最新原生 SMC：`results/local_runs/native-smc-continuation-20261125/Continuation.ipynb`。保留非有限值、MPS 内存中止和单题恢复的原回执；未形成完整公平对照。不能把继续运行的实验写成成功。
- Lean 证明文本、数值误差见证、研究 notebook 和原始运行回执不随 Git 克隆交付。weight/toolchain/vendor 是环境依赖，不能作为结果提交；没有二次验证不要从仍在运行的实验中删除。
- 投机 MH 的理论界限应按“接受的完整状态须生成”来读；不能把它扩大为“任何接受决定都无法提前判定”。有限剩余长度可能给出接受下界。这里不改写原证明，留给后续理论复核。

## 归档规则

每一条值得保留的结论要有：任务、分母、比较对象、记录来源和不能推出的结论。Git 优先保留原著源码、轻量的核验、冻结回执及索引；不收模型权重、工具链副本、全部输出 notebook 或重复压缩备份。不删除唯一证据，不依靠当前机器的绝对路径引用已晋升的结果。
