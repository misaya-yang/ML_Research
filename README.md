# ICML 2027：LLM 解码与采样研究

先看[研究证据索引](RESEARCH_INDEX.md)。这里是研究工作区，不是投稿版本。有数学证明、真实运行、离线复算和失败记录。四类证据不能混用。

- [当前研究问题：分叉、续写与从头重抽](RESEARCH_QUESTION.md)
- [旧主线：首次验证搜索](paper/research_notes/first_hit_research_question_20261123.md)
- [MBPP56：祖先证书、冻结回执和复算](results/first_passage/mbpp56_audit/README.md)
- [冻结模型：单次生成与零训练采样](results/evidence/single_pass.md)
- [实验入口与本机依赖](EXPERIMENTS_README.md)
- [冷启动 IMH：旧研究线](paper/research_notes/research_brief_zh.md) · [证明](theory/proofs/three_channels.md) · [披露对照](results/disclosure/README.md)

`theory/` 是证明；`experiments/` 是实验入口；`results/` 是冻结回执及索引；`paper/` 是笔记和旧稿。新运行写进忽略的 `results/local_runs/`，不要覆盖冻结证据。
