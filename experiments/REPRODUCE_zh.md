# CPU 复跑
从项目根目录操作。Python 3，有限状态测试依赖 NumPy/SciPy；EWT 另需 pandas 和 GCC/Clang（代码默认用 `gcc`）。无需 GPU 或 API Key。有限状态核验不需要数据；复跑 EWT 前按 [原包链接](../paper/sources/SOURCE_LINKS.md) 补齐两份语料。

```sh
mkdir -p results/local_runs
run_dir="$(mktemp -d "$PWD/results/local_runs/finite_state-XXXXXX")"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/half_disclosure.py > "$run_dir/half_disclosure.csv"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/half_gate.py > "$run_dir/half_gate.log"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/p_only.py > "$run_dir/p_only.log"
OPENBLAS_NUM_THREADS=1 python3 experiments/finite_state/scale_floor.py > "$run_dir/scale_floor.log"
```

EWT 每次执行自动创建独立的子目录，默认在 `results/local_runs/`；可以设置 `IMH_RUN_DIR` 指定该目录下的子路径，或项目外的新输出位置。脚本启动时打印实际目录。归档 CSV 和日志不受影响。独立参考每项重跑 4,096 条长度 1,024 的链，明显慢于有限例。可先只跑已有强基线，它只需要目标样本即可签发：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/ewt/baseline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/ewt/sources/pilot.py
```

[高标记逐例表](../results/ewt/high_mark/pilot.csv)包含独立风险参考；[同信息强基线](../results/ewt/stationary_coupling/baseline.csv)覆盖相同六份 EWT 文档。每个配额是单独的固定试验，不能从看过结果后的比较直接推出同时覆盖。浮点与伪随机数未形式化。Pro 原始脚本、语料和冻结协议仍请按[原包链接](../paper/sources/SOURCE_LINKS.md)在另一个目录解压使用；[小配额诊断](../results/diagnostics/RESULTS.md)不作为主线胜利。

旧版谱辅助模块 `ewt/curve/risk_profile.py` 的 `filt` 是强基线依赖项；六个 EWT 任务不需要单独运行该文件。若另有旧版 `p.npz` 和 `q.npz`，可通过 `--input-dir` 显式传入，在独立运行目录生成曲线 CSV。
