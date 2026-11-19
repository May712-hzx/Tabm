# 复现手册：命令、产物与耗时

[← 返回 README](../README.md)

所有命令在项目根目录 `D:\project_tabm` 下执行，解释器用 `D:\python3.13\python.exe`（或任何装好 `requirements.lock` 的 Python 3.13）。

## 1 产物总览

| 类别 | 文件 | 说明 |
|---|---|---|
| 主结果原始表 | `results/all_result.csv` | 一行一次运行，165 行 = 3 数据集 × 5 种子 × 11 配置 |
| 模型权重 | `ckpt/tabm_<ds>_seed<s>_k16.pt` | 仅 k=16，15 个，供 H3 裁剪加载 |
| 实验记录表 | `results/experiments_log.md` | 一次实验一行（编号/问题/唯一改动/环境/结果/判断），写入 csv 时自动追加 |
| 运行日志 | `results/logs/<脚本>_<时间戳>.log` + `index.md` | 一次脚本运行一份，含命令、环境、全部 stdout/stderr、耗时、状态 |
| 主图 | `plots/<ds>_tabm_k_ablation.png` | 主指标 / LogLoss / 训练时间 / ECE 随 k |
| H3 图 | `plots/<ds>_h3_crop_ablation.png` | 主指标 / ECE / 成员相关性三合一 |
| 错误分析 | `results/error_analysis/` + `plots/error_analysis/` | 混淆矩阵、按类 P/R/F1、高置信错分样本、`summary.txt` |
| AUC 补算 | `results/extra_metrics/covertype_auc.csv` | macro-OvR AUC（主表 + 裁剪行） |
| 成本 | `results/cost/infer_cost.csv`（105 行）、`train_peak_seed0.csv`（21 行） | 推理延迟 / 权重内存 / 训练峰值 RSS |
| 多样性 | `results/variance/member_variance.csv`（30 行） | 逐样本概率方差、argmax 分歧率、成员相关性 |
| 拓展板块 | `results/extension/`（`indep_mlp_result.csv`、`cost_comparison.csv`、`member_probs_*.npz`） | k 个独立 MLP 成本上界 |
| 报告 | `paper/黄思桦 202425310307.pdf` / `.docx` + `paper/fig1_arch.png` | 课程报告（18 页 = 封面 1 + 正文 17），以学校封面模板 `paper/cover_template.docx` 为底稿，由 `paper/build_report.py` 生成 |

## 2 命令表

| # | 命令 | 产物 | 内容 | 耗时（本机纯 CPU） |
|---|---|---|---|---|
| 0 | `python -m unittest discover -s tests -v` | 控制台 | 21 项测试：划分无重叠、预处理只 fit 训练集、输出形状、逐成员损失梯度、指标一致性 | ~5 s |
| 1 | `python scripts/run_exp.py` | `results/all_result.csv`、`ckpt/*.pt` | 主实验矩阵（11 配置 × 5 种子 × 3 数据集） | ~3 h |
| 2 | `python scripts/backfill_experiment_log.py` | `results/experiments_log.md` | 把已有 csv 历史行补进记录表（幂等） | ~10 s |
| 3 | `python scripts/run_crop_ablation.py` | `results/all_result.csv`（追加） | H3 裁剪：随机 / val 择优 × crop∈{4,8}，纯推理 | ~5 min |
| 4 | `python scripts/eval_extra_auc.py --fill` | `results/extra_metrics/covertype_auc.csv` | 补 covertype macro-OvR AUC（k=16 走 ckpt，其余重训），支持断点续跑 | ~1 h |
| 5 | `python scripts/fill_crop_auc.py --fill` | `results/all_result.csv`（回填裁剪行 AUC） | 裁剪行 AUC，纯推理 | ~45 s |
| 6 | `python scripts/eval_cost.py` | `results/cost/infer_cost.csv` | 推理延迟、理论权重内存、RSS 增量 | ~2.5 min |
| 7 | `python scripts/eval_cost.py --train-peak` | `results/cost/train_peak_seed0.csv` | 训练峰值 RSS，每配置单独子进程 | ~50 min |
| 8 | `python scripts/error_analysis.py` | `results/error_analysis/`、`plots/error_analysis/` | 混淆矩阵、按类指标、100 条高置信错分样本 | ~20 min |
| 9 | `python plots/plot_result.py` | `plots/<ds>_tabm_k_ablation.png` | 主实验四联图 | ~20 s |
| 10 | `python plots/plot_h3_crop.py` | `plots/<ds>_h3_crop_ablation.png` | H3 三合一 | ~10 s |
| 11 | `python extension/run_indep_mlp.py` | `results/extension/indep_mlp_result.csv`、`member_probs_*.npz` | 每个 (数据集, 种子) 训练 16 个完全独立 MLP | ~2.5 h |
| 12 | `python scripts/eval_member_variance.py` | `results/variance/member_variance.csv` | 逐样本概率方差 / 分歧率（依赖 11 的 npz） | ~20 s |
| 13 | `python extension/compare_with_tabm.py` | `results/extension/cost_comparison.csv` | TabM vs 独立 MLP 逐 k 对照 | ~5 s |
| 14 | `python extension/plot_indep_mlp.py` | `plots/extension/<ds>_indep_mlp_vs_tabm.png` | 性能 / 参数 / 时间 / 参数-性能 四联图 | ~15 s |
| 15 | `python paper/build_report.py` | `paper/黄思桦 202425310307.docx` | 由结果文件生成课程报告（以学校封面模板为底稿，封面 1 页 + 正文 17 页；Python 需装 `python-docx`、`matplotlib`；再用 Word「另存为 PDF」导出同名 PDF） | ~10 s |

依赖顺序：3 依赖 1 的 ckpt；10 依赖 3；8 独立（k=16 走 ckpt）；4 依赖 1 的 ckpt；5 依赖 1 的 ckpt 与 3 的裁剪行；6 依赖 1 的 ckpt；7 独立重训；12 依赖 1 与 11；2 在 1/3 之后；13、14 依赖 11。

## 3 一键按序执行

```bash
python -m unittest discover -s tests -v
python scripts/run_exp.py
python scripts/backfill_experiment_log.py
python scripts/run_crop_ablation.py
python scripts/eval_extra_auc.py --fill
python scripts/fill_crop_auc.py --fill
python scripts/eval_cost.py
python scripts/eval_cost.py --train-peak
python scripts/error_analysis.py
python plots/plot_result.py
python plots/plot_h3_crop.py
python extension/run_indep_mlp.py
python scripts/eval_member_variance.py
python extension/compare_with_tabm.py
python extension/plot_indep_mlp.py
python paper/build_report.py
```

## 4 两类记录的分工

- `results/experiments_log.md` —— **一次实验一行**（数据集 / seed / 模型 / k / 指标 / 判断），由 `save_result()` 写 csv 时自动追加，无需手工维护；旧行可用 `backfill_experiment_log.py` 回填，按 (数据集, seed, 模型, k, 主指标值) 去重。
- `results/logs/<脚本>_<时间戳>.log` —— **一次脚本运行一份**（命令行、工作目录、Python 与依赖版本、全部 stdout/stderr、traceback、耗时、状态），由 `src/run_log.py` 的 `run_log()` 在任何入口自动生成；`results/logs/index.md` 追加索引行。

> `extension/run_log.txt` 是 `extension/run_indep_mlp.py` 早期自行重定向 stdout 留下的过程记录，**不是实验记录表**；接入 `run_log()` 后其作用已由 `results/logs/ext_run_indep_mlp_*.log` 取代，保留仅作历史。

## 5 清空重跑

只删产物，不动代码：

```bash
rm results/all_result.csv results/experiments_log.md results/experiments_log.keys
rm -r results/extension results/error_analysis results/extra_metrics results/cost results/variance results/logs
rm ckpt/*.pt
rm -r plots/*.png plots/extension/*.png plots/error_analysis/*.png
```

## 6 常见故障

- **`logloss` 报概率超过 1**：`predict_*` 已内置 softmax，自行改代码时别漏。
- **`At least one array or dtype is required`**：纯数值数据集（california / covertype）没有类别列，`dataset.py` 已在 `cat_cols` 为空时跳过类别分支。
- **H3 脚本提示 ckpt 不存在**：步骤 1 没跑到 `H3_SOURCE_K` 那一档，或 `ckpt/` 被清空。
- **`legacy/`**：旧版代码留下的权重与 csv（架构与列名已过时），新代码不读取，仅存档。
- **时间口径污染**：多任务同机并行会让 covertype 的训练时间被拖高。所有涉及倍数的讨论统一用中位数口径（见 [results.md](results.md)）。
