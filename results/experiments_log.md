# 实验记录表（自动生成）

> 由 `src/utils.py: save_result()` 在每次写入 `results/all_result.csv` 时**自动追加一行**，
> 不需要手工维护。字段对应指导书 §9.3：编号 / 时间 / 研究问题 / 唯一改动 / 命令 / 结果 / 判断。
> 去重规则：同一 (数据集, seed, 模型, k, 主指标值) 只记一次，重跑完全相同的结果不会重复。
> 用 `scripts/backfill_experiment_log.py` 批量补录的历史行，其「时间」列是回填时刻而非原始运行时刻。
> 「判断」列留给人填（或写报告时批注），其余列均为机器写入。

| # | 时间 | 数据集 | seed | 模型 | k | 研究问题 | 唯一改动 | 命令 | 主指标 | 值 | 参数量 | 训练(s) | 判断 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-18 22:23 | adult | 0 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8149 | 0 | 0.3 | — |
| 2 | 2026-11-18 22:23 | adult | 0 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8126 | 0 | 0.3 | — |
| 3 | 2026-11-18 22:23 | adult | 0 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7888 | 102482 | 10.3 | — |
| 4 | 2026-11-18 22:23 | adult | 0 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7862 | 94829 | 12.4 | — |
| 5 | 2026-11-18 22:23 | adult | 0 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7911 | 100532 | 23.1 | — |
| 6 | 2026-11-18 22:23 | adult | 0 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7892 | 108136 | 57.7 | — |
| 7 | 2026-11-18 22:23 | adult | 0 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7931 | 123344 | 131.8 | — |
| 8 | 2026-11-18 22:23 | adult | 1 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8179 | 0 | 0.2 | — |
| 9 | 2026-11-18 22:23 | adult | 1 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8204 | 0 | 0.3 | — |
| 10 | 2026-11-18 22:23 | adult | 1 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7967 | 102482 | 11.0 | — |
| 11 | 2026-11-18 22:23 | adult | 1 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7957 | 94829 | 9.4 | — |
| 12 | 2026-11-18 22:23 | adult | 1 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.8007 | 100532 | 22.8 | — |
| 13 | 2026-11-18 22:23 | adult | 1 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.8001 | 108136 | 37.9 | — |
| 14 | 2026-11-18 22:23 | adult | 1 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.8005 | 123344 | 122.1 | — |
| 15 | 2026-11-18 22:23 | adult | 2 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8098 | 0 | 0.3 | — |
| 16 | 2026-11-18 22:23 | adult | 2 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8141 | 0 | 0.3 | — |
| 17 | 2026-11-18 22:23 | adult | 2 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7788 | 102498 | 11.2 | — |
| 18 | 2026-11-18 22:23 | adult | 2 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7882 | 95086 | 11.9 | — |
| 19 | 2026-11-18 22:23 | adult | 2 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7927 | 100792 | 43.5 | — |
| 20 | 2026-11-18 22:23 | adult | 2 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7939 | 108400 | 42.8 | — |
| 21 | 2026-11-18 22:23 | adult | 2 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7895 | 123616 | 102.5 | — |
| 22 | 2026-11-18 22:23 | adult | 3 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8117 | 0 | 0.3 | — |
| 23 | 2026-11-18 22:23 | adult | 3 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8107 | 0 | 0.3 | — |
| 24 | 2026-11-18 22:23 | adult | 3 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7826 | 102482 | 12.4 | — |
| 25 | 2026-11-18 22:23 | adult | 3 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7849 | 94829 | 9.8 | — |
| 26 | 2026-11-18 22:23 | adult | 3 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7828 | 100532 | 29.5 | — |
| 27 | 2026-11-18 22:23 | adult | 3 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7934 | 108136 | 64.8 | — |
| 28 | 2026-11-18 22:23 | adult | 3 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7886 | 123344 | 158.1 | — |
| 29 | 2026-11-18 22:23 | adult | 4 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8120 | 0 | 0.8 | — |
| 30 | 2026-11-18 22:23 | adult | 4 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8121 | 0 | 0.4 | — |
| 31 | 2026-11-18 22:23 | adult | 4 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7838 | 102498 | 15.8 | — |
| 32 | 2026-11-18 22:23 | adult | 4 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7889 | 95086 | 13.2 | — |
| 33 | 2026-11-18 22:23 | adult | 4 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7925 | 100792 | 26.4 | — |
| 34 | 2026-11-18 22:23 | adult | 4 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7950 | 108400 | 69.5 | — |
| 35 | 2026-11-18 22:23 | adult | 4 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7927 | 123616 | 92.9 | — |
| 36 | 2026-11-18 22:23 | california | 0 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8314 | 0 | 0.3 | — |
| 37 | 2026-11-18 22:23 | california | 0 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8304 | 0 | 0.2 | — |
| 38 | 2026-11-18 22:23 | california | 0 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | r2 | 0.7922 | 68353 | 15.0 | — |
| 39 | 2026-11-18 22:23 | california | 0 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | r2 | 0.8011 | 69129 | 22.3 | — |
| 40 | 2026-11-18 22:23 | california | 0 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | r2 | 0.8106 | 73764 | 69.6 | — |
| 41 | 2026-11-18 22:23 | california | 0 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | r2 | 0.7990 | 79944 | 96.7 | — |
| 42 | 2026-11-18 22:23 | california | 0 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | r2 | 0.7972 | 92304 | 156.0 | — |
| 43 | 2026-11-18 22:23 | california | 1 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8261 | 0 | 0.5 | — |
| 44 | 2026-11-18 22:23 | california | 1 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8312 | 0 | 0.3 | — |
| 45 | 2026-11-18 22:23 | california | 1 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8007 | 68353 | 25.6 | — |
| 46 | 2026-11-18 22:23 | california | 1 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | r2 | 0.8054 | 69129 | 28.2 | — |
| 47 | 2026-11-18 22:23 | california | 1 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | r2 | 0.8066 | 73764 | 63.5 | — |
| 48 | 2026-11-18 22:23 | california | 1 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | r2 | 0.8008 | 79944 | 94.8 | — |
| 49 | 2026-11-18 22:23 | california | 1 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | r2 | 0.8017 | 92304 | 211.1 | — |
| 50 | 2026-11-18 22:23 | california | 2 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8261 | 0 | 0.3 | — |
| 51 | 2026-11-18 22:23 | california | 2 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8276 | 0 | 0.2 | — |
| 52 | 2026-11-18 22:23 | california | 2 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8068 | 68353 | 22.2 | — |
| 53 | 2026-11-18 22:23 | california | 2 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | r2 | 0.8051 | 69129 | 24.2 | — |
| 54 | 2026-11-18 22:23 | california | 2 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | r2 | 0.8057 | 73764 | 52.9 | — |
| 55 | 2026-11-18 22:23 | california | 2 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | r2 | 0.8077 | 79944 | 104.5 | — |
| 56 | 2026-11-18 22:23 | california | 2 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | r2 | 0.8085 | 92304 | 206.7 | — |
| 57 | 2026-11-18 22:23 | california | 3 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8344 | 0 | 0.3 | — |
| 58 | 2026-11-18 22:23 | california | 3 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8440 | 0 | 0.2 | — |
| 59 | 2026-11-18 22:23 | california | 3 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8065 | 68353 | 13.8 | — |
| 60 | 2026-11-18 22:23 | california | 3 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | r2 | 0.8187 | 69129 | 22.2 | — |
| 61 | 2026-11-18 22:23 | california | 3 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | r2 | 0.8182 | 73764 | 56.9 | — |
| 62 | 2026-11-18 22:23 | california | 3 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | r2 | 0.8218 | 79944 | 105.7 | — |
| 63 | 2026-11-18 22:23 | california | 3 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | r2 | 0.8106 | 92304 | 154.2 | — |
| 64 | 2026-11-18 22:23 | california | 4 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8283 | 0 | 0.3 | — |
| 65 | 2026-11-18 22:23 | california | 4 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | r2 | 0.8313 | 0 | 0.2 | — |
| 66 | 2026-11-18 22:23 | california | 4 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | r2 | 0.7976 | 68353 | 19.7 | — |
| 67 | 2026-11-18 22:23 | california | 4 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | r2 | 0.7994 | 69129 | 19.2 | — |
| 68 | 2026-11-18 22:23 | california | 4 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | r2 | 0.8028 | 73764 | 59.3 | — |
| 69 | 2026-11-18 22:23 | california | 4 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | r2 | 0.7955 | 79944 | 79.4 | — |
| 70 | 2026-11-18 22:23 | california | 4 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | r2 | 0.8024 | 92304 | 208.5 | — |
| 71 | 2026-11-18 22:23 | covertype | 0 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7556 | 0 | 1.1 | — |
| 72 | 2026-11-18 22:23 | covertype | 0 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7527 | 0 | 2.4 | — |
| 73 | 2026-11-18 22:23 | covertype | 0 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7662 | 81671 | 37.3 | — |
| 74 | 2026-11-18 22:23 | covertype | 0 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7866 | 82493 | 45.3 | — |
| 75 | 2026-11-18 22:23 | covertype | 0 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7919 | 91892 | 151.3 | — |
| 76 | 2026-11-18 22:23 | covertype | 0 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7872 | 104424 | 266.4 | — |
| 77 | 2026-11-18 22:23 | covertype | 0 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7645 | 129488 | 509.5 | — |
| 78 | 2026-11-18 22:23 | covertype | 1 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7740 | 0 | 1.1 | — |
| 79 | 2026-11-18 22:23 | covertype | 1 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7552 | 0 | 2.4 | — |
| 80 | 2026-11-18 22:23 | covertype | 1 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.8003 | 81671 | 40.3 | — |
| 81 | 2026-11-18 22:23 | covertype | 1 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.8079 | 82493 | 44.9 | — |
| 82 | 2026-11-18 22:23 | covertype | 1 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.8100 | 91892 | 151.5 | — |
| 83 | 2026-11-18 22:23 | covertype | 1 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7952 | 104424 | 266.1 | — |
| 84 | 2026-11-18 22:23 | covertype | 1 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7809 | 129488 | 510.7 | — |
| 85 | 2026-11-18 22:23 | covertype | 2 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7544 | 0 | 1.1 | — |
| 86 | 2026-11-18 22:23 | covertype | 2 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7640 | 0 | 2.4 | — |
| 87 | 2026-11-18 22:23 | covertype | 2 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7889 | 81671 | 42.6 | — |
| 88 | 2026-11-18 22:23 | covertype | 2 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7650 | 82493 | 33.1 | — |
| 89 | 2026-11-18 22:23 | covertype | 2 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7932 | 91892 | 151.9 | — |
| 90 | 2026-11-18 22:23 | covertype | 2 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7819 | 104424 | 270.2 | — |
| 91 | 2026-11-18 22:23 | covertype | 2 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7551 | 129488 | 456.0 | — |
| 92 | 2026-11-18 22:23 | covertype | 3 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7656 | 0 | 1.1 | — |
| 93 | 2026-11-18 22:23 | covertype | 3 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7141 | 0 | 0.9 | — |
| 94 | 2026-11-18 22:23 | covertype | 3 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7915 | 81671 | 46.9 | — |
| 95 | 2026-11-18 22:23 | covertype | 3 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.8035 | 82493 | 53.5 | — |
| 96 | 2026-11-18 22:23 | covertype | 3 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7993 | 91892 | 150.1 | — |
| 97 | 2026-11-18 22:23 | covertype | 3 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7774 | 104424 | 262.9 | — |
| 98 | 2026-11-18 22:23 | covertype | 3 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7661 | 129488 | 1394.2 | — |
| 99 | 2026-11-18 22:23 | covertype | 4 | xgb | — | 基线：XGBoost | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7487 | 0 | 5.9 | — |
| 100 | 2026-11-18 22:23 | covertype | 4 | histgb | — | 基线：HistGBM | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.6619 | 0 | 1.8 | — |
| 101 | 2026-11-18 22:23 | covertype | 4 | mlp | — | 基线：同预算 MLP | 无（基线） | `python scripts/run_exp.py` | macro_f1 | 0.7450 | 81671 | 101.2 | — |
| 102 | 2026-11-18 22:23 | covertype | 4 | tabm | 1 | H1/H2：成员数 k 对性能与成本的影响 | k=1 | `python scripts/run_exp.py` | macro_f1 | 0.7622 | 82493 | 126.5 | — |
| 103 | 2026-11-18 22:23 | covertype | 4 | tabm | 4 | H1/H2：成员数 k 对性能与成本的影响 | k=4 | `python scripts/run_exp.py` | macro_f1 | 0.7702 | 91892 | 379.9 | — |
| 104 | 2026-11-18 22:23 | covertype | 4 | tabm | 8 | H1/H2：成员数 k 对性能与成本的影响 | k=8 | `python scripts/run_exp.py` | macro_f1 | 0.7681 | 104424 | 754.2 | — |
| 105 | 2026-11-18 22:23 | covertype | 4 | tabm | 16 | H1/H2：成员数 k 对性能与成本的影响 | k=16 | `python scripts/run_exp.py` | macro_f1 | 0.7486 | 129488 | 1412.9 | — |
| 106 | 2026-11-18 22:23 | adult | 0 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7949 | — | — | — |
| 107 | 2026-11-18 22:23 | adult | 0 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7970 | — | — | — |
| 108 | 2026-11-18 22:23 | adult | 0 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7937 | — | — | — |
| 109 | 2026-11-18 22:23 | adult | 0 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7951 | — | — | — |
| 110 | 2026-11-18 22:23 | adult | 1 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.8000 | — | — | — |
| 111 | 2026-11-18 22:23 | adult | 1 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.8008 | — | — | — |
| 112 | 2026-11-18 22:23 | adult | 1 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.8002 | — | — | — |
| 113 | 2026-11-18 22:23 | adult | 1 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.8016 | — | — | — |
| 114 | 2026-11-18 22:23 | adult | 2 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7869 | — | — | — |
| 115 | 2026-11-18 22:23 | adult | 2 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7912 | — | — | — |
| 116 | 2026-11-18 22:23 | adult | 2 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7904 | — | — | — |
| 117 | 2026-11-18 22:23 | adult | 2 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7902 | — | — | — |
| 118 | 2026-11-18 22:23 | adult | 3 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7886 | — | — | — |
| 119 | 2026-11-18 22:23 | adult | 3 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7888 | — | — | — |
| 120 | 2026-11-18 22:23 | adult | 3 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7895 | — | — | — |
| 121 | 2026-11-18 22:23 | adult | 3 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7884 | — | — | — |
| 122 | 2026-11-18 22:23 | adult | 4 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7923 | — | — | — |
| 123 | 2026-11-18 22:23 | adult | 4 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7940 | — | — | — |
| 124 | 2026-11-18 22:23 | adult | 4 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7950 | — | — | — |
| 125 | 2026-11-18 22:23 | adult | 4 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7943 | — | — | — |
| 126 | 2026-11-18 22:23 | california | 0 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.7960 | — | — | — |
| 127 | 2026-11-18 22:23 | california | 0 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.7948 | — | — | — |
| 128 | 2026-11-18 22:23 | california | 0 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.7944 | — | — | — |
| 129 | 2026-11-18 22:23 | california | 0 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.7944 | — | — | — |
| 130 | 2026-11-18 22:23 | california | 1 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8008 | — | — | — |
| 131 | 2026-11-18 22:23 | california | 1 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8023 | — | — | — |
| 132 | 2026-11-18 22:23 | california | 1 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8018 | — | — | — |
| 133 | 2026-11-18 22:23 | california | 1 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8023 | — | — | — |
| 134 | 2026-11-18 22:23 | california | 2 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8072 | — | — | — |
| 135 | 2026-11-18 22:23 | california | 2 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8094 | — | — | — |
| 136 | 2026-11-18 22:23 | california | 2 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8085 | — | — | — |
| 137 | 2026-11-18 22:23 | california | 2 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8093 | — | — | — |
| 138 | 2026-11-18 22:23 | california | 3 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8087 | — | — | — |
| 139 | 2026-11-18 22:23 | california | 3 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8118 | — | — | — |
| 140 | 2026-11-18 22:23 | california | 3 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8105 | — | — | — |
| 141 | 2026-11-18 22:23 | california | 3 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8117 | — | — | — |
| 142 | 2026-11-18 22:23 | california | 4 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8014 | — | — | — |
| 143 | 2026-11-18 22:23 | california | 4 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | r2 | 0.8023 | — | — | — |
| 144 | 2026-11-18 22:23 | california | 4 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8017 | — | — | — |
| 145 | 2026-11-18 22:23 | california | 4 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | r2 | 0.8029 | — | — | — |
| 146 | 2026-11-18 22:23 | covertype | 0 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7501 | — | — | — |
| 147 | 2026-11-18 22:23 | covertype | 0 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7694 | — | — | — |
| 148 | 2026-11-18 22:23 | covertype | 0 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7607 | — | — | — |
| 149 | 2026-11-18 22:23 | covertype | 0 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7621 | — | — | — |
| 150 | 2026-11-18 22:23 | covertype | 1 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7670 | — | — | — |
| 151 | 2026-11-18 22:23 | covertype | 1 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7759 | — | — | — |
| 152 | 2026-11-18 22:23 | covertype | 1 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7773 | — | — | — |
| 153 | 2026-11-18 22:23 | covertype | 1 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7788 | — | — | — |
| 154 | 2026-11-18 22:23 | covertype | 2 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7449 | — | — | — |
| 155 | 2026-11-18 22:23 | covertype | 2 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7561 | — | — | — |
| 156 | 2026-11-18 22:23 | covertype | 2 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7562 | — | — | — |
| 157 | 2026-11-18 22:23 | covertype | 2 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7512 | — | — | — |
| 158 | 2026-11-18 22:23 | covertype | 3 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7645 | — | — | — |
| 159 | 2026-11-18 22:23 | covertype | 3 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7689 | — | — | — |
| 160 | 2026-11-18 22:23 | covertype | 3 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7675 | — | — | — |
| 161 | 2026-11-18 22:23 | covertype | 3 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7683 | — | — | — |
| 162 | 2026-11-18 22:23 | covertype | 4 | tabm_crop_random | 4 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7428 | — | — | — |
| 163 | 2026-11-18 22:23 | covertype | 4 | tabm_crop_valbest | 4 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=4 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7437 | — | — | — |
| 164 | 2026-11-18 22:23 | covertype | 4 | tabm_crop_random | 8 | H3：从 k=16 随机裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7432 | — | — | — |
| 165 | 2026-11-18 22:23 | covertype | 4 | tabm_crop_valbest | 8 | H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k | k=8 | `python scripts/run_crop_ablation.py` | macro_f1 | 0.7508 | — | — | — |
