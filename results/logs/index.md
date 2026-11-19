# 运行日志索引（自动生成）

> 每运行一次脚本追加一行。日志正文在同目录的 `<脚本名>_<时间戳>.log`。
> 与 `results/experiments_log.md`（一次实验配置一行）互补：这里是「一次运行」一行。

| # | 开始时间 | 脚本 | 命令 | 耗时(s) | 状态 | 日志文件 |
|---|---|---|---|---|---|---|
| 1 | 2026-11-19 10:14:52 | backfill_experiment_log | `python scripts\backfill_experiment_log.py` | 0.3 | OK | `backfill_experiment_log_20261119_101452.log` |
| 2 | 2026-11-19 10:15:31 | backfill_experiment_log | `python scripts\backfill_experiment_log.py` | 0.2 | OK | `backfill_experiment_log_20261119_101531.log` |
| 3 | 2026-11-19 10:40:31 | eval_extra_auc | `python scripts\eval_extra_auc.py --seeds 0` | 3.0 | OK | `eval_extra_auc_20261119_104031.log` |
| 4 | 2026-11-19 10:41:03 | eval_extra_auc | `python scripts\eval_extra_auc.py --fill` | 3025.6 | OK | `eval_extra_auc_20261119_104103.log` |
| 5 | 2026-11-19 11:38:43 | fill_crop_auc | `python scripts\fill_crop_auc.py --fill` | 28.6 | OK | `fill_crop_auc_20261119_113843.log` |
| 6 | 2026-11-19 14:04:42 | fill_crop_auc | `python scripts\fill_crop_auc.py --fill` | 43.5 | OK | `fill_crop_auc_20261119_140442.log` |
| 7 | 2026-11-19 14:53:03 | eval_member_variance | `python scripts\eval_member_variance.py` | 21.4 | OK | `eval_member_variance_20261119_145303.log` |
| 8 | 2026-11-19 14:53:56 | eval_cost | `python scripts\eval_cost.py` | 96.7 | OK | `eval_cost_20261119_145356.log` |
| 9 | 2026-11-19 14:58:57 | eval_cost | `python scripts\eval_cost.py` | 139.5 | OK | `eval_cost_20261119_145857.log` |
| 10 | 2026-11-19 15:01:53 | eval_cost | `python scripts\eval_cost.py --train-peak` | 3149.9 | OK | `eval_cost_20261119_150153.log` |
