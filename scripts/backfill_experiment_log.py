"""
把已经跑完的 results/all_result.csv 回填成一份实验记录表。

为什么需要它：
    save_result() 现在会自动追加日志，但那只对「以后新跑的实验」生效；
    all_result.csv 里已有的历史结果不会再触发写日志。这个脚本把历史行补进去。

幂等：
    复用 save_result 里的去重索引 results/experiments_log.keys，
    重复执行不会产生重复行。

用法（项目根目录）：
    python scripts/backfill_experiment_log.py
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CSV_OUT
from src.utils import _append_experiment_log


def main():
    if not os.path.exists(CSV_OUT):
        print(f"找不到 {CSV_OUT}")
        return

    df = pd.read_csv(CSV_OUT)
    field_order = ["dataset", "seed", "model_kind", "k", "macro_f1", "r2",
                   "roc_auc", "logloss", "ece", "member_corr", "train_time_s", "params"]

    before = 0
    for _, r in df.iterrows():
        row = {c: r[c] for c in field_order if c in df.columns}
        _append_experiment_log(row, CSV_OUT)
        before += 1

    log_path = os.path.join(os.path.dirname(os.path.abspath(CSV_OUT)), "experiments_log.md")
    n_lines = 0
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for ln in f if ln.startswith("| ") and not ln.startswith("| # "))
    print(f"csv 共 {before} 行，实验记录表已写入 {n_lines} 条 -> {log_path}")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("backfill_experiment_log"):
        main()
