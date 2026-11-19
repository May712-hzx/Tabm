"""
[拓展板块] 把「k 个独立 MLP 集成」与主实验的 TabM 对齐比较。

只读 results/all_result.csv 与 results/extension/indep_mlp_result.csv,
输出 results/extension/cost_comparison.csv (每个 dataset × k 一行)。

用法:
    python extension/compare_with_tabm.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CSV_OUT

EXT_DIR = os.path.join(ROOT, "results", "extension")
INDEP_CSV = os.path.join(EXT_DIR, "indep_mlp_result.csv")
OUT_CSV = os.path.join(EXT_DIR, "cost_comparison.csv")

TASK = {"adult": "binclass", "california": "regression", "covertype": "multiclass"}
MAIN = {"adult": "macro_f1", "california": "r2", "covertype": "macro_f1"}


def main():
    if not os.path.exists(INDEP_CSV):
        print(f"缺少 {INDEP_CSV}, 请先运行 extension/run_indep_mlp.py")
        return
    main_df = pd.read_csv(CSV_OUT)
    ind_df = pd.read_csv(INDEP_CSV)

    # 时间口径: 剔除被同机负载污染的运行(单成员耗时 > 3× 中位数),
    # 再用干净的 per-member 平均 × k 表示"k 个成员串行训练"的真实成本
    med = float(ind_df["train_time_per_member_avg"].median())
    clean_ind = ind_df[ind_df["train_time_per_member_avg"] <= 3 * med]
    dropped = len(ind_df) - len(clean_ind)
    print(f"时间口径: per-member 中位数 {med:.1f}s × k;"
          f" 剔除受负载污染的运行 {dropped} 条 (per-member > {3*med:.1f}s)")

    rows = []
    for ds in sorted(ind_df["dataset"].unique()):
        m = MAIN[ds]
        sub_t = main_df[(main_df["dataset"] == ds) & (main_df["model_kind"] == "tabm")]
        sub_i = ind_df[ind_df["dataset"] == ds]

        print(f"\n{'='*92}")
        print(f"{ds}  主指标={m}  (均值 ± 标准差, 5 种子)")
        print(f"{'='*92}")
        print(f"{'k':>3} | {'TabM':>16} {'参数':>9} {'训练s':>8} | "
              f"{'独立k×MLP':>16} {'参数':>9} {'训练s':>8} | {'Δ指标':>8} {'参数×':>6} {'时间×':>6}")
        print("-" * 92)

        for k in [1, 4, 8, 16]:
            t = sub_t[sub_t["k"] == k]
            i = sub_i[sub_i["k"] == k]
            ic = clean_ind[(clean_ind["dataset"] == ds) & (clean_ind["k"] == k)]
            if t.empty or i.empty:
                continue
            tm, ts = t[m].mean(), t[m].std()
            im, isd = i[m].mean(), i[m].std()
            tp, tt = t["params"].mean(), t["train_time_s"].mean()
            ip = i["params"].mean()
            it = float(ic["train_time_per_member_avg"].mean()) * k if not ic.empty else np.nan

            print(f"{k:>3} | {tm:.4f}±{ts:.4f} {tp:>9.0f} {tt:>8.1f} | "
                  f"{im:.4f}±{isd:.4f} {ip:>9.0f} {it:>8.1f} | "
                  f"{im - tm:>+8.4f} {ip / tp:>6.2f} {it / tt:>6.2f}")

            rows.append({
                "dataset": ds, "k": k, "main_metric": m,
                "tabm_metric_mean": round(float(tm), 4), "tabm_metric_std": round(float(ts), 4),
                "tabm_params": int(tp), "tabm_train_s": round(float(tt), 2),
                "tabm_infer_ms": np.nan,
                "indep_metric_mean": round(float(im), 4), "indep_metric_std": round(float(isd), 4),
                "indep_params": int(ip), "indep_train_s": round(float(it), 2),
                "indep_infer_ms": round(float(i["infer_ms_total"].mean()), 2),
                "delta_metric": round(float(im - tm), 4),
                "params_ratio": round(float(ip / tp), 3),
                "train_time_ratio": round(float(it / tt), 3),
                "indep_ensemble_gain": round(float(i[f"ensemble_gain_{m}"].mean()), 4),
            })

        # 每单位参数/时间换来的性能
        print(f"\n  [成本效率] 相对本数据集 k=1 的性能增量, 按参数/时间归一化:")
        base_t = sub_t[sub_t["k"] == 1]
        ic1 = clean_ind[(clean_ind["dataset"] == ds) & (clean_ind["k"] == 1)]
        it1 = float(ic1["train_time_per_member_avg"].mean()) if not ic1.empty else np.nan
        for k in [4, 8, 16]:
            t = sub_t[sub_t["k"] == k]
            i = sub_i[sub_i["k"] == k]
            ic = clean_ind[(clean_ind["dataset"] == ds) & (clean_ind["k"] == k)]
            if t.empty or i.empty or base_t.empty:
                continue
            it = float(ic["train_time_per_member_avg"].mean()) * k if not ic.empty else np.nan
            print(f"    k={k:2d}  TabM 参数×{t['params'].mean()/base_t['params'].mean():.2f} "
                  f"时间×{t['train_time_s'].mean()/base_t['train_time_s'].mean():.2f} "
                  f"-> Δm={t[m].mean()-base_t[m].mean():+.4f} | "
                  f"独立MLP 参数×{i['params'].mean()/sub_i[sub_i['k']==1]['params'].mean():.2f} "
                  f"时间×{it/it1:.2f} "
                  f"-> Δm={i[m].mean()-sub_i[sub_i['k']==1][m].mean():+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n对照表已保存: {OUT_CSV}")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("ext_compare_with_tabm"):
        main()
