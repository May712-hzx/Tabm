"""
[拓展板块] 画「TabM vs k 个独立 MLP」的成本-性能对照图。

每个数据集一张 2×2:
  (1) 主指标 vs k
  (2) 参数量 vs k
  (3) 训练时间 vs k
  (4) 参数量-性能散点(成本效率视角)

输出: plots/extension/<ds>_indep_mlp_vs_tabm.png
用法: python extension/plot_indep_mlp.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CSV_OUT

EXT_DIR = os.path.join(ROOT, "results", "extension")
PLOTS_DIR = os.path.join(ROOT, "plots", "extension")
INDEP_CSV = os.path.join(EXT_DIR, "indep_mlp_result.csv")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MAIN = {"adult": "macro_f1", "california": "r2", "covertype": "macro_f1"}
LABEL = {"macro_f1": "Macro-F1", "r2": "R2"}


def robust_time_series(ind_df):
    """
    训练时间的稳健口径。

    背景: 独立 MLP 是串行训练, 总时间对同机其他进程的抢占极其敏感
    (实测 covertype 某个种子单成员耗时从 70s 被拖到 1000s)。
    做法: 单成员平均耗时 > 3× 全体中位数的运行判为受污染并剔除;
          剩下的用 per-member 平均 × k 表示"k 个成员串行训练"的成本。
    返回 (每 k 的时间序列, 被剔除的运行条数, 全局中位数)
    """
    med = float(ind_df["train_time_per_member_avg"].median())
    clean = ind_df[ind_df["train_time_per_member_avg"] <= 3 * med].copy()
    clean["train_time_robust"] = clean["train_time_per_member_avg"] * clean["k"]
    s = clean.groupby("k")["train_time_robust"].agg(["mean", "std"]).reset_index()
    s.columns = ["k", "train_time_s_mean", "train_time_s_std"]
    return s, len(ind_df) - len(clean), med


def plot_dataset(ds, tabm_df, indep_df, dropped=0):
    m = MAIN[ds]
    ylab = LABEL[m]
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) 主指标 vs k
    ax = axes[0, 0]
    ax.errorbar(tabm_df["k"], tabm_df[f"{m}_mean"], yerr=tabm_df[f"{m}_std"],
                marker="o", capsize=4, lw=2, label="TabM (shared backbone)")
    ax.errorbar(indep_df["k"], indep_df[f"{m}_mean"], yerr=indep_df[f"{m}_std"],
                marker="s", capsize=4, lw=2, ls="--", label="k independent MLPs")
    ax.set_xlabel("k (# members)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{ds}: {ylab} vs k")
    ax.set_xticks([1, 4, 8, 16])
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9)

    # (2) 参数量 vs k
    ax = axes[0, 1]
    ax.plot(tabm_df["k"], tabm_df["params_mean"], marker="o", lw=2, label="TabM")
    ax.plot(indep_df["k"], indep_df["params_mean"], marker="s", lw=2, ls="--",
            label="k independent MLPs")
    ax.set_yscale("log")
    ax.set_xlabel("k (# members)")
    ax.set_ylabel("Trainable parameters (log scale)")
    ax.set_title(f"{ds}: parameter cost")
    ax.set_xticks([1, 4, 8, 16])
    ax.grid(alpha=0.3, which="both")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9)

    # (3) 训练时间 vs k
    ax = axes[1, 0]
    ax.plot(tabm_df["k"], tabm_df["train_time_s_mean"], marker="o", lw=2, label="TabM")
    ax.plot(indep_df["k"], indep_df["train_time_s_mean"], marker="s", lw=2, ls="--",
            label="k independent MLPs")
    ax.set_xlabel("k (# members)")
    ax.set_ylabel("Training time (s)")
    ax.set_title(f"{ds}: training cost")
    ax.set_xticks([1, 4, 8, 16])
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9)

    # (4) 参数量-性能散点
    ax = axes[1, 1]
    ax.plot(tabm_df["params_mean"], tabm_df[f"{m}_mean"], marker="o", lw=2,
            label="TabM")
    for x, y, k in zip(tabm_df["params_mean"], tabm_df[f"{m}_mean"], tabm_df["k"]):
        ax.annotate(f"k={int(k)}", (x, y), textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax.plot(indep_df["params_mean"], indep_df[f"{m}_mean"], marker="s", lw=2, ls="--",
            label="k independent MLPs")
    for x, y, k in zip(indep_df["params_mean"], indep_df[f"{m}_mean"], indep_df["k"]):
        ax.annotate(f"k={int(k)}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{ds}: performance vs parameter budget")
    ax.grid(alpha=0.3, which="both")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9)

    fig.suptitle(f"{ds} — TabM vs k independent MLPs (mean ± std over seeds"
                 + (f", {dropped} outlier run(s) dropped for timing)" if dropped else "") + ")",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{ds}_indep_mlp_vs_tabm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {out}")


def main():
    if not os.path.exists(INDEP_CSV):
        print(f"缺少 {INDEP_CSV}, 请先运行 extension/run_indep_mlp.py")
        return

    main_df = pd.read_csv(CSV_OUT)
    ind_df = pd.read_csv(INDEP_CSV)

    for ds in sorted(ind_df["dataset"].unique()):
        t = main_df[(main_df["dataset"] == ds) & (main_df["model_kind"] == "tabm")].copy()
        t = t.groupby("k").agg(
            **{f"{MAIN[ds]}_mean": (MAIN[ds], "mean"), f"{MAIN[ds]}_std": (MAIN[ds], "std"),
               "params_mean": ("params", "mean"), "train_time_s_mean": ("train_time_s", "mean")}
        ).reset_index()
        i = ind_df[ind_df["dataset"] == ds].groupby("k").agg(
            **{f"{MAIN[ds]}_mean": (MAIN[ds], "mean"), f"{MAIN[ds]}_std": (MAIN[ds], "std"),
               "params_mean": ("params", "mean")}
        ).reset_index()
        ts, dropped, med = robust_time_series(ind_df[ind_df["dataset"] == ds])
        i = i.merge(ts, on="k", how="left")
        print(f"\n绘图: {ds}  (时间口径: per-member 中位数 {med:.1f}s × k, 剔除 {dropped} 条污染运行)")
        plot_dataset(ds, t, i, dropped)

    print("\n完成, 图片在 plots/extension/")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("ext_plot_indep_mlp"):
        main()
