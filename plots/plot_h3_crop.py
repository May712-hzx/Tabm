"""
读取 results/all_result.csv, 画 H3 裁剪消融对比图。

对比对象(每个 crop_size ∈ {4, 8}):
  原生训练 tabm k=N  |  从 k=16 随机裁剪 N 个成员  |  从 k=16 按验证集择优裁剪 N 个成员

每个数据集输出 3 个子图:
  1. 任务主指标 (分类 Macro-F1 / 回归 R²)
  2. ECE 校准误差 (仅分类)
  3. 成员预测相关性 (H2 多样性, 越低越多样)

绘图形式用「点 + 误差条」而不是柱状: 同一数据集内差异常在 0.003 量级,
柱状图从 0 起画会把差异压成一条线, 点图配合自适应 y 轴才能看清方向。

用法(项目根目录):
    python plots/plot_h3_crop.py
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

from src.config import CROP_SIZES, CSV_OUT, PLOTS_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GROUP_ORDER = [
    ("tabm", "原生训练 k=N", "o"),
    ("tabm_crop_random", "随机裁剪 N/16", "s"),
    ("tabm_crop_valbest", "val 择优裁剪 N/16", "^"),
]
COLORS = {
    "tabm": "#1f77b4",
    "tabm_crop_random": "#ff7f0e",
    "tabm_crop_valbest": "#2ca02c",
}


def task_of(sub):
    if "r2" in sub.columns and sub["r2"].notna().any():
        return "regression"
    return "classification"


def stat(sub, kind, k, col):
    rows = sub[(sub["model_kind"] == kind) & (sub["k"] == k)]
    if rows.empty or col not in rows.columns:
        return np.nan, np.nan
    return float(rows[col].mean()), float(rows[col].std())


def fit_ylim(ax, values, pad_ratio=0.15):
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else (abs(hi) * 0.05 + 1e-9)
    ax.set_ylim(lo - span * pad_ratio, hi + span * pad_ratio)


def plot_dataset(ds, sub):
    task = task_of(sub)
    metric = "r2" if task == "regression" else "macro_f1"
    metric_label = "R2" if task == "regression" else "Macro-F1"

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    n_group = len(GROUP_ORDER)
    width = 0.22

    for axi, (col, ylabel, title) in enumerate([
        (metric, metric_label, f"{ds}: H3 裁剪消融 — {metric_label}"),
        ("ece", "ECE (lower is better)", f"{ds}: 校准误差 ECE"),
        ("member_corr", "成员预测相关性 (lower = more diverse)", f"{ds}: 成员多样性"),
    ]):
        ax = axes[axi]
        if col == "ece" and task == "regression":
            ax.axis("off")
            ax.set_title(f"{ds}: 回归任务不做 ECE")
            continue

        x_base = np.arange(len(CROP_SIZES))
        ys = []
        for gi, (kind, label, marker) in enumerate(GROUP_ORDER):
            means, stds = [], []
            for k in CROP_SIZES:
                m, s = stat(sub, kind, k, col)
                means.append(m)
                stds.append(s)

            if all(np.isnan(means)):
                continue

            offset = x_base + (gi - (n_group - 1) / 2) * width
            ax.errorbar(offset, means, yerr=stds, fmt=marker, capsize=4,
                        markersize=9, lw=1.6, label=label, color=COLORS[kind])
            for xv, mv in zip(offset, means):
                if not np.isnan(mv):
                    ax.annotate(f"{mv:.3f}", (xv, mv), textcoords="offset points",
                                xytext=(0, 10), ha="center", fontsize=8)
                    ys.append(mv)

        ax.set_xticks(x_base)
        ax.set_xticklabels([f"crop size = {k}" for k in CROP_SIZES])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fit_ylim(ax, ys)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"{ds}_h3_crop_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {out}")


def main():
    if not os.path.exists(CSV_OUT):
        print(f"找不到 {CSV_OUT}, 请先运行主实验与 H3 裁剪脚本")
        return

    df = pd.read_csv(CSV_OUT)
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        if not (sub["model_kind"].astype(str).str.startswith("tabm_crop")).any():
            print(f"[{ds}] 没有裁剪实验记录, 跳过 (先运行 scripts/run_crop_ablation.py)")
            continue
        print(f"\n绘图: {ds}")
        plot_dataset(ds, sub)

    print("\nH3 图绘制完成, 图片在 plots/ 目录")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("plot_h3_crop"):
        main()
