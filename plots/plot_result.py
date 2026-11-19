"""
读取 results/all_result.csv, 画主实验图:
  1. 主指标 vs k (TabM 曲线 + MLP/XGB/HistGBM 基线水平线, 带多种子标准差)
  2. 损失类指标 vs k (logloss / rmse)
  3. 训练时间 vs k
  4. ECE vs k (仅分类)

用法(项目根目录):
    python plots/plot_result.py
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

from src.config import CSV_OUT, PLOTS_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASELINES = ["xgb", "histgb", "mlp"]

# 三种基线必须可区分: 颜色 + 线型双编码
BASE_STYLE = {
    "xgb": {"color": "#d62728", "ls": "-", "label": "XGBoost"},
    "histgb": {"color": "#9467bd", "ls": "-.", "label": "HistGBM"},
    "mlp": {"color": "#8c564b", "ls": ":", "label": "MLP"},
}
TABM_COLOR = "#1f77b4"


def task_of(sub):
    """根据 csv 里是否有回归列判断任务类型。"""
    if "r2" in sub.columns and sub["r2"].notna().any():
        return "regression"
    return "classification"


def aggregate(sub, cols):
    """按 (model_kind, k) 聚合为 mean/std, 返回扁平列名的 DataFrame。"""
    sub = sub.copy()
    sub["k"] = sub["k"].fillna(-1)
    agg = {c: ["mean", "std"] for c in cols if c in sub.columns}
    g = sub.groupby(["model_kind", "k"]).agg(agg)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def fit_ylim(ax, values, pad_ratio=0.08):
    """把画过的所有 y (曲线 + 基线) 都纳入坐标范围, 避免基线被裁到画布外。"""
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else (abs(hi) * 0.1 + 1e-9)
    ax.set_ylim(lo - span * pad_ratio, hi + span * pad_ratio)


def plot_dataset(ds, sub):
    task = task_of(sub)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    cols_needed = ["train_time_s", "params", "ece"]
    if task == "regression":
        cols_needed += ["r2", "rmse", "mae"]
        main_metric, loss_metric, main_ylabel = "r2", "rmse", "R2"
    else:
        cols_needed += ["macro_f1", "logloss", "roc_auc"]
        main_metric, loss_metric, main_ylabel = "macro_f1", "logloss", "Macro-F1"

    agg = aggregate(sub, cols_needed)
    tabm = agg[agg["model_kind"] == "tabm"].copy()
    tabm = tabm[tabm["k"] > 0].sort_values("k")
    if tabm.empty:
        print(f"[{ds}] 没有 tabm 记录, 跳过绘图")
        return
    base = agg[agg["model_kind"].isin(BASELINES)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    # ---- (1) 主指标 vs k ----
    ax = axes[0]
    ys = list(tabm[f"{main_metric}_mean"])
    ax.errorbar(tabm["k"], tabm[f"{main_metric}_mean"], yerr=tabm[f"{main_metric}_std"],
                marker="o", capsize=4, lw=2, color=TABM_COLOR, label="TabM (5 seeds)")
    for kind in BASELINES:
        row = base[base["model_kind"] == kind]
        if row.empty or pd.isna(row[f"{main_metric}_mean"].iloc[0]):
            continue
        v = float(row[f"{main_metric}_mean"].iloc[0])
        st = BASE_STYLE[kind]
        ax.axhline(v, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
        ys.append(v)
    ax.set_xlabel("k (# ensemble members)")
    ax.set_ylabel(main_ylabel)
    ax.set_title(f"{ds}: {main_ylabel} vs k")
    ax.set_xticks(tabm["k"].astype(int).tolist())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fit_ylim(ax, ys)

    # ---- (2) 损失类指标 vs k ----
    ax = axes[1]
    ys = []
    if f"{loss_metric}_mean" in tabm.columns:
        ys += list(tabm[f"{loss_metric}_mean"])
        ax.errorbar(tabm["k"], tabm[f"{loss_metric}_mean"], yerr=tabm[f"{loss_metric}_std"],
                    marker="s", capsize=4, lw=2, color=TABM_COLOR, label="TabM")
    for kind in BASELINES:
        row = base[base["model_kind"] == kind]
        if row.empty or pd.isna(row[f"{loss_metric}_mean"].iloc[0]):
            continue
        v = float(row[f"{loss_metric}_mean"].iloc[0])
        st = BASE_STYLE[kind]
        ax.axhline(v, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
        ys.append(v)
    ax.set_xlabel("k (# ensemble members)")
    ax.set_ylabel(loss_metric.upper())
    ax.set_title(f"{ds}: {loss_metric.upper()} vs k (lower is better)")
    ax.set_xticks(tabm["k"].astype(int).tolist())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fit_ylim(ax, ys, pad_ratio=0.12)

    # ---- (3) 训练时间 vs k ----
    ax = axes[2]
    ys = list(tabm["train_time_s_mean"])
    ax.errorbar(tabm["k"], tabm["train_time_s_mean"], yerr=tabm["train_time_s_std"],
                marker="^", capsize=4, lw=2, color=TABM_COLOR, label="TabM")
    for kind in BASELINES:
        row = base[base["model_kind"] == kind]
        if row.empty or pd.isna(row["train_time_s_mean"].iloc[0]):
            continue
        v = float(row["train_time_s_mean"].iloc[0])
        st = BASE_STYLE[kind]
        ax.axhline(v, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
        ys.append(v)
    ax.set_xlabel("k (# ensemble members)")
    ax.set_ylabel("Training time (s)")
    ax.set_title(f"{ds}: training time vs k")
    ax.set_xticks(tabm["k"].astype(int).tolist())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fit_ylim(ax, ys, pad_ratio=0.10)

    # ---- (4) ECE vs k (仅分类) ----
    ax = axes[3]
    if task != "regression" and f"ece_mean" in tabm.columns:
        ys = list(tabm["ece_mean"].dropna())
        ax.errorbar(tabm["k"], tabm["ece_mean"], yerr=tabm["ece_std"],
                    marker="D", capsize=4, lw=2, color=TABM_COLOR, label="TabM")
        for kind in BASELINES:
            row = base[base["model_kind"] == kind]
            if row.empty or "ece_mean" not in row.columns or pd.isna(row["ece_mean"].iloc[0]):
                continue
            v = float(row["ece_mean"].iloc[0])
            st = BASE_STYLE[kind]
            ax.axhline(v, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
            ys.append(v)
        ax.set_xlabel("k (# ensemble members)")
        ax.set_ylabel("ECE (lower is better)")
        ax.set_title(f"{ds}: calibration (ECE) vs k")
        ax.set_xticks(tabm["k"].astype(int).tolist())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fit_ylim(ax, ys, pad_ratio=0.12)
    else:
        ax.axis("off")
        ax.set_title(f"{ds}: regression task, ECE not applicable")

    fig.suptitle(f"{ds} — TabM ensemble-size ablation (mean ± std over seeds)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{ds}_tabm_k_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {out}")


def main():
    if not os.path.exists(CSV_OUT):
        print(f"找不到 {CSV_OUT}, 请先运行 scripts/run_exp.py")
        return

    df = pd.read_csv(CSV_OUT)
    # H3 裁剪的行不属于主实验, 这里排除
    df = df[~df["model_kind"].astype(str).str.startswith("tabm_crop")]

    for ds in sorted(df["dataset"].unique()):
        print(f"\n绘图: {ds}")
        plot_dataset(ds, df[df["dataset"] == ds])

    print("\n绘图完成, 图片在 plots/ 目录")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("plot_result"):
        main()
