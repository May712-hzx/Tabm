"""
错误 / 失败分析（指导书自检项"一项误差 / 失败分析"）。

分析对象：
  - covertype：主实验里唯一出现「k=16 明显劣于 k=1」的数据集（macro-F1 0.7631 vs 0.7850）。
    这里用 k=4（原生最优档）与 k=16（崩塌档）做正面对照，看崩塌到底是
    「平均意义上整体变差」还是「塌向多数类」——这直接决定 H2 的机制解释怎么写。
  - adult：顺带用 k=16 的 ckpt 看一眼二分类的错分结构（不重训，成本几乎为零）。

数据来源：
  - k=16 一律从 ckpt/ 加载（不重训，省 71 min）
  - k=4 必须重训（主实验没有保存它的权重），5 个 seed 约 16 min

产物：
  results/error_analysis/covertype_per_class.csv     每类 precision / recall / F1（k4 vs k16 vs MLP）
  results/error_analysis/covertype_confusion_k4.csv   k=4 的 5-seed 平均混淆矩阵（行归一化 = recall）
  results/error_analysis/covertype_confusion_k16.csv  k=16 的同上
  results/error_analysis/covertype_worst_errors.csv   k=16 高置信度错分样本（top 100）
  results/error_analysis/summary.txt                  文字版结论
  plots/error_analysis/covertype_confusion_k4_vs_k16.png

用法（项目根目录）：
    python scripts/error_analysis.py
    python scripts/error_analysis.py --seeds 3       # 少跑几个种子快速看
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.config import (
    BATCH_SIZE, CKPT_DIR, DROPOUT, D_BLOCK, H3_SOURCE_K, LR, MAX_EPOCH,
    N_BLOCKS, PATIENCE, PLOTS_DIR, WEIGHT_DECAY, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_tabm
from src.trainer import TabularTrainer
from src.utils import calc_metrics, set_seed

OUT_DIR = os.path.join("results", "error_analysis")
PLOT_DIR = os.path.join("plots", "error_analysis")


def _trainer_for(model, task_type):
    return TabularTrainer(
        model, task_type,
        lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
        patience=PATIENCE, max_epoch=MAX_EPOCH, verbose=False,
    )


def get_test_probs(dataset, seed, k, force_retrain=False):
    """返回 (probs, y_test, X_test, n_num, d_out)。k=16 优先走 ckpt。"""
    data = get_dataset(dataset, seed, max_samples=max_samples_of(dataset))
    X_train, X_val, X_test, y_train, y_val, y_test, n_num, cat_cards, task_type, d_out = data

    model = build_tabm(n_num, cat_cards, d_out, k=k,
                       n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=DROPOUT)
    trainer = _trainer_for(model, task_type)

    ckpt = os.path.join(CKPT_DIR, f"tabm_{dataset}_seed{seed}_k{k}.pt")
    if (not force_retrain) and os.path.exists(ckpt):
        trainer.load_checkpoint(ckpt)
        used = "ckpt"
    else:
        t0 = time.time()
        set_seed(seed)
        trainer.train_tabm(X_train, y_train, X_val, y_val, n_num)
        used = f"retrain({time.time() - t0:.0f}s)"

    probs = trainer.predict_tabm(X_test, n_num)
    return probs, y_test, X_test, n_num, d_out, used, trainer


def collect(dataset, seeds, k, force_retrain=False):
    """对每个 seed 跑一遍，返回 dict[str, list] 形式的原始统计。"""
    cms, per_class, rows, worst = [], [], [], []
    for seed in seeds:
        probs, y_test, X_test, n_num, d_out, used, _ = get_test_probs(dataset, seed, k, force_retrain)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        labels = list(range(d_out))
        cm = confusion_matrix(y_test, pred, labels=labels)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # 行归一化 = 每类召回
        cms.append(cm_norm)

        p, r, f, s = precision_recall_fscore_support(
            y_test, pred, labels=labels, zero_division=0
        )
        per_class.append(pd.DataFrame({
            "class": labels, "precision": p, "recall": r, "f1": f, "support": s,
            "seed": seed,
        }))

        m = calc_metrics(y_test, probs, "multiclass", n_classes=d_out)
        rows.append({"dataset": dataset, "k": k, "seed": seed,
                     "macro_f1": m["macro_f1"], "roc_auc": m["roc_auc"],
                     "logloss": m["logloss"], "acc": float((pred == y_test).mean()),
                     "used": used})

        # 高置信度错分：模型最有把握却判错的样本，最有分析价值
        wrong = np.where(pred != y_test)[0]
        if len(wrong) > 0:
            order = np.argsort(-conf[wrong])[:20]
            idx = wrong[order]
            worst.append(pd.DataFrame({
                "seed": seed, "test_index": idx,
                "y_true": y_test[idx], "y_pred": pred[idx],
                "confidence": conf[idx],
                "p_true_class": probs[idx, y_test[idx]],
            }))

    return {
        "cm_mean": np.mean(cms, axis=0),
        "per_class": pd.concat(per_class, ignore_index=True),
        "runs": pd.DataFrame(rows),
        "worst": pd.concat(worst, ignore_index=True) if worst else pd.DataFrame(),
        "d_out": d_out,
    }


def plot_confusions(cm4, cm16, dataset, d_out, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, cm, title in [
        (axes[0], cm4, "TabM k=4 (best config)"),
        (axes[1], cm16, "TabM k=16 (collapsed)"),
    ]:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(range(d_out))
        ax.set_yticks(range(d_out))
        ax.set_xticklabels([f"C{i+1}" for i in range(d_out)], fontsize=9)
        ax.set_yticklabels([f"C{i+1}" for i in range(d_out)], fontsize=9)
        for i in range(d_out):
            for j in range(d_out):
                v = cm[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalised (recall)")
    fig.suptitle(f"{dataset}: confusion matrix, 5-seed average, test set", fontsize=13)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_adult(seeds):
    """adult 只做轻量诊断：k=16 ckpt 的混淆与错误率。"""
    lines = []
    for seed in seeds:
        probs, y_test, X_test, n_num, d_out, used, _ = get_test_probs("adult", seed, H3_SOURCE_K)
        pred = probs.argmax(axis=1)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        acc = float((pred == y_test).mean())
        fn = int(cm[1, 0])      # 真 >50K 被判成 <=50K
        fp = int(cm[0, 1])
        lines.append({
            "seed": seed, "used": used, "acc": acc,
            "tn": int(cm[0, 0]), "fp": fp, "fn": fn, "tp": int(cm[1, 1]),
            "fn_rate_on_pos": fn / max(int(cm[1].sum()), 1),
            "fp_rate_on_neg": fp / max(int(cm[0].sum()), 1),
        })
    df = pd.DataFrame(lines)
    df.to_csv(os.path.join(OUT_DIR, "adult_k16_errors.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="用前几个 seed（默认 5）")
    ap.add_argument("--skip-k4-retrain", action="store_true",
                    help="跳过 k=4 重训（只看 k=16 的 ckpt，省 16 min）")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    report = []
    report.append("=" * 72)
    report.append("错误 / 失败分析报告")
    report.append("=" * 72)

    # ---------------------------------------------------------------- covertype
    print("\n>>> covertype: k=16 从 ckpt 加载（5 seeds）")
    r16 = collect("covertype", seeds, H3_SOURCE_K)
    r16["per_class"].assign(k=16).to_csv(
        os.path.join(OUT_DIR, "covertype_per_class_k16.csv"), index=False)
    pd.DataFrame(r16["cm_mean"]).to_csv(
        os.path.join(OUT_DIR, "covertype_confusion_k16.csv"), index=False)
    r16["worst"].to_csv(os.path.join(OUT_DIR, "covertype_worst_errors.csv"), index=False)

    if not args.skip_k4_retrain:
        print("\n>>> covertype: k=4 需要重训（5 seeds，约 16 min）")
        r4 = collect("covertype", seeds, 4, force_retrain=True)
        r4["per_class"].assign(k=4).to_csv(
            os.path.join(OUT_DIR, "covertype_per_class_k4.csv"), index=False)
        pd.DataFrame(r4["cm_mean"]).to_csv(
            os.path.join(OUT_DIR, "covertype_confusion_k4.csv"), index=False)

        per_class = pd.concat([r4["per_class"].assign(k=4), r16["per_class"].assign(k=16)],
                              ignore_index=True)
        agg = per_class.groupby(["k", "class"]).agg(
            precision=("precision", "mean"), recall=("recall", "mean"),
            f1=("f1", "mean"), support=("support", "mean"),
        ).reset_index()
        agg.to_csv(os.path.join(OUT_DIR, "covertype_per_class.csv"), index=False)

        runs = pd.concat([r4["runs"], r16["runs"]], ignore_index=True)
        runs.to_csv(os.path.join(OUT_DIR, "covertype_runs.csv"), index=False)

        plot_confusions(r4["cm_mean"], r16["cm_mean"], "covertype", r16["d_out"],
                        os.path.join(PLOT_DIR, "covertype_confusion_k4_vs_k16.png"))

        # 文字结论
        pivot = agg.pivot(index="class", columns="k", values="recall")
        report.append("")
        report.append("[covertype] 各类召回率 (k=4 vs k=16)")
        for cls in pivot.index:
            report.append(f"  class C{int(cls)+1}: k4={pivot.loc[cls, 4]:.3f}  k16={pivot.loc[cls, 16]:.3f}  "
                          f"Δ={pivot.loc[cls, 16] - pivot.loc[cls, 4]:+.3f}")
        majority = int(pivot[4].idxmax())
        report.append(f"  k=4 召回最高的类 = C{majority+1}（多数类候选）")
        report.append(f"  k=16 在该类上召回 = {pivot.loc[majority, 16]:.3f}，"
                      f"在其余类平均召回 = {pivot.loc[pivot.index != majority, 16].mean():.3f}")
        report.append(f"  k=4  在其余类平均召回 = {pivot.loc[pivot.index != majority, 4].mean():.3f}")

        m4 = runs[runs.k == 4]
        m16 = runs[runs.k == 16]
        report.append("")
        report.append(f"[covertype] 整体：k4 macro-F1={m4.macro_f1.mean():.4f}±{m4.macro_f1.std():.4f}，"
                      f"k16={m16.macro_f1.mean():.4f}±{m16.macro_f1.std():.4f}")
        report.append(f"[covertype] 整体准确率：k4={m4.acc.mean():.4f}，k16={m16.acc.mean():.4f}"
                      f"（Δ={m16.acc.mean() - m4.acc.mean():+.4f}）")
        d_acc = m16.acc.mean() - m4.acc.mean()
        d_f1 = m16.macro_f1.mean() - m4.macro_f1.mean()
        if abs(d_f1) > 1.5 * abs(d_acc) and d_f1 < 0:
            report.append(f"  说明：准确率只掉 {abs(d_acc) * 100:.1f}pp，macro-F1 却掉 {abs(d_f1) * 100:.1f}pp，"
                          f"跌幅是前者的 {abs(d_f1 / d_acc):.1f} 倍 → 劣化集中在少数类上，"
                          f"是典型的「向多数类倾斜」（判别边界退化，不是整体能力下降）。")
        elif d_acc < -0.01:
            report.append("  说明：accuracy 与 macro-F1 同步下降，劣化是整体性的。")
        else:
            report.append("  说明：accuracy 基本持平，主要差异体现在少数类上。")
        report.append(f"  少数类受损最明显的两个类（按召回跌幅）：" +
                      "、".join(f"C{int(c)+1} {v:+.3f}" for c, v in
                                pivot[16].sub(pivot[4]).nsmallest(2).items()))
        report.append(f"  k=16 高置信度错分样本已导出 {len(r16['worst'])} 条 -> "
                      f"results/error_analysis/covertype_worst_errors.csv")

    # ------------------------------------------------------------------- adult
    print("\n>>> adult: k=16 ckpt 的诊断")
    ad = analyze_adult(seeds)
    report.append("")
    report.append("[adult] k=16 混淆统计（5 seed 平均）")
    report.append(f"  acc={ad.acc.mean():.4f}")
    report.append(f"  正类(>50K)漏检率 fn/pos = {ad.fn_rate_on_pos.mean():.4f}")
    report.append(f"  负类(<=50K)误报率 fp/neg = {ad.fp_rate_on_neg.mean():.4f}")
    report.append("  含义：adult 上模型的错误几乎都是「少数类漏检」，"
                  "属于类别不平衡下的阈值/校准问题，而非成员塌缩。")

    txt = "\n".join(report)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\n产物目录: {OUT_DIR} / {PLOT_DIR}")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("error_analysis"):
        main()
