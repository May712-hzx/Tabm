"""
H2 多样性补充指标：逐样本概率方差（指导书要求之外、用来补 member_corr 的不足）。

对同一个测试集、同一划分，比两种「16 个成员」的成员分歧：
  - source = tabm_k16     : 一个 TabM 的 16 个成员(head)，权重来自 ckpt/（不训练）
  - source = indep_mlp_16 : 16 个完全独立训练的 MLP，逐成员概率来自
                            results/extension/member_probs_<ds>_seed<s>.npz

指标（每个 (数据集, seed) 一行）：
  mean_prob_var   逐样本、逐类别的跨成员方差再平均（越大 = 成员分歧越大 = 多样性越高）
  frac_disagree   成员 argmax 类别不完全一致的样本占比（分类；回归为 NaN）
  member_corr     成员预测相关性均值（越大 = 越趋同），口径与 all_result.csv 一致

产物：results/variance/member_variance.csv

用法（项目根目录）：
    python scripts/eval_member_variance.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    CKPT_DIR, DATASETS, D_BLOCK, DROPOUT, N_BLOCKS, SEEDS, CSV_OUT, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_tabm
from src.trainer import TabularTrainer, get_device

VAR_DIR = os.path.join(ROOT, "results", "variance")
VAR_CSV = os.path.join(VAR_DIR, "member_variance.csv")
EXT_DIR = os.path.join(ROOT, "results", "extension")
FULL_K = 16


def _stats(probs, task_type):
    """probs: (N, k, d) —— 分类为概率，回归为预测值。"""
    probs = np.asarray(probs, dtype=np.float64)
    var = np.var(probs, axis=1)                 # (N, d)
    mean_var = float(var.mean())
    if task_type == "regression":
        frac = float("nan")
    else:
        pred = probs.argmax(axis=-1)            # (N, k)
        frac = float((pred.max(axis=1) != pred.min(axis=1)).mean())
    corr = TabularTrainer.calc_member_correlation(probs, task_type)
    return mean_var, frac, corr


def tabm_probs(ds, seed, task_type, d_out):
    ckpt = os.path.join(CKPT_DIR, f"tabm_{ds}_seed{seed}_k{FULL_K}.pt")
    if not os.path.exists(ckpt):
        return None
    (_, _, X_test, _, _, _, n_num, cat_cards, _, _) = get_dataset(
        ds, seed, max_samples=max_samples_of(ds))
    model = build_tabm(n_num, cat_cards, d_out, k=FULL_K, n_blocks=N_BLOCKS,
                       d_block=D_BLOCK, dropout=DROPOUT)
    tr = TabularTrainer(model, task_type, device=get_device(), verbose=False)
    saved = tr.load_checkpoint(ckpt)
    if saved is not None:
        n_num = saved
    raw = tr.get_raw_member_output(X_test, n_num)      # (N, k, d_out)
    if task_type == "regression":
        return raw
    return TabularTrainer.stable_softmax(raw)


def indep_mlp_probs(ds, seed):
    p = os.path.join(EXT_DIR, f"member_probs_{ds}_seed{seed}.npz")
    if not os.path.exists(p):
        return None
    arr = np.load(p)["test"]                            # (k, N, d)
    return np.transpose(arr, (1, 0, 2))


def main():
    os.makedirs(VAR_DIR, exist_ok=True)
    df_all = pd.read_csv(CSV_OUT) if os.path.exists(CSV_OUT) else pd.DataFrame()

    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            (_, _, X_test, _, _, _, n_num, cat_cards, task_type, d_out) = get_dataset(
                ds, s, max_samples=max_samples_of(ds))
            n_test = len(X_test)

            tp = tabm_probs(ds, s, task_type, d_out)
            if tp is not None:
                mv, frac, corr = _stats(tp, task_type)
                rows.append({
                    "dataset": ds, "seed": s, "source": "tabm_k16", "k": FULL_K,
                    "n_test": n_test, "mean_prob_var": mv,
                    "frac_disagree": frac, "member_corr": corr,
                })
                print(f"  {ds} s{s} tabm_k16      var={mv:.5f} "
                      f"disagree={frac if np.isfinite(frac) else float('nan'):.3f} "
                      f"corr={corr:.3f}")

            ip = indep_mlp_probs(ds, s)
            if ip is not None:
                mv, frac, corr = _stats(ip, task_type)
                rows.append({
                    "dataset": ds, "seed": s, "source": "indep_mlp_16", "k": FULL_K,
                    "n_test": n_test, "mean_prob_var": mv,
                    "frac_disagree": frac, "member_corr": corr,
                })
                print(f"  {ds} s{s} indep_mlp_16  var={mv:.5f} "
                      f"disagree={frac if np.isfinite(frac) else float('nan'):.3f} "
                      f"corr={corr:.3f}")

    if not rows:
        print("没有可计算的成员概率，请先跑主实验与 extension/run_indep_mlp.py")
        return

    out = pd.DataFrame(rows)
    out.to_csv(VAR_CSV, index=False, encoding="utf-8-sig")

    print("\n=== 按数据集+来源平均 ===")
    g = out.groupby(["dataset", "source"]).agg(
        var_mean=("mean_prob_var", "mean"),
        disagree=("frac_disagree", "mean"),
        corr=("member_corr", "mean"),
        n=("seed", "count"),
    ).round(5)
    print(g.to_string())

    if len(df_all):
        ref = df_all[(df_all["model_kind"] == "tabm") & (df_all["k"] == 16)]
        if len(ref):
            print("\n=== TabM k=16 在 all_result.csv 里记录的 member_corr（对照）===")
            print(ref.groupby("dataset")["member_corr"].mean().round(4).to_string())

    print(f"\n已写入 {VAR_CSV}")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("eval_member_variance"):
        main()
