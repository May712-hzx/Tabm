"""
补齐 H3 裁剪行（tabm_crop_random / tabm_crop_valbest）的 roc_auc。

这些行是在 calc_metrics 支持多分类 AUC 之前生成的，roc_auc 列因此是空的。
裁剪只是成员子集推理，不需要训练：加载 ckpt/ 里 k=16 的权重，
按与 run_crop_ablation.py 完全一致的成员选择逻辑重算一遍，只回填 roc_auc 列。

用法（项目根目录）：
    python scripts/fill_crop_auc.py            # 只打印
    python scripts/fill_crop_auc.py --fill     # 回填 results/all_result.csv
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    CKPT_DIR, CROP_SIZES, CSV_OUT, D_BLOCK, DATASETS, DROPOUT, H3_SOURCE_K,
    N_BLOCKS, SEEDS, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_tabm
from src.trainer import TabularTrainer, get_device
from src.utils import calc_metrics

FULL_K = H3_SOURCE_K
CLASS_DATASETS = {"adult", "covertype"}


def _auc(probs, y_true, task_type, d_out):
    if task_type == "binclass":
        return float(roc_auc_score(y_true, probs[:, 1]))
    if task_type == "multiclass":
        return float(roc_auc_score(y_true, probs, multi_class="ovr",
                                   average="macro", labels=list(range(d_out))))
    return float("nan")


def run_one(dataset_name, seed):
    if dataset_name not in CLASS_DATASETS:
        return []
    (X_train, X_val, X_test, y_train, y_val, y_test,
     n_num, cat_cards, task_type, d_out) = get_dataset(
        dataset_name, seed, max_samples=max_samples_of(dataset_name))

    ckpt = os.path.join(CKPT_DIR, f"tabm_{dataset_name}_seed{seed}_k{FULL_K}.pt")
    if not os.path.exists(ckpt):
        print(f"  跳过 {dataset_name} seed={seed}: 缺 ckpt")
        return []

    model = build_tabm(n_num, cat_cards, d_out, k=FULL_K,
                       n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=DROPOUT)
    tr = TabularTrainer(model, task_type, device=get_device(), verbose=False)
    saved = tr.load_checkpoint(ckpt)
    if saved is not None:
        n_num = saved

    raw_val = tr.get_raw_member_output(X_val, n_num)
    raw_test = tr.get_raw_member_output(X_test, n_num)

    out = []
    for crop in CROP_SIZES:
        # 随机裁剪：与 run_crop_ablation.py 同一个 rng 口径
        rng = np.random.default_rng(seed * 100 + crop)
        sel_rand = rng.choice(FULL_K, size=crop, replace=False)
        p_rand = tr.predict_by_crop_members(raw_test, sel_rand)
        out.append((f"tabm_crop_random", crop, seed,
                    _auc(p_rand, y_test, task_type, d_out)))

        # val 择优：与主口径一致（分类取 val macro-F1）
        scores = []
        for mid in range(FULL_K):
            pv = tr.predict_by_crop_members(raw_val, [mid])
            m = calc_metrics(y_val, pv, task_type, n_classes=d_out)
            scores.append((mid, float(m["macro_f1"])))
        scores.sort(key=lambda x: x[1], reverse=True)
        sel_best = [m for m, _ in scores[:crop]]
        p_best = tr.predict_by_crop_members(raw_test, sel_best)
        out.append(("tabm_crop_valbest", crop, seed,
                    _auc(p_best, y_test, task_type, d_out)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(CSV_OUT):
        print(f"找不到 {CSV_OUT}")
        return

    results = []
    for ds in DATASETS:
        for s in SEEDS:
            print(f"推理 {ds} seed={s}")
            # 带上数据集名，避免不同数据集的同名 (model_kind, k, seed) 互相覆盖
            results += [(ds, kind, crop, seed, v)
                        for kind, crop, seed, v in run_one(ds, s)]

    df = pd.read_csv(CSV_OUT)
    auc_map = {(ds, kind, int(k), int(seed)): v
               for ds, kind, k, seed, v in results}

    n = 0
    for i, row in df.iterrows():
        kind = str(row["model_kind"])
        if not kind.startswith("tabm_crop"):
            continue
        if not np.isfinite(row["k"]):
            continue
        key = (str(row["dataset"]), kind, int(row["k"]), int(row["seed"]))
        if key in auc_map and np.isfinite(auc_map[key]):
            df.at[i, "roc_auc"] = auc_map[key]
            n += 1

    print(f"可回填 {n} 行")
    if args.fill:
        bak = CSV_OUT + ".bak2"
        if not os.path.exists(bak):
            shutil.copy2(CSV_OUT, bak)
            print(f"已备份 -> {bak}")
        df.to_csv(CSV_OUT, index=False)
        print(f"已写入 {CSV_OUT}")
        still = df[df["model_kind"].astype(str).str.startswith("tabm_crop")]["roc_auc"].isna().sum()
        print(f"校验：crop 行仍缺 AUC = {int(still)}")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("fill_crop_auc"):
        main()
