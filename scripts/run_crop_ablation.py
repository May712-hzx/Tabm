"""
H3 成员裁剪消融: 利用主实验训练好的 k=16 TabM, 只做推理, 不需要重新训练。

两种裁剪策略:
  1. tabm_crop_random  : 从 16 个成员里随机抽 crop_size 个
  2. tabm_crop_valbest : 在【验证集】上按单成员指标排序, 取最好的 crop_size 个
                         (严格只用验证集挑选, 不碰测试集)

对比基准: 主实验里原生训练的 tabm k=4 / k=8

用法(项目根目录):
    python scripts/run_crop_ablation.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    CKPT_DIR, CROP_SIZES, CSV_OUT, DATASETS, DROPOUT, D_BLOCK, H3_SOURCE_K,
    N_BLOCKS, SEEDS, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_tabm
from src.trainer import TabularTrainer, get_device
from src.utils import calc_metrics, save_result

FULL_K = H3_SOURCE_K   # 主实验保存权重用的是这个 k


def _row(dataset_name, seed, model_kind, k, metrics, task_type):
    r = {
        "dataset": dataset_name,
        "seed": seed,
        "model_kind": model_kind,
        "k": k,
        "train_time_s": np.nan,
        "params": np.nan,
    }
    r.update(metrics)
    r["ece"] = float(TabularTrainer.calc_ece(metrics["probs"], metrics["y_true"], task_type))
    r["member_corr"] = float(metrics["member_corr"])
    r.pop("probs")
    r.pop("y_true")
    return r


def run_single_crop(dataset_name, seed):
    print(f"\n===== H3 裁剪 | {dataset_name}, seed={seed} =====")

    # 注意: 解包 10 个值(dataset.py 的返回顺序包含 n_num_features)
    (X_train, X_val, X_test, y_train, y_val, y_test,
     n_num, cat_cards, task_type, d_out) = get_dataset(
        dataset_name, seed, max_samples=max_samples_of(dataset_name)
    )

    ckpt_path = os.path.join(CKPT_DIR, f"tabm_{dataset_name}_seed{seed}_k{FULL_K}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  跳过: {ckpt_path} 不存在, 请先跑完 scripts/run_exp.py 的 tabm k={FULL_K}")
        return

    model = build_tabm(n_num, cat_cards, d_out, k=FULL_K,
                       n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=DROPOUT)
    trainer = TabularTrainer(model, task_type, device=get_device(), verbose=False)
    saved_n_num = trainer.load_checkpoint(ckpt_path)
    if saved_n_num is not None:
        n_num = saved_n_num

    raw_val = trainer.get_raw_member_output(X_val, n_num)     # (N_val, 16, d_out)
    raw_test = trainer.get_raw_member_output(X_test, n_num)   # (N_test, 16, d_out)

    for crop_size in CROP_SIZES:
        # ---------------- 策略 1: 随机裁剪 ----------------
        rng = np.random.default_rng(seed * 100 + crop_size)
        sel_rand = rng.choice(FULL_K, size=crop_size, replace=False)

        pred_rand = trainer.predict_by_crop_members(raw_test, sel_rand)
        met_rand = calc_metrics(y_test, pred_rand, task_type, n_classes=d_out)
        met_rand["probs"] = pred_rand
        met_rand["y_true"] = y_test
        met_rand["member_corr"] = TabularTrainer.calc_member_correlation(
            raw_test[:, sel_rand, :], task_type
        )

        row_rand = _row(dataset_name, seed, "tabm_crop_random", crop_size, met_rand, task_type)
        print("[random crop]", row_rand)
        save_result(row_rand, CSV_OUT)

        # ---------------- 策略 2: 验证集择优裁剪 ----------------
        member_scores = []
        for mid in range(FULL_K):
            pred_val = trainer.predict_by_crop_members(raw_val, [mid])
            met_val = calc_metrics(y_val, pred_val, task_type, n_classes=d_out)
            score = met_val["r2"] if task_type == "regression" else met_val["macro_f1"]
            member_scores.append((mid, float(score)))

        member_scores.sort(key=lambda x: x[1], reverse=True)
        sel_best = [m for m, _ in member_scores[:crop_size]]

        pred_best = trainer.predict_by_crop_members(raw_test, sel_best)
        met_best = calc_metrics(y_test, pred_best, task_type, n_classes=d_out)
        met_best["probs"] = pred_best
        met_best["y_true"] = y_test
        met_best["member_corr"] = TabularTrainer.calc_member_correlation(
            raw_test[:, sel_best, :], task_type
        )

        row_best = _row(dataset_name, seed, "tabm_crop_valbest", crop_size, met_best, task_type)
        row_best["selected_members"] = "|".join(map(str, sel_best))
        print("[valbest crop]", row_best)
        save_result(row_best, CSV_OUT)


def main():
    failed = []
    for ds in DATASETS:
        for seed in SEEDS:
            try:
                run_single_crop(ds, seed)
            except Exception:
                import traceback
                failed.append((ds, seed))
                print(f"!! 跳过失败项: {ds} seed={seed}")
                traceback.print_exc()
    print("\nH3 裁剪实验完成")
    if failed:
        print("失败项:", failed)


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("run_crop_ablation"):
        main()
