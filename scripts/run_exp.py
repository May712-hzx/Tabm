"""
主实验脚本: 3 个数据集 × 5 个种子 × {XGBoost, HistGBM, MLP, TabM(k=1,4,8,16)}

用法(在项目根目录 D:\\project_tabm 下):
    python scripts/run_exp.py

输出:
    results/all_result.csv   # 原始结果, 一行一次运行
    ckpt/tabm_<ds>_seed<s>_k16.pt   # k=16 的 TabM 权重, 供 H3 裁剪脚本使用
"""

import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.config import (
    BATCH_SIZE, CKPT_DIR, CSV_OUT, DATASETS, DROPOUT, D_BLOCK, H3_SOURCE_K,
    K_LIST, LR, MAX_EPOCH, N_BLOCKS, N_ESTIMATORS, PATIENCE,
    SEEDS, WEIGHT_DECAY, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_mlp, build_tabm
from src.trainer import TabularTrainer
from src.utils import calc_metrics, count_params, save_result, set_seed


def _train_neural(model, model_kind, data, seed):
    """训练 + 推理 + 指标, 返回 (metrics, train_time, n_params, probs, raw_members)"""
    X_train, X_val, X_test, y_train, y_val, y_test, n_num, cat_cards, task_type, d_out = data

    trainer = TabularTrainer(
        model, task_type,
        lr=LR, weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE, patience=PATIENCE, max_epoch=MAX_EPOCH,
        verbose=False,
    )

    t0 = time.time()
    if model_kind == "tabm":
        trainer.train_tabm(X_train, y_train, X_val, y_val, n_num)
    else:
        trainer.train_mlp(X_train, y_train, X_val, y_val, n_num)
    train_time = time.time() - t0

    if model_kind == "tabm":
        probs = trainer.predict_tabm(X_test, n_num)
        raw = trainer.get_raw_member_output(X_test, n_num)
    else:
        probs = trainer.predict_mlp(X_test, n_num)
        raw = None

    metrics = calc_metrics(y_test, probs, task_type, n_classes=d_out)
    metrics["ece"] = TabularTrainer.calc_ece(probs, y_test, task_type)
    metrics["member_corr"] = (
        TabularTrainer.calc_member_correlation(raw, task_type) if raw is not None else float("nan")
    )
    return metrics, train_time, count_params(model), trainer


def run_one_exp(dataset_name, seed, model_kind, k=None):
    data = get_dataset(dataset_name, seed, max_samples=max_samples_of(dataset_name))
    X_train, X_val, X_test, y_train, y_val, y_test, n_num, cat_cards, task_type, d_out = data

    row = {
        "dataset": dataset_name,
        "seed": seed,
        "model_kind": model_kind,
        "k": k if k is not None else np.nan,
    }

    t0 = time.time()

    if model_kind == "tabm":
        model = build_tabm(n_num, cat_cards, d_out, k=k,
                           n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=DROPOUT)
        metrics, train_time, n_param, trainer = _train_neural(model, "tabm", data, seed)

        # 只保存 H3 需要的那一档权重(默认 k=16), 省磁盘省时间
        if k == H3_SOURCE_K:
            ckpt_path = os.path.join(CKPT_DIR, f"tabm_{dataset_name}_seed{seed}_k{k}.pt")
            trainer.save_checkpoint(ckpt_path, n_num)
            print(f"    [ckpt] {ckpt_path}")

    elif model_kind == "mlp":
        model = build_mlp(n_num, cat_cards, d_out,
                          hidden_dim=D_BLOCK, n_blocks=N_BLOCKS, dropout=DROPOUT)
        metrics, train_time, n_param, _ = _train_neural(model, "mlp", data, seed)

    elif model_kind == "xgb":
        if task_type == "binclass":
            clf = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS, random_state=seed,
                tree_method="hist", verbosity=0, n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
        elif task_type == "multiclass":
            clf = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS, random_state=seed,
                tree_method="hist", verbosity=0, n_jobs=-1,
                objective="multi:softprob", num_class=d_out,
            )
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
        else:
            reg = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS, random_state=seed,
                tree_method="hist", verbosity=0, n_jobs=-1,
            )
            reg.fit(X_train, y_train)
            probs = reg.predict(X_test).reshape(-1, 1)

        train_time = time.time() - t0
        n_param = 0
        metrics = calc_metrics(y_test, probs, task_type, n_classes=d_out)
        metrics["ece"] = TabularTrainer.calc_ece(probs, y_test, task_type)
        metrics["member_corr"] = float("nan")

    elif model_kind == "histgb":
        if task_type in ("binclass", "multiclass"):
            clf = HistGradientBoostingClassifier(random_state=seed)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
        else:
            reg = HistGradientBoostingRegressor(random_state=seed)
            reg.fit(X_train, y_train)
            probs = reg.predict(X_test).reshape(-1, 1)

        train_time = time.time() - t0
        n_param = 0
        metrics = calc_metrics(y_test, probs, task_type, n_classes=d_out)
        metrics["ece"] = TabularTrainer.calc_ece(probs, y_test, task_type)
        metrics["member_corr"] = float("nan")

    else:
        raise ValueError(f"未知模型类型: {model_kind}")

    row.update(metrics)
    row["train_time_s"] = round(train_time, 3)
    row["params"] = n_param

    print(row)
    save_result(row, CSV_OUT)
    return row


def main():
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    if H3_SOURCE_K not in K_LIST:
        print(f"!! 警告: H3_SOURCE_K={H3_SOURCE_K} 不在 K_LIST={K_LIST} 中, "
              f"将不会保存 H3 裁剪所需的权重, run_crop_ablation.py 会跳过所有项")

    model_plan = ["xgb", "histgb", "mlp"] + [("tabm", k) for k in K_LIST]

    total = len(DATASETS) * len(SEEDS) * len(model_plan)
    done = 0
    failed = []

    for ds in DATASETS:
        for seed in SEEDS:
            set_seed(seed)
            for item in model_plan:
                done += 1
                if isinstance(item, tuple):
                    kind, k = item
                else:
                    kind, k = item, None

                tag = f"{kind}(k={k})" if k is not None else kind
                print(f"\n[{done}/{total}] dataset={ds} seed={seed} model={tag}")
                try:
                    run_one_exp(ds, seed, kind, k=k)
                except Exception:
                    failed.append((ds, seed, tag))
                    print(f"!! 运行失败, 已跳过: {ds} / seed={seed} / {tag}")
                    traceback.print_exc()

    print("\n========== 全部结束 ==========")
    print(f"成功 {total - len(failed)} / {total}")
    if failed:
        print("失败列表:")
        for f in failed:
            print("  ", f)


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("run_exp"):
        main()
