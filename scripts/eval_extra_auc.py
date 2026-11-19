"""
补齐 covertype 的第三个分类指标：macro-OvR AUC。

背景：
    主实验跑的时候 calc_metrics 只给二分类算 roc_auc，多分类的 roc_auc 列因此全是 NaN，
    导致 covertype 只有 logloss / macro_f1 两个指标，不满足"分类三指标齐全"的自检要求。
    现在 calc_metrics 已经支持多分类（每类 one-vs-rest 后对 AUC 取宏平均），
    但已经跑完的 csv 不会自己更新，所以要用这个脚本重算一遍并回填。

成本优化：
    k=16 的权重已经在 ckpt/ 里，直接加载推理（省 71 min）；
    k=1/4/8 + MLP + XGB + HistGB 必须重训，约 55~60 min（纯 CPU）。

产物：
    results/extra_metrics/covertype_auc.csv     本次重算的 AUC 与耗时
    results/extra_metrics/consistency.csv       重算的 macro_f1 与 csv 原值的差异（复现性校验）
    results/all_result.csv                      回填 roc_auc 列（先用 .bak 备份）

用法（项目根目录）：
    python scripts/eval_extra_auc.py                  # 只算，不回填
    python scripts/eval_extra_auc.py --fill           # 算完回填 all_result.csv
    python scripts/eval_extra_auc.py --fill --seeds 2 # 快速试跑
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import (
    BATCH_SIZE, CKPT_DIR, CSV_OUT, DROPOUT, D_BLOCK, H3_SOURCE_K, K_LIST,
    LR, MAX_EPOCH, N_BLOCKS, N_ESTIMATORS, PATIENCE, SEEDS,
    WEIGHT_DECAY, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_mlp, build_tabm
from src.trainer import TabularTrainer
from src.utils import calc_metrics, set_seed

DATASET = "covertype"
OUT_DIR = os.path.join("results", "extra_metrics")
AUC_CSV = os.path.join(OUT_DIR, "covertype_auc.csv")
CONS_CSV = os.path.join(OUT_DIR, "consistency.csv")


def _key(seed, kind, k):
    """k=None 统一记成 -1，便于跨进程续跑时比对。"""
    return (int(seed), str(kind), -1 if k is None else int(k))


def _load_done():
    """读已经算完的行；脚本被中断后可从这里接着跑。"""
    if not os.path.exists(AUC_CSV):
        return set()
    df = pd.read_csv(AUC_CSV)
    keys = set()
    for _, r in df.iterrows():
        kk = -1 if pd.isna(r["k"]) else int(r["k"])
        keys.add((int(r["seed"]), str(r["model_kind"]), kk))
    return keys


def _append_row(row, check):
    """逐条落盘，防止长时间运行被中断后全部丢失。"""
    pd.DataFrame([row]).to_csv(
        AUC_CSV, mode="a", header=not os.path.exists(AUC_CSV), index=False, encoding="utf-8-sig")
    pd.DataFrame([check]).to_csv(
        CONS_CSV, mode="a", header=not os.path.exists(CONS_CSV), index=False, encoding="utf-8-sig")


def _trainer(model, task_type):
    return TabularTrainer(
        model, task_type, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
        patience=PATIENCE, max_epoch=MAX_EPOCH, verbose=False,
    )


def compute_one(seed, kind, k, data):
    """返回 (metrics, 用时, 权重来源)。"""
    X_train, X_val, X_test, y_train, y_val, y_test, n_num, cat_cards, task_type, d_out = data
    set_seed(seed)
    t0 = time.time()

    if kind == "tabm":
        model = build_tabm(n_num, cat_cards, d_out, k=k,
                           n_blocks=N_BLOCKS, d_block=D_BLOCK, dropout=DROPOUT)
        tr = _trainer(model, task_type)
        ckpt = os.path.join(CKPT_DIR, f"tabm_{DATASET}_seed{seed}_k{k}.pt")
        if os.path.exists(ckpt):
            tr.load_checkpoint(ckpt)
            src = "ckpt"
        else:
            tr.train_tabm(X_train, y_train, X_val, y_val, n_num)
            src = "retrain"
        probs = tr.predict_tabm(X_test, n_num)

    elif kind == "mlp":
        model = build_mlp(n_num, cat_cards, d_out, hidden_dim=D_BLOCK,
                          n_blocks=N_BLOCKS, dropout=DROPOUT)
        tr = _trainer(model, task_type)
        tr.train_mlp(X_train, y_train, X_val, y_val, n_num)
        probs = tr.predict_mlp(X_test, n_num)
        src = "retrain"

    elif kind == "xgb":
        clf = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, random_state=seed, tree_method="hist",
            verbosity=0, n_jobs=-1, objective="multi:softprob", num_class=d_out,
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        src = "retrain"

    elif kind == "histgb":
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        src = "retrain"

    else:
        raise ValueError(kind)

    metrics = calc_metrics(y_test, probs, task_type, n_classes=d_out)
    return metrics, time.time() - t0, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--fill", action="store_true", help="把结果回填到 results/all_result.csv")
    args = ap.parse_args()

    seeds = SEEDS[: args.seeds]
    os.makedirs(OUT_DIR, exist_ok=True)
    data = get_dataset(DATASET, 0, max_samples=max_samples_of(DATASET))

    plan = [("xgb", None), ("histgb", None), ("mlp", None)] + [("tabm", k) for k in K_LIST]
    done_keys = _load_done()
    if done_keys:
        print(f"续跑：已有 {len(done_keys)} 条结果，自动跳过", flush=True)
    total = len(seeds) * len(plan)
    idx = 0

    for seed in seeds:
        d = get_dataset(DATASET, seed, max_samples=max_samples_of(DATASET))
        for kind, k in plan:
            idx += 1
            tag = f"{kind}(k={k})" if k else kind
            if _key(seed, kind, k) in done_keys:
                print(f"[{idx}/{total}] seed={seed} {tag}  (已有，跳过)", flush=True)
                continue
            print(f"[{idx}/{total}] seed={seed} {tag}", flush=True)
            try:
                m, dur, src = compute_one(seed, kind, k, d)
            except Exception as e:
                print(f"    !! 失败，跳过: {e}", flush=True)
                continue
            row = {
                "dataset": DATASET, "seed": seed, "model_kind": kind,
                "k": k if k is not None else np.nan,
                "roc_auc": m["roc_auc"], "macro_f1": m["macro_f1"],
                "logloss": m["logloss"], "eval_time_s": round(dur, 1), "weight_source": src,
            }
            check = {
                "seed": seed, "model_kind": kind,
                "k": k if k is not None else np.nan,
                "macro_f1_new": m["macro_f1"], "weight_source": src,
            }
            _append_row(row, check)      # 逐条落盘，中断不丢
            print(f"    auc={m['roc_auc']:.4f} f1={m['macro_f1']:.4f} ({src}, {dur:.0f}s)", flush=True)

    df = pd.read_csv(AUC_CSV).drop_duplicates(
        subset=["dataset", "seed", "model_kind", "k"], keep="last")
    df.to_csv(AUC_CSV, index=False, encoding="utf-8-sig")
    pd.read_csv(CONS_CSV).drop_duplicates(
        subset=["seed", "model_kind", "k"], keep="last"
    ).to_csv(CONS_CSV, index=False, encoding="utf-8-sig")
    rows = df.to_dict("records")
    print(f"\n算完 {len(df)} 条 -> {AUC_CSV}")

    if not args.fill:
        print("未加 --fill，不修改 all_result.csv")
        return

    # ------------------------------------------------------------- 回填 csv
    if not os.path.exists(CSV_OUT):
        print(f"!! 找不到 {CSV_OUT}，无法回填")
        return
    bak = CSV_OUT + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(CSV_OUT, bak)
        print(f"已备份 -> {bak}")

    df_all = pd.read_csv(CSV_OUT)
    auc_map = {}
    for r in rows:
        kk = "nan" if (r["k"] is None or (isinstance(r["k"], float) and not np.isfinite(r["k"]))) else int(r["k"])
        auc_map[(r["seed"], r["model_kind"], str(kk))] = r["roc_auc"]

    filled = 0
    for i, row in df_all.iterrows():
        if row["dataset"] != DATASET:
            continue
        kk = "nan" if not np.isfinite(row["k"]) else int(row["k"])
        key = (int(row["seed"]), row["model_kind"], str(kk))
        if key in auc_map:
            v = auc_map[key]
            if not np.isfinite(row["roc_auc"]) or abs(row["roc_auc"] - v) > 1e-9:
                df_all.at[i, "roc_auc"] = v
                filled += 1

    df_all.to_csv(CSV_OUT, index=False)
    print(f"回填完成，更新 {filled} 行 -> {CSV_OUT}")
    print("校验：covertype 仍有 NaN 的行数 =",
          int(df_all[(df_all.dataset == DATASET)].roc_auc.isna().sum()))


if __name__ == "__main__":
    # 注意：正在跑的进程不读这行；下次启动时会自动在 results/logs/ 留一份完整日志
    from src.run_log import run_log
    with run_log("eval_extra_auc"):
        main()
