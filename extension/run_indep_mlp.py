"""
[拓展板块 · 独立] 与 k 个独立 MLP 的严格成本对照

目的
----
论文的核心主张是「参数高效集成」：TabM 用远小于 k× 的参数达到接近 k 个独立模型的集成效果。
本脚本提供一个严格的上界参照 —— 用 k 个**完全独立**的 MLP（各自初始化、各自训练）做同口径集成，
量化「多花多少参数和时间，换来多少性能」。

与主实验的隔离
--------------
- 不修改 src/ 下任何文件，不写 results/all_result.csv，不写 plots/ 下已有图
- 只读取 src.config 的划分/下采样配置，保证与 TabM 的 train/val/test 完全一致
- 输出全部落在 results/extension/ 与 plots/extension/

严格性设计
----------
1. 数据划分、预处理、早停预算、优化器超参 与主实验完全相同（复用 src.dataset + src.trainer）
2. 只训练 **K_MEMBERS=16 个独立成员**，然后取「前 k 个成员」做 k∈{1,4,8,16} 的集成，
   成员训练过程互不知晓 k，因此不同 k 之间不存在额外训练量差异
3. 成本口径：
   - params      = k × 单成员参数量（真实部署需要 k 份权重）
   - train_time  = 前 k 个成员训练时间之和（真实串行训练成本）
   - infer_ms    = k 次独立前向的实测总耗时
4. 集成方式与 TabM 一致：分类先 softmax 再平均概率，回归直接平均预测值

用法(项目根目录)
----------------
    python extension/run_indep_mlp.py
    python extension/run_indep_mlp.py --datasets adult --seeds 0 1
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    BATCH_SIZE, DATASETS, DROPOUT, D_BLOCK, LR, MAX_EPOCH, N_BLOCKS,
    PATIENCE, SEEDS, WEIGHT_DECAY, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_mlp
from src.trainer import TabularTrainer
from src.utils import calc_metrics, count_params, save_result, set_seed

EXT_DIR = os.path.join(ROOT, "results", "extension")
CSV_OUT = os.path.join(EXT_DIR, "indep_mlp_result.csv")

K_MEMBERS = 16          # 训练多少个独立成员
K_EVAL = [1, 4, 8, 16]  # 用前 k 个成员做集成评估


def _peak_rss_mb():
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / 1024 ** 2
    except Exception:
        return float("nan")


def _ensemble_predict(probs_list, task_type):
    """与 TabM 相同的聚合方式。probs_list 内每个元素是 (N, d_out) 的概率/预测。"""
    if task_type in ("binclass", "multiclass"):
        return np.mean(np.stack(probs_list, axis=0), axis=0)
    return np.mean(np.stack(probs_list, axis=0), axis=0)


def run_one(dataset_name, seed, k_members=K_MEMBERS):
    print(f"\n===== [extension] 独立 MLP 集成 | {dataset_name}, seed={seed} =====")
    (X_train, X_val, X_test, y_train, y_val, y_test,
     n_num, cat_cards, task_type, d_out) = get_dataset(
        dataset_name, seed, max_samples=max_samples_of(dataset_name)
    )

    members_probs_test = []
    members_probs_val = []
    member_metrics = []
    member_times = []
    member_infer_s = []
    member_params = []
    peak_rss = _peak_rss_mb()

    # ---------- 训练 K_MEMBERS 个完全独立的 MLP ----------
    for m in range(k_members):
        set_seed(seed * 1000 + m)
        model = build_mlp(n_num, cat_cards, d_out,
                          hidden_dim=D_BLOCK, n_blocks=N_BLOCKS, dropout=DROPOUT)
        trainer = TabularTrainer(
            model, task_type,
            lr=LR, weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE, patience=PATIENCE, max_epoch=MAX_EPOCH,
            verbose=False,
        )

        t0 = time.perf_counter()
        trainer.train_mlp(X_train, y_train, X_val, y_val, n_num)
        member_times.append(time.perf_counter() - t0)

        # 推理计时: 先预热一次, 再测第二次, 避免 dataloader 首次构建开销污染
        trainer.model.eval()
        _ = trainer.predict_mlp(X_test, n_num)
        t_inf = time.perf_counter()
        p_test = trainer.predict_mlp(X_test, n_num)
        member_infer_s.append(time.perf_counter() - t_inf)

        p_val = trainer.predict_mlp(X_val, n_num)
        members_probs_test.append(p_test)
        members_probs_val.append(p_val)

        mt = calc_metrics(y_test, p_test, task_type, n_classes=d_out)
        member_metrics.append(mt)
        member_params.append(count_params(model))

        peak_rss = max(peak_rss, _peak_rss_mb())
        line = "  ".join(f"{k}={v:.4f}" for k, v in mt.items())
        print(f"  member {m + 1:2d}/{k_members}  {line}  t={member_times[-1]:.1f}s")

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    single_params = int(np.mean(member_params))

    # ---------- 前 k 个成员集成 ----------
    for k in [x for x in K_EVAL if x <= k_members]:
        sel_p_test = members_probs_test[:k]
        ens_test = _ensemble_predict(sel_p_test, task_type)

        metrics = calc_metrics(y_test, ens_test, task_type, n_classes=d_out)
        metrics["ece"] = TabularTrainer.calc_ece(ens_test, y_test, task_type)

        # member_corr: 把 k 个成员的概率堆成 (N, k, d_out) 后按主实验同口径计算
        if task_type in ("binclass", "multiclass"):
            raw = np.stack(sel_p_test, axis=1)      # 已是概率, argmax 一致
            metrics["member_corr"] = TabularTrainer.calc_member_correlation(raw, task_type)
        else:
            raw = np.stack(sel_p_test, axis=1)
            metrics["member_corr"] = TabularTrainer.calc_member_correlation(raw, task_type)

        # 集成增益: 单成员指标均值 -> 集成指标
        main = "r2" if task_type == "regression" else "macro_f1"
        single_mean = float(np.mean([mm[main] for mm in member_metrics[:k]]))
        metrics[f"single_member_{main}_mean"] = single_mean
        metrics[f"ensemble_gain_{main}"] = float(metrics[main]) - single_mean

        # ---------- 推理成本实测(k 次独立前向之和) ----------
        infer_k_s = float(sum(member_infer_s[:k]))
        metrics["infer_ms_total"] = float(infer_k_s * 1000)
        metrics["infer_ms_per_sample"] = float(infer_k_s * 1000 / len(X_test))
        metrics["train_time_per_member_avg"] = float(np.mean(member_times[:k]))

        row = {
            "dataset": dataset_name,
            "seed": seed,
            "model_kind": "indep_mlp",
            "k": k,
            "train_time_s": round(float(sum(member_times[:k])), 3),
            "params": int(single_params * k),
            "single_params": single_params,
            "peak_rss_mb": round(float(peak_rss), 1),
        }
        row.update({kk: vv for kk, vv in metrics.items()})
        print(f"  [ens k={k:2d}] " + "  ".join(
            f"{kk}={vv:.4f}" for kk, vv in metrics.items() if isinstance(vv, float)))
        save_result(row, CSV_OUT)

    # ---------- 同时落盘逐成员预测, 供后续多样性分析 ----------
    np.savez_compressed(
        os.path.join(EXT_DIR, f"member_probs_{dataset_name}_seed{seed}.npz"),
        test=np.stack(members_probs_test, axis=0).astype(np.float32),
        val=np.stack(members_probs_val, axis=0).astype(np.float32),
        y_test=y_test, y_val=y_val,
    )
    print(f"  逐成员预测已保存: member_probs_{dataset_name}_seed{seed}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--members", type=int, default=K_MEMBERS,
                    help="每个 (dataset, seed) 训练多少个独立成员, 默认 16")
    args = ap.parse_args()

    os.makedirs(EXT_DIR, exist_ok=True)
    failed = []
    for ds in args.datasets:
        for seed in args.seeds:
            try:
                run_one(ds, seed, k_members=args.members)
            except Exception:
                failed.append((ds, seed))
                print(f"!! 失败跳过: {ds} seed={seed}")
                traceback.print_exc()

    print("\n===== extension 完成 =====")
    if failed:
        print("失败项:", failed)
    print(f"结果: {CSV_OUT}")


if __name__ == "__main__":
    from src.run_log import run_log
    with run_log("ext_run_indep_mlp"):
        main()
