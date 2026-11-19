"""
补指导书步骤 6 要求的「推理时间 + 峰值内存」。

两个阶段，都不写 results/all_result.csv，不覆盖 ckpt：

A. 推理成本（默认，快）：对每个 (数据集, seed, 配置) 测
   - 测试集整轮推理耗时 total 与 per-sample latency
     （仅模型前向 + softmax/平均，batch=BATCH_SIZE，不含 DataLoader 开销）
   - model_mb = 权重的理论上限（params × 4 B）
   - infer_rss_delta_mb = 推理窗口内 RSS 增量（同进程里 RSS 不归还 OS，
     该列只作参考；真正的每配置峰值由阶段 B 用独立进程测）
   - k=16 的 TabM 直接加载 ckpt 权重；其余配置用同结构模型（推理路径与训练后一致）
   产物：results/cost/infer_cost.csv

B. 训练峰值内存（--train-peak）：**每个配置单独起一个子进程**测量，
   这样 RSS 从干净状态开始，峰值只归因于该配置（含优化器状态、梯度与激活）。
   RSS 在同一进程里单调累积、跨配置不可比，所以必须进程隔离。
   按与主实验相同的预算重训，只做 seed=0（峰值由结构 + batch 决定，与 seed 无关）。
   产物：results/cost/train_peak_seed0.csv

用法（项目根目录）：
    python scripts/eval_cost.py                # 只做推理成本（约 1.5 min）
    python scripts/eval_cost.py --train-peak   # 追加训练峰值（约 35~40 min）
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    BATCH_SIZE, CKPT_DIR, D_BLOCK, DATASETS, DROPOUT, K_LIST, N_BLOCKS,
    N_ESTIMATORS, SEEDS, max_samples_of,
)
from src.dataset import get_dataset
from src.models import build_mlp, build_tabm
from src.trainer import TabularTrainer, get_device
from src.utils import count_params, set_seed

COST_DIR = os.path.join(ROOT, "results", "cost")
INFER_CSV = os.path.join(COST_DIR, "infer_cost.csv")
TRAIN_CSV = os.path.join(COST_DIR, "train_peak_seed0.csv")

INFER_REPEAT = 5
INFER_WARMUP = 2
# 训练峰值只测 seed=0 的配置清单: (model_kind, k)，k=0 表示该模型无 k（避免 argparse 把 -1 当选项）
TRAIN_PEAK_CONFIGS = [("xgb", 0), ("histgb", 0),
                      ("tabm", 1), ("tabm", 4), ("tabm", 8), ("tabm", 16),
                      ("mlp", 0)]


def _rss_mb():
    import psutil
    return float(psutil.Process(os.getpid()).memory_info().rss) / 1024 ** 2


class PeakSampler:
    """后台线程采样进程 RSS，返回观测到的峰值 (MB)。"""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0.0
        self.base = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        gc.collect()
        self.base = _rss_mb()
        self.peak = self.base
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, _rss_mb())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    @property
    def delta(self):
        return self.peak - self.base


def _forward_once(model, X, n_num, task_type):
    """与 trainer.predict_* 同口径的一轮前向（仅计时用）。"""
    device = get_device()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH_SIZE):
            bx = Xt[i:i + BATCH_SIZE].to(device)
            x_num = bx[:, :n_num]
            x_cat = bx[:, n_num:].long() if bx.shape[1] > n_num else None
            pred = model(x_num, x_cat)
            if pred.dim() == 3:
                if task_type in ("binclass", "multiclass"):
                    pred = torch.softmax(pred, dim=-1)
                pred = pred.mean(dim=1)


def measure_infer(model, X, n_num, task_type):
    for _ in range(INFER_WARMUP):
        _forward_once(model, X, n_num, task_type)
    times = []
    with PeakSampler() as sp:
        for _ in range(INFER_REPEAT):
            t0 = time.perf_counter()
            _forward_once(model, X, n_num, task_type)
            times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    return med, med * 1000.0 / len(X), float(sp.delta)


def _append(row, path):
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path),
              index=False, encoding="utf-8-sig")


def _load(ds, seed):
    return get_dataset(ds, seed, max_samples=max_samples_of(ds))


def _build_tree(kind, task_type, d_out, seed):
    if kind == "xgb":
        if task_type == "regression":
            return xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=seed, verbosity=0)
        if task_type == "binclass":
            return xgb.XGBClassifier(n_estimators=N_ESTIMATORS, random_state=seed, verbosity=0)
        return xgb.XGBClassifier(n_estimators=N_ESTIMATORS, random_state=seed, verbosity=0,
                                 objective="multi:softprob", num_class=d_out)
    if task_type == "regression":
        return HistGradientBoostingRegressor(random_state=seed)
    return HistGradientBoostingClassifier(random_state=seed)


# ---------------------------------------------------------------- 阶段 A：推理
def run_infer_one(ds, seed):
    (X_train, X_val, X_test, y_train, y_val, y_test,
     n_num, cat_cards, task_type, d_out) = _load(ds, seed)
    rows = []

    # -------- 树模型: 重训(便宜)后只测 predict 耗时 --------
    for kind in ["xgb", "histgb"]:
        m = _build_tree(kind, task_type, d_out, seed)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        calls = m.predict_proba if hasattr(m, "predict_proba") else m.predict

        times = []
        gc.collect()
        calls(X_test)  # warmup
        for _ in range(INFER_REPEAT):
            t1 = time.perf_counter()
            calls(X_test)
            times.append(time.perf_counter() - t1)
        med = float(np.median(times))
        rows.append({
            "dataset": ds, "seed": seed, "model_kind": kind, "k": np.nan,
            "params": 0, "n_test": len(X_test),
            "infer_total_s": round(med, 4),
            "infer_ms_per_sample": round(med * 1000.0 / len(X_test), 4),
            "model_mb": np.nan,
            "infer_rss_delta_mb": np.nan,
            "weight_source": "retrain",
            "train_time_s_this_run": round(fit_s, 2),
        })
        del m
        gc.collect()

    # -------- 神经网络: 结构同主实验, k=16 用 ckpt 权重 --------
    for k in K_LIST:
        model = build_tabm(n_num, cat_cards, d_out, k=k, n_blocks=N_BLOCKS,
                           d_block=D_BLOCK, dropout=DROPOUT)
        src = "untrained_same_arch"
        if k == 16:
            ckpt = os.path.join(CKPT_DIR, f"tabm_{ds}_seed{seed}_k{16}.pt")
            if os.path.exists(ckpt):
                tr = TabularTrainer(model, task_type, device=get_device(), verbose=False)
                tr.load_checkpoint(ckpt)
                src = "ckpt"
        total_s, ms_ps, delta = measure_infer(model, X_test, n_num, task_type)
        rows.append({
            "dataset": ds, "seed": seed, "model_kind": "tabm", "k": k,
            "params": count_params(model), "n_test": len(X_test),
            "infer_total_s": round(total_s, 4),
            "infer_ms_per_sample": round(ms_ps, 4),
            "model_mb": round(count_params(model) * 4 / 1024 ** 2, 3),
            "infer_rss_delta_mb": round(delta, 1),
            "weight_source": src, "train_time_s_this_run": np.nan,
        })
        del model
        gc.collect()

    model = build_mlp(n_num, cat_cards, d_out, hidden_dim=D_BLOCK,
                      n_blocks=N_BLOCKS, dropout=DROPOUT)
    total_s, ms_ps, delta = measure_infer(model, X_test, n_num, task_type)
    rows.append({
        "dataset": ds, "seed": seed, "model_kind": "mlp", "k": np.nan,
        "params": count_params(model), "n_test": len(X_test),
        "infer_total_s": round(total_s, 4),
        "infer_ms_per_sample": round(ms_ps, 4),
        "model_mb": round(count_params(model) * 4 / 1024 ** 2, 3),
        "infer_rss_delta_mb": round(delta, 1),
        "weight_source": "untrained_same_arch", "train_time_s_this_run": np.nan,
    })
    del model
    gc.collect()

    for r in rows:
        _append(r, INFER_CSV)
        print(f"  {ds} s{seed} {r['model_kind']:<6} k={r['k']} "
              f"infer={r['infer_ms_per_sample']:.3f} ms/sample "
              f"model={r['model_mb']} MB")
    return rows


# ------------------------------------------------------------ 阶段 B：训练峰值
def run_train_one(ds, kind, k):
    """在一个全新进程里训练单个配置，返回该配置的训练峰值（MB）。"""
    (X_train, X_val, _, y_train, y_val, _, n_num, cat_cards, task_type, d_out) = _load(ds, 0)
    set_seed(0)

    if kind in ("xgb", "histgb"):
        m = _build_tree(kind, task_type, d_out, 0)
        t0 = time.perf_counter()
        with PeakSampler() as sp:
            m.fit(X_train, y_train)
        return {"dataset": ds, "seed": 0, "model_kind": kind, "k": np.nan,
                "params": 0, "n_train": len(X_train),
                "train_time_s": round(time.perf_counter() - t0, 2),
                "peak_rss_mb": round(float(sp.peak), 1)}

    if kind == "tabm":
        model = build_tabm(n_num, cat_cards, d_out, k=k, n_blocks=N_BLOCKS,
                           d_block=D_BLOCK, dropout=DROPOUT)
    else:
        model = build_mlp(n_num, cat_cards, d_out, hidden_dim=D_BLOCK,
                          n_blocks=N_BLOCKS, dropout=DROPOUT)
    tr = TabularTrainer(model, task_type, device=get_device(), verbose=False)
    t0 = time.perf_counter()
    with PeakSampler() as sp:
        if kind == "tabm":
            tr.train_tabm(X_train, y_train, X_val, y_val, n_num)
        else:
            tr.train_mlp(X_train, y_train, X_val, y_val, n_num)
    return {"dataset": ds, "seed": 0, "model_kind": kind,
            "k": (k if kind == "tabm" else np.nan),
            "params": count_params(model), "n_train": len(X_train),
            "train_time_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(float(sp.peak), 1)}


def run_train_peak_all():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    rows = []
    for ds in DATASETS:
        for kind, k in TRAIN_PEAK_CONFIGS:
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--train-one", ds, kind, str(k)]
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
            line = None
            for ln in (p.stdout or "").splitlines():
                if ln.startswith("PEAK_JSON:"):
                    line = ln[len("PEAK_JSON:"):]
            if line is None:
                print(f"  [失败] {ds} {kind} k={k}: {(p.stderr or '')[-300:]}")
                continue
            row = json.loads(line)
            rows.append(row)
            _append(row, TRAIN_CSV)
            print(f"  {ds} {kind:<6} k={row['k']} train={row['train_time_s']}s "
                  f"peak={row['peak_rss_mb']:.0f} MB")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-peak", action="store_true",
                    help="额外做训练峰值内存测量（子进程隔离，seed=0，较慢）")
    ap.add_argument("--train-one", nargs=3, metavar=("DS", "KIND", "K"),
                    help="内部使用：单独训练一个配置并打印 PEAK_JSON")
    args = ap.parse_args()
    os.makedirs(COST_DIR, exist_ok=True)

    if args.train_one:
        ds, kind, kk = args.train_one
        row = run_train_one(ds, kind, int(kk))
        print("PEAK_JSON:" + json.dumps(row, ensure_ascii=True))
        return

    # 阶段 A 已在本次跑过（infer_cost.csv 存在）时不再重复追加
    if not args.train_peak or not os.path.exists(INFER_CSV):
        print("=== 阶段 A: 推理时间（batch=%d，5 次取中位）===" % BATCH_SIZE)
        for ds in DATASETS:
            for s in SEEDS:
                print(f"[infer] {ds} seed={s}")
                run_infer_one(ds, s)
    else:
        print("=== 阶段 A 已有结果，跳过（删除 results/cost/infer_cost.csv 可重跑）===")

    if args.train_peak:
        print("=== 阶段 B: 训练峰值 RSS（每配置独立进程，seed=0）===")
        run_train_peak_all()

    print(f"推理成本 -> {INFER_CSV}")
    if args.train_peak:
        print(f"训练峰值 -> {TRAIN_CSV}")


if __name__ == "__main__":
    if "--train-one" in sys.argv:
        main()                       # 子进程模式：不写 run_log，避免 21 份日志
    else:
        from src.run_log import run_log
        with run_log("eval_cost"):
            main()
