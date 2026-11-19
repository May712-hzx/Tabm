"""通用工具: 随机种子 / 参数量 / 指标计算 / 结果落盘。"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def set_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _normalize_rows(prob):
    """把每行概率重新归一化到和为 1, 消除 float32 累加误差导致的 sklearn 警告。"""
    if prob.ndim != 2:
        return prob
    s = prob.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return prob / s


def calc_metrics(y_true, y_pred, task_type, n_classes=None):
    """
    分类: roc_auc(二分类用正类; 多类用 macro-OvR) / logloss / macro_f1
    回归: rmse / mae / r2
    """
    res = {}
    y_true = np.asarray(y_true).reshape(-1)

    if task_type == "binclass":
        prob = np.asarray(y_pred, dtype=np.float64)
        if prob.ndim == 1:
            prob = np.stack([1 - prob, prob], axis=1)
        prob = _normalize_rows(prob)
        res["roc_auc"] = float(roc_auc_score(y_true, prob[:, 1]))
        res["logloss"] = float(log_loss(y_true, prob, labels=[0, 1]))
        res["macro_f1"] = float(f1_score(y_true, prob.argmax(axis=1), average="macro"))

    elif task_type == "multiclass":
        prob = _normalize_rows(np.asarray(y_pred, dtype=np.float64))
        labels = list(range(n_classes)) if n_classes is not None else None
        # macro-OvR AUC: 每个类各算一条 one-vs-rest ROC, 再对各类 AUC 取宏平均。
        # 下采样后若某个类在测试集里完全缺失, sklearn 会抛 ValueError, 此时记 NaN 而不是中断实验。
        try:
            res["roc_auc"] = float(roc_auc_score(
                y_true, prob, multi_class="ovr", average="macro", labels=labels,
            ))
        except ValueError:
            # 某个类在测试集里完全没有样本时 sklearn 会报 ValueError, 记 NaN 而不是中断实验
            res["roc_auc"] = float("nan")
        res["logloss"] = float(log_loss(y_true, prob, labels=labels))
        res["macro_f1"] = float(f1_score(y_true, prob.argmax(axis=1), average="macro"))

    elif task_type == "regression":
        pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        res["rmse"] = float(np.sqrt(mean_squared_error(y_true, pred)))
        res["mae"] = float(mean_absolute_error(y_true, pred))
        res["r2"] = float(r2_score(y_true, pred))

    else:
        raise ValueError(f"未知 task_type: {task_type}")

    return res


def _to_native(v):
    """把 numpy 标量转成 python 原生类型, 否则写 csv 会出现 np.float64(...) 字面量。"""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ---------------------------------------------------------------- 实验记录表
LOG_HEADER = """# 实验记录表（自动生成）

> 由 `src/utils.py: save_result()` 在每次写入 `results/all_result.csv` 时**自动追加一行**，
> 不需要手工维护。字段对应指导书 §9.3：编号 / 时间 / 研究问题 / 唯一改动 / 命令 / 结果 / 判断。
> 去重规则：同一 (数据集, seed, 模型, k, 主指标值) 只记一次，重跑完全相同的结果不会重复。
> 用 `scripts/backfill_experiment_log.py` 批量补录的历史行，其「时间」列是回填时刻而非原始运行时刻。
> 「判断」列留给人填（或写报告时批注），其余列均为机器写入。

| # | 时间 | 数据集 | seed | 模型 | k | 研究问题 | 唯一改动 | 命令 | 主指标 | 值 | 参数量 | 训练(s) | 判断 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""

_QUESTION = {
    "tabm": "H1/H2：成员数 k 对性能与成本的影响",
    "tabm_crop_random": "H3：从 k=16 随机裁剪是否优于直接训练小 k",
    "tabm_crop_valbest": "H3：从 k=16 按验证集择优裁剪是否优于直接训练小 k",
    "mlp": "基线：同预算 MLP",
    "xgb": "基线：XGBoost",
    "histgb": "基线：HistGBM",
}

_COMMAND = {
    "tabm": "python scripts/run_exp.py",
    "mlp": "python scripts/run_exp.py",
    "xgb": "python scripts/run_exp.py",
    "histgb": "python scripts/run_exp.py",
    "tabm_crop_random": "python scripts/run_crop_ablation.py",
    "tabm_crop_valbest": "python scripts/run_crop_ablation.py",
}


def _main_metric(row):
    """挑出这一行的主指标 (分类看 macro_f1, 回归看 r2)。"""
    for name in ("macro_f1", "r2"):
        v = row.get(name)
        if v is not None and isinstance(v, float) and np.isfinite(v):
            return name, v
    return "—", float("nan")


def _k_str(k):
    if k is None:
        return "—"
    try:
        if isinstance(k, float) and not np.isfinite(k):
            return "—"
    except TypeError:
        pass
    return str(int(k))


def _fmt_num(v, spec="{:.0f}"):
    """把可能是 numpy / pandas 取值、也可能是 NaN 的数字格式化成字符串。"""
    try:
        f = float(v)
        if not np.isfinite(f):
            return "—"
        return spec.format(f)
    except (TypeError, ValueError):
        return "—"


def _append_experiment_log(row, csv_path):
    """
    自动把一行结果写进 results/experiments_log.md。

    - 首次调用时创建表头。
    - 用 results/experiments_log.keys 做去重索引, 避免重跑产生重复行。
    - 编码固定 UTF-8，写失败不影响主流程（csv 已经写好了）。
    """
    import datetime

    try:
        row = {k: _to_native(v) for k, v in row.items()}

        base = os.path.dirname(os.path.abspath(csv_path))
        md_path = os.path.join(base, "experiments_log.md")
        key_path = os.path.join(base, "experiments_log.keys")

        ds = row.get("dataset", "—")
        seed = row.get("seed", "—")
        kind = row.get("model_kind", "—")
        k_str = _k_str(row.get("k"))
        metric_name, metric_val = _main_metric(row)
        mv = f"{metric_val:.4f}" if np.isfinite(metric_val) else "nan"
        key = f"{ds}|{seed}|{kind}|{k_str}|{metric_name}={mv}"

        keys = set()
        if os.path.exists(key_path):
            with open(key_path, "r", encoding="utf-8") as f:
                keys = {ln.strip() for ln in f if ln.strip()}
        if key in keys:
            return

        is_new = not os.path.exists(md_path) or os.path.getsize(md_path) == 0
        with open(md_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write(LOG_HEADER)
            idx = len(keys) + 1
            question = _QUESTION.get(kind, "—")
            change = f"k={k_str}" if kind in ("tabm", "tabm_crop_random", "tabm_crop_valbest") else "无（基线）"
            cmd = _COMMAND.get(kind, "—")
            f.write(
                f"| {idx} | {datetime.datetime.now():%Y-%m-%d %H:%M} | {ds} | {seed} | {kind} | {k_str} | "
                f"{question} | {change} | `{cmd}` | {metric_name} | {mv} | "
                f"{_fmt_num(row.get('params'))} | {_fmt_num(row.get('train_time_s'), '{:.1f}')} | — |\n"
            )

        with open(key_path, "a", encoding="utf-8") as f:
            f.write(key + "\n")
    except Exception as e:                       # 日志失败绝不能影响实验结果落盘
        print(f"    [warn] 实验日志写入失败: {e}")


def save_result(row_dict, csv_path):
    """追加一行结果到 csv，并同步写进 results/experiments_log.md。"""
    row = {k: _to_native(v) for k, v in row_dict.items()}
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df_old = pd.read_csv(csv_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(csv_path, index=False)
    _append_experiment_log(row, csv_path)
