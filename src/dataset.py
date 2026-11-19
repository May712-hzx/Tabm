"""
数据集加载与预处理。

支持: "adult"(二分类) / "covertype"(多分类) / "california"(回归)

设计要点:
1. 原始数据 + 按 seed 的预处理结果都在内存里缓存, 同一进程内重复调用不会重复下载/重复划分。
2. 类别特征为空时(california / covertype)自动跳过类别分支, 不会再给 SimpleImputer 传空矩阵。
3. 类别特征一律先转 str, 缺失值填 "__NA__"(作为一个独立类别), 因此不需要 ObjectImputer。
4. 测试集出现训练集没见过的类别(OOV)时, 替换成【训练集该列众数】, 保证编码值恒在 0..card-1,
   这样 TabM 内部 one-hot 不会越界, MLP 的 Embedding 也不会越界。
5. 所有预处理器只在训练集 fit, 严格避免数据泄露。
"""

import os

import numpy as np
import pandas as pd
import openml
import sklearn.datasets

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

NA_TOKEN = "__NA__"

_RAW_CACHE = {}
_PREP_CACHE = {}


def _load_california():
    """
    加载 California Housing。

    优先级:
      1. sklearn 本地缓存(存在才用, 避免 fetch 触网被 403 后反复重试)
      2. OpenML id=44977(california_housing), 用原始字段重建出与 sklearn 完全一致的 8 个特征
      3. 都失败则抛出明确错误

    重建口径与 sklearn.datasets.fetch_california_housing 一致:
      MedInc=medianIncome, HouseAge=housingMedianAge,
      AveRooms=totalRooms/households, AveBedrms=totalBedrooms/households,
      Population=population, AveOccup=population/households,
      Latitude, Longitude; target=medianHouseValue/100000
    """
    cache_pkz = os.path.join(os.path.expanduser("~"), "scikit_learn_data",
                             "california_housing", "cal_housing.pkz")
    if os.path.exists(cache_pkz):
        data = sklearn.datasets.fetch_california_housing()
        X = pd.DataFrame(data.data, columns=list(data.feature_names))
        return X, data.target.astype(np.float32)

    ds = openml.datasets.get_dataset(44977, download_data=True)
    raw, target, _, _ = ds.get_data(target=ds.default_target_attribute)
    raw = raw.reset_index(drop=True)

    households = raw["households"].replace(0, np.nan)
    X = pd.DataFrame({
        "MedInc": raw["medianIncome"],
        "HouseAge": raw["housingMedianAge"],
        "AveRooms": raw["totalRooms"] / households,
        "AveBedrms": raw["totalBedrooms"] / households,
        "Population": raw["population"],
        "AveOccup": raw["population"] / households,
        "Latitude": raw["latitude"],
        "Longitude": raw["longitude"],
    }).replace([np.inf, -np.inf], np.nan).astype(np.float64)

    y_arr = (pd.Series(target).astype(np.float64) / 100000.0).astype(np.float32).to_numpy()
    return X, y_arr


def _load_raw(dataset_name, max_samples=None, sample_seed=0):
    """读取原始数据, 返回 (X: DataFrame, y: ndarray, task_type, d_out, 原始类别数)。"""
    key = (dataset_name, max_samples)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]

    if dataset_name == "adult":
        ds = openml.datasets.get_dataset(1590, download_data=True)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        X = X.reset_index(drop=True)
        # 标签可能是 "<=50K" / ">50K" 或带结尾点号的变体, 用前缀判断最稳
        labels = pd.Series(y).astype(str).str.strip().str.replace(".", "", regex=False)
        y_arr = (labels == ">50K").astype(np.int64).to_numpy()
        task_type, d_out = "binclass", 2

    elif dataset_name == "covertype":
        data = sklearn.datasets.fetch_covtype()
        X = pd.DataFrame(data.data, columns=[f"f{i}" for i in range(data.data.shape[1])])
        y_arr = (data.target - 1).astype(np.int64)
        task_type, d_out = "multiclass", 7

    elif dataset_name == "california":
        X, y_arr = _load_california()
        task_type, d_out = "regression", 1

    else:
        raise ValueError(f"不支持的数据集: {dataset_name} (可选 adult / covertype / california)")

    X = X.reset_index(drop=True)

    # 大数据集可选下采样(控制训练时间预算)。分层抽样尽量保留类别比例。
    if max_samples is not None and len(X) > max_samples:
        strat = None if task_type == "regression" else pd.Series(y_arr)
        try:
            X_sub, _, y_sub, _ = train_test_split(
                X, y_arr,
                train_size=max_samples,
                random_state=sample_seed,
                stratify=strat,
            )
        except ValueError:
            rng = np.random.default_rng(sample_seed)
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X_sub, y_sub = X.iloc[idx], y_arr[idx]
        X_sub = X_sub.reset_index(drop=True)
        y_arr = np.asarray(y_sub)
        X = X_sub

    _RAW_CACHE[key] = (X, y_arr, task_type, d_out)
    return _RAW_CACHE[key]


def _split(X, y, seed, task_type):
    """60% / 20% / 20% 划分, 分类任务尽量分层。"""
    strat_full = pd.Series(y) if task_type != "regression" else None
    try:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=strat_full
        )
    except ValueError:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

    strat_tmp = pd.Series(y_tmp) if task_type != "regression" else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.25, random_state=seed, stratify=strat_tmp
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.25, random_state=seed
        )

    return (
        X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True),
        np.asarray(y_train), np.asarray(y_val), np.asarray(y_test),
    )


def get_dataset(dataset_name, seed=42, max_samples=None, use_cache=True):
    """
    返回:
        X_train, X_val, X_test, y_train, y_val, y_test,
        n_num_features, cat_cardinalities, task_type, d_out
    其中 X_* 为 float32 ndarray, 列顺序 = [数值特征..., 类别特征(编码后)...]
    """
    key = (dataset_name, seed, max_samples)
    if use_cache and key in _PREP_CACHE:
        return _PREP_CACHE[key]

    X_raw, y_all, task_type, d_out = _load_raw(dataset_name, max_samples, sample_seed=0)
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X_raw, y_all, seed, task_type)

    # 数值列 / 类别列 (显式判断 dtype, 不依赖 select_dtypes 的隐式规则)
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    n_num_features = len(num_cols)

    # ---------- 数值分支 ----------
    if n_num_features > 0:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        num_pipe.fit(X_train[num_cols].astype(np.float64))
    else:
        num_pipe = None

    # ---------- 类别分支(仅当真的有类别列才构建) ----------
    encoder = None
    train_modes = None
    cat_cardinalities = []

    if len(cat_cols) > 0:
        cat_train = (
            X_train[cat_cols]
            .astype(object)
            .where(pd.notna(X_train[cat_cols]), NA_TOKEN)
            .astype(str)
        )
        # 训练集每列众数(用于填充测试集 OOV)
        train_modes = {c: cat_train[c].value_counts().idxmax() for c in cat_cols}

        encoder = OrdinalEncoder(dtype=np.int64)
        encoder.fit(cat_train.to_numpy())
        cat_cardinalities = [int(len(cats)) for cats in encoder.categories_]

    def _encode_num(df):
        if num_pipe is None:
            return np.zeros((len(df), 0), dtype=np.float32)
        return num_pipe.transform(df[num_cols].astype(np.float64)).astype(np.float32)

    def _encode_cat(df):
        if encoder is None:
            return np.zeros((len(df), 0), dtype=np.float32)
        cat = (
            df[cat_cols]
            .astype(object)
            .where(pd.notna(df[cat_cols]), NA_TOKEN)
            .astype(str)
        )
        arr = cat.to_numpy()
        # OOV -> 训练集该列众数, 保证编码值恒在 0..card-1
        for i, c in enumerate(cat_cols):
            known = set(encoder.categories_[i])
            col = arr[:, i]
            mask = np.fromiter((v not in known for v in col), dtype=bool, count=len(col))
            if mask.any():
                col[mask] = train_modes[c]
                arr[:, i] = col
        return encoder.transform(arr).astype(np.float32)

    def _prep(df):
        parts = [_encode_num(df), _encode_cat(df)]
        X = np.concatenate(parts, axis=1).astype(np.float32)
        return np.ascontiguousarray(X)

    X_train_p = _prep(X_train)
    X_val_p = _prep(X_val)
    X_test_p = _prep(X_test)

    # 安全检查: 类别编码必须严格小于 cardinality, 否则 TabM 的 one-hot 会抛
    # RuntimeError: Class values must be smaller than num_classes.
    for i, card in enumerate(cat_cardinalities):
        col = n_num_features + i
        mx = max(X_train_p[:, col].max(), X_val_p[:, col].max(), X_test_p[:, col].max())
        if mx >= card:
            raise ValueError(
                f"类别编码越界: 第 {i} 个类别列最大值 {mx} >= cardinality {card}"
            )

    if task_type in ("binclass", "multiclass"):
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)
        y_test = y_test.astype(np.int64)
    else:
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)

    result = (X_train_p, X_val_p, X_test_p, y_train, y_val, y_test,
              n_num_features, cat_cardinalities, task_type, d_out)

    if use_cache:
        _PREP_CACHE[key] = result
    return result


def describe_dataset(dataset_name, seed=42, max_samples=None):
    """打印数据集规模, 方便写实验记录/报告。"""
    X_train, X_val, X_test, y_train, _, _, n_num, cat_cards, task_type, d_out = get_dataset(
        dataset_name, seed, max_samples
    )
    print(f"[{dataset_name}] task={task_type} d_out={d_out}")
    print(f"  train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"  n_num_features={n_num}, cat_cardinalities={cat_cards}")
    return {"dataset": dataset_name, "task_type": task_type, "d_out": d_out,
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            "n_num_features": n_num, "cat_cardinalities": cat_cards}
