"""
测试 1/3：数据划分与预处理。

覆盖指导书要求的三件事：
  1. train / val / test 严格不重叠，且比例为 60/20/20；
  2. 预处理只在训练集 fit（训练集被标准化为 mean≈0、std≈1）；
  3. 同一 seed 重复调用结果完全一致（可复现）；无类别列的数据集不会误走类别分支。

跑法（项目根目录）：
    python -m unittest tests.test_dataset -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import _split, get_dataset


class TestSplitOnSyntheticData(unittest.TestCase):
    """用合成数据验证划分本身：唯一 id 列可以直接证明三个集合不重叠。"""

    def setUp(self):
        n = 1000
        # id 列唯一，作为"这一行是谁"的证据
        self.X = pd.DataFrame({
            "id": np.arange(n, dtype=np.float64),
            "v": np.random.RandomState(0).randn(n),
        })
        # 三类标签，保证 stratify 可用
        self.y = np.arange(n) % 3

    def test_disjoint_and_ratio(self):
        X_tr, X_va, X_te, y_tr, y_va, y_te = _split(self.X, self.y, seed=0, task_type="multiclass")

        ids = [set(X_tr["id"]), set(X_va["id"]), set(X_te["id"])]
        # 两两交集为空 → 三个集合严格不重叠
        self.assertEqual(ids[0] & ids[1], set(), "train 与 val 有重叠样本")
        self.assertEqual(ids[0] & ids[2], set(), "train 与 test 有重叠样本")
        self.assertEqual(ids[1] & ids[2], set(), "val 与 test 有重叠样本")

        total = len(self.X)
        self.assertEqual(len(X_tr) + len(X_va) + len(X_te), total, "划分后样本总数应与原始一致")
        self.assertAlmostEqual(len(X_tr) / total, 0.60, delta=0.01)
        self.assertAlmostEqual(len(X_va) / total, 0.20, delta=0.01)
        self.assertAlmostEqual(len(X_te) / total, 0.20, delta=0.01)

    def test_labels_keep_class_ratio(self):
        """分层划分：每个集合都应包含全部类别。"""
        X_tr, X_va, X_te, y_tr, y_va, y_te = _split(self.X, self.y, seed=0, task_type="multiclass")
        for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
            self.assertEqual(sorted(np.unique(y)), [0, 1, 2], f"{name} 丢失了某些类别")

    def test_split_is_deterministic(self):
        a = _split(self.X, self.y, seed=7, task_type="multiclass")
        b = _split(self.X, self.y, seed=7, task_type="multiclass")
        self.assertTrue(np.array_equal(a[3], b[3]), "同 seed 两次划分的标签应完全一致")
        self.assertTrue(np.array_equal(a[0]["id"].to_numpy(), b[0]["id"].to_numpy()))


class TestRealDatasetPreprocessing(unittest.TestCase):
    """真实数据集：验证预处理只 fit 训练集，以及无类别列时的分支跳过。"""

    @classmethod
    def setUpClass(cls):
        # covertype 走本地缓存，不联网；2000 行足以验证流程
        cls.data = get_dataset("covertype", seed=0, max_samples=2000)

    def test_meta(self):
        X_tr, X_va, X_te, y_tr, y_va, y_te, n_num, cards, task, d_out = self.data
        self.assertEqual(task, "multiclass")
        self.assertEqual(d_out, 7)
        self.assertEqual(n_num, 54, "covertype 应有 54 个数值特征")
        self.assertEqual(cards, [], "covertype 没有类别列，cat_cardinalities 必须为空")
        for X in (X_tr, X_va, X_te):
            self.assertEqual(X.dtype, np.float32)
            self.assertEqual(X.shape[1], 54)
            self.assertTrue(np.isfinite(X).all(), "预处理后不应残留 NaN/inf")

    def test_train_is_standardized_by_train_only(self):
        """StandardScaler 只在训练集 fit → 训练集每列均值≈0、标准差≈1。"""
        X_tr = self.data[0]
        mean = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        self.assertLess(np.abs(mean).max(), 1e-4, f"训练集列均值应为 0，实际最大 {np.abs(mean).max()}")
        # 常数列（方差为 0）会被 StandardScaler 置 0，标准差会偏一下，所以只检查非平凡列
        nontrivial = std > 1e-6
        self.assertGreater(nontrivial.sum(), 40, "多数列应有非零方差")
        self.assertLess(np.abs(std[nontrivial] - 1.0).max(), 1e-3,
                        "训练集非平凡列的标准差应为 1")

    def test_reproducible_without_cache(self):
        """绕过缓存重新算一遍，结果应与缓存版逐位一致。"""
        a = self.data
        b = get_dataset("covertype", seed=0, max_samples=2000, use_cache=False)
        self.assertTrue(np.array_equal(a[0], b[0]), "训练集特征不可复现")
        self.assertTrue(np.array_equal(a[2], b[2]), "测试集特征不可复现")
        self.assertTrue(np.array_equal(a[5], b[5]), "测试集标签不可复现")


if __name__ == "__main__":
    unittest.main(verbosity=2)
