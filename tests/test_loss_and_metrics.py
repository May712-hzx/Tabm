"""
测试 3/3：损失计算口径与指标正确性。

这是本项目最容易写错、也最致命的地方（指导书明确点名的红线）：
  - 训练时损失必须是「k 个成员各自算 loss 再平均」，
    绝不能「先把 k 个预测平均成一个，再算 loss」；
  - 推理时分类必须「先 softmax 再对概率平均」，
    不能「先对 logits 平均再 softmax」；
  - 指标本身要与 sklearn 完全一致。

跑法（项目根目录）：
    python -m unittest tests.test_loss_and_metrics -v
"""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models import build_tabm
from src.trainer import TabularTrainer
from src.utils import calc_metrics

B, N_NUM, D_OUT, K = 64, 6, 4, 4


def _make_model(k=K):
    torch.manual_seed(0)
    model = build_tabm(N_NUM, [], D_OUT, k=k, n_blocks=2, d_block=32, dropout=0.0)
    model.train()
    return model


def _grads(model):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


class TestTabmLossContract(unittest.TestCase):
    """逐成员 loss 平均 vs 先平均预测再算 loss：数值与梯度都必须不同。"""

    def setUp(self):
        torch.manual_seed(1)
        self.x = torch.randn(B, N_NUM)
        self.y = torch.randint(0, D_OUT, (B,))
        self.ce = nn.CrossEntropyLoss()

    def test_values_differ(self):
        model = _make_model()
        pred = model(self.x, None)                       # (B, K, D_OUT)
        self.assertEqual(tuple(pred.shape), (B, K, D_OUT))

        loss_per_member = torch.stack(
            [self.ce(pred[:, r, :], self.y) for r in range(K)]
        ).mean()
        loss_avg_pred = self.ce(pred.mean(dim=1), self.y)

        # 两种口径的数值必然不同（除非 k 个成员输出完全一致）
        self.assertFalse(torch.allclose(loss_per_member, loss_avg_pred),
                         "逐成员 loss 平均与平均预测的 loss 竟然相等，说明成员退化了")
        self.assertGreater(abs(float(loss_per_member.detach()) - float(loss_avg_pred.detach())), 1e-4)

    def test_gradients_differ(self):
        """两条 loss 对同一组参数的梯度不同 → 训练出来的模型不是同一个。"""
        model = _make_model()
        pred = model(self.x, None)

        model.zero_grad()
        torch.stack([self.ce(pred[:, r, :], self.y) for r in range(K)]).mean().backward()
        g_member = _grads(model)

        model.zero_grad()
        self.ce(model(self.x, None).mean(dim=1), self.y).backward()
        g_avg = _grads(model)

        diffs = []
        for name in g_member:
            if g_member[name] is None or g_avg[name] is None:
                continue
            diffs.append(float((g_member[name] - g_avg[name]).abs().max()))
        self.assertTrue(diffs, "没有采集到任何梯度")
        self.assertGreater(max(diffs), 1e-6,
                           "两种损失口径的梯度完全相同，说明实现退化成了同一条路径")


class TestInferenceAggregation(unittest.TestCase):
    """推理聚合：softmax 再平均 ≠ 平均再 softmax。"""

    def test_softmax_then_average_differs_from_average_then_softmax(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(16, K, D_OUT)) * 3.0        # 故意放大 logit 差异

        # 正确口径：先 softmax，再对 k 个概率求平均
        prob_correct = TabularTrainer.stable_softmax(raw).mean(axis=1)
        # 错误口径：先对 logits 平均，再 softmax
        prob_wrong = TabularTrainer.stable_softmax(raw.mean(axis=1))

        self.assertFalse(np.allclose(prob_correct, prob_wrong),
                         "两种聚合方式结果相同，测试本身失效")
        self.assertGreater(np.abs(prob_correct - prob_wrong).max(), 1e-6)
        # 正确口径的输出仍是合法概率分布
        np.testing.assert_allclose(prob_correct.sum(axis=1), 1.0, rtol=1e-6, atol=1e-9)

    def test_stable_softmax_handles_large_logits(self):
        """logits 很大时不能溢出成 nan。"""
        raw = np.array([[1e4, 1e4 - 1.0, 0.0]])
        p = TabularTrainer.stable_softmax(raw)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)


class TestMetrics(unittest.TestCase):

    def test_binclass_matches_sklearn(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1])
        prob = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9],
            [0.4, 0.6], [0.95, 0.05], [0.3, 0.7],
        ])
        m = calc_metrics(y, prob, "binclass")
        self.assertAlmostEqual(m["roc_auc"], roc_auc_score(y, prob[:, 1]), places=10)
        self.assertAlmostEqual(m["macro_f1"], f1_score(y, prob.argmax(1), average="macro"), places=10)
        self.assertGreaterEqual(m["logloss"], 0.0)

    def test_multiclass_macro_ovr_auc_matches_sklearn(self):
        rng = np.random.default_rng(3)
        n, d_out = 60, 5
        y = np.tile(np.arange(d_out), n // d_out)          # 每个类都有样本
        prob = rng.random((n, d_out))
        prob /= prob.sum(axis=1, keepdims=True)            # 已是合法概率

        m = calc_metrics(y, prob, "multiclass", n_classes=d_out)

        ref_auc = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(d_out)))
        self.assertAlmostEqual(m["roc_auc"], ref_auc, places=10,
                               msg="多类 macro-OvR AUC 与 sklearn 不一致")
        self.assertAlmostEqual(m["macro_f1"],
                               f1_score(y, prob.argmax(1), average="macro"), places=10)

    def test_multiclass_auc_is_nan_when_a_class_is_missing(self):
        """下采样后某类在测试集完全缺失时，应返回 NaN 而不是让实验崩溃。"""
        n, d_out = 20, 5
        y = np.zeros(n, dtype=int)                          # 只有类 0
        prob = np.full((n, d_out), 1.0 / d_out)
        m = calc_metrics(y, prob, "multiclass", n_classes=d_out)
        self.assertTrue(np.isnan(m["roc_auc"]))
        self.assertFalse(np.isnan(m["macro_f1"]))

    def test_regression_metrics(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([[1.0], [2.0], [3.0]])
        m = calc_metrics(y, pred, "regression")
        self.assertAlmostEqual(m["rmse"], 0.0, places=10)
        self.assertAlmostEqual(m["mae"], 0.0, places=10)
        self.assertAlmostEqual(m["r2"], 1.0, places=10)


class TestCalibration(unittest.TestCase):

    def test_perfect_prediction_has_zero_ece(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        ece = TabularTrainer.calc_ece(prob, y, "binclass")
        self.assertAlmostEqual(ece, 0.0, places=9)

    def test_worst_case_calibration_is_positive(self):
        """全押同一类但一半猜错 → ECE 应接近 1。"""
        y = np.array([0, 1, 0, 1])
        prob = np.array([[1.0, 0.0]] * 4)
        ece = TabularTrainer.calc_ece(prob, y, "binclass")
        self.assertGreater(ece, 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
