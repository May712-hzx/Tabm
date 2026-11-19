"""
测试 2/3：模型输出形状与参数效率。

对齐指导书步骤 3「跑通官方 TabM 示例、打印参数量、核对 k 个成员的输出形状与平均方式」：
  - TabM 输出必须是 (B, k, d_out)，即 k 个成员各自的 logits；
  - MLP 基线输出是 (B, d_out)；
  - 参数量必须远小于 k× MLP —— 这是"参数高效集成"这一主张的直接断言。

跑法（项目根目录）：
    python -m unittest tests.test_models -v
"""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models import build_mlp, build_tabm
from src.utils import count_params

B, N_NUM, D_OUT = 32, 10, 5
CATS = [7, 3]          # 两个类别列，基数分别为 7 和 3


def _batch():
    x_num = torch.randn(B, N_NUM)
    x_cat = torch.tensor([[i % 7, i % 3] for i in range(B)], dtype=torch.long)
    return x_num, x_cat


class TestOutputShapes(unittest.TestCase):

    def test_tabm_output_is_b_k_d_out(self):
        for k in (1, 4, 8, 16):
            model = build_tabm(N_NUM, CATS, D_OUT, k=k, n_blocks=2, d_block=64, dropout=0.0)
            model.eval()
            x_num, x_cat = _batch()
            with torch.no_grad():
                out = model(x_num, x_cat)
            self.assertEqual(tuple(out.shape), (B, k, D_OUT),
                             f"k={k} 时 TabM 输出应为 (B,{k},{D_OUT})，实际 {tuple(out.shape)}")

    def test_mlp_output_is_b_d_out(self):
        model = build_mlp(N_NUM, CATS, D_OUT, hidden_dim=64, n_blocks=2, dropout=0.0)
        model.eval()
        x_num, x_cat = _batch()
        with torch.no_grad():
            out = model(x_num, x_cat)
        self.assertEqual(tuple(out.shape), (B, D_OUT))

    def test_tabm_members_are_not_identical(self):
        """同一批次里 k 个成员的输出应当各不相同（否则集成没有意义）。"""
        model = build_tabm(N_NUM, CATS, D_OUT, k=4, n_blocks=2, d_block=64, dropout=0.0)
        model.eval()
        x_num, x_cat = _batch()
        with torch.no_grad():
            out = model(x_num, x_cat)          # (B, 4, D_OUT)
        self.assertFalse(torch.allclose(out[:, 0, :], out[:, 1, :]),
                         "成员 0 与成员 1 输出完全相同，集成退化")


class TestParameterEfficiency(unittest.TestCase):
    """参数量：TabM 应远小于 k× 独立 MLP，这是论文的核心卖点。

    注意：比例依赖 d_block。d_block 很小时，k 个成员的 rank-1 缩放参数
    相对开销会显得很大；这里用项目真实配置 d_block=256（见 src/config.py）。
    """

    D_BLOCK = 256

    def _p_tabm(self, k):
        return count_params(build_tabm(N_NUM, CATS, D_OUT, k=k,
                                       n_blocks=2, d_block=self.D_BLOCK, dropout=0.0))

    def test_params_scale_slowly_with_k(self):
        p = {k: self._p_tabm(k) for k in (1, 4, 8, 16)}
        # 严格单调
        self.assertLess(p[1], p[4])
        self.assertLess(p[4], p[8])
        self.assertLess(p[8], p[16])
        # 真实配置下 k=16 参数量约为 k=1 的 1.3 倍（主实验实测 94932 -> 123453）
        self.assertLess(p[16] / p[1], 1.8,
                        f"k=16 参数量是 k=1 的 {p[16] / p[1]:.2f} 倍，超出预期上限")

    def test_tabm_uses_far_fewer_params_than_k_independent_mlps(self):
        p_tabm = self._p_tabm(16)
        p_mlp1 = count_params(build_mlp(N_NUM, CATS, D_OUT,
                                        hidden_dim=self.D_BLOCK, n_blocks=2, dropout=0.0))
        self.assertLess(p_tabm, 3 * p_mlp1,
                        f"k=16 TabM 参数 {p_tabm} 不应超过 3 倍单 MLP 参数 {p_mlp1}")
        self.assertLess(p_tabm, 16 * p_mlp1,
                        "k=16 TabM 参数必须小于 16 个独立 MLP 的总参数")


if __name__ == "__main__":
    unittest.main(verbosity=2)
