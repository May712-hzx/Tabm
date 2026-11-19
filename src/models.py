"""
模型定义。

- build_tabm: 官方 pip 包 tabm 的 TabM (权重共享的高效 MLP 集成), 输出 shape (B, k, d_out)
- build_mlp : 普通 MLP 基线, 主干宽度/层数与 TabM 对齐, 用于假设 H2 的公平对比
"""

import torch
import torch.nn as nn

from tabm import TabM


def build_tabm(n_num_features, cat_cardinalities, d_out, k,
               n_blocks=2, d_block=256, dropout=0.1):
    """构建 TabM。cat_cardinalities 为空时退化为纯数值输入。"""
    return TabM.make(
        n_num_features=int(n_num_features),
        cat_cardinalities=list(cat_cardinalities) if len(cat_cardinalities) > 0 else None,
        d_out=int(d_out),
        k=int(k),
        n_blocks=int(n_blocks),
        d_block=int(d_block),
        dropout=float(dropout),
    )


class MLP(nn.Module):
    """普通 MLP 基线: 类别特征 -> Embedding(16 维) 拼接数值特征 -> 若干 Linear+ReLU+Dropout 块。"""

    CAT_EMB_DIM = 16

    def __init__(self, n_num_features, cat_cardinalities, d_out,
                 hidden_dim=256, n_blocks=2, dropout=0.1):
        super().__init__()
        self.n_num = int(n_num_features)
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]

        if len(self.cat_cardinalities) > 0:
            self.cat_embs = nn.ModuleList([
                nn.Embedding(c, self.CAT_EMB_DIM) for c in self.cat_cardinalities
            ])
            cat_dim = len(self.cat_cardinalities) * self.CAT_EMB_DIM
        else:
            self.cat_embs = nn.ModuleList([])
            cat_dim = 0

        input_dim = self.n_num + cat_dim
        if input_dim <= 0:
            raise ValueError("输入特征维度为 0, 请检查 n_num_features / cat_cardinalities")

        layers = []
        cur = input_dim
        for _ in range(int(n_blocks)):
            layers.append(nn.Linear(cur, int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            cur = int(hidden_dim)
        layers.append(nn.Linear(cur, int(d_out)))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x_num, x_cat=None):
        parts = []
        if self.n_num > 0:
            parts.append(x_num)
        if x_cat is not None and len(self.cat_embs) > 0:
            emb_parts = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embs)]
            parts.append(torch.cat(emb_parts, dim=-1))
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.backbone(x)


def build_mlp(n_num_features, cat_cardinalities, d_out,
              hidden_dim=256, n_blocks=2, dropout=0.1):
    return MLP(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout=dropout,
    )
