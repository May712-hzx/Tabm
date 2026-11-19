"""生成论文用的 TabM 结构示意图（图 1）。

只画框图，不涉及任何实验数据；数值/符号与报告 §3 一致。
输出：paper/fig1_arch.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_arch.png")

BOXES = [
    (0.02, "表格特征\n数值 + 类别", "#eef3fb"),
    (0.25, "k 个成员适配\nγ_r ⊙ x（rank-1 缩放）", "#fdeee0"),
    (0.47, "共享 MLP 主干\n2 × 256，ReLU", "#e8f5e9"),
    (0.69, "k 个预测头\n并行成员输出 z_r", "#e8f5e9"),
    (0.88, "平均集成\n分类: 概率均值\n回归: 数值均值", "#eef3fb"),
]

W, H = 0.15, 0.42
Y = 0.34


def main():
    fig, ax = plt.subplots(figsize=(11, 3.1), dpi=220)
    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x, text, color in BOXES:
        w = W if x < 0.85 else 0.17
        box = FancyBboxPatch(
            (x, Y), w, H,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2, edgecolor="#4a4a4a", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, Y + H / 2, text, ha="center", va="center",
                fontsize=10.5, color="#1f2d3d", linespacing=1.5)

    for i in range(len(BOXES) - 1):
        x0 = BOXES[i][0] + (W if BOXES[i][0] < 0.85 else 0.17)
        x1 = BOXES[i + 1][0]
        if x1 - x0 < 0.03:
            x1 = x0 + 0.03
        ax.add_patch(FancyArrowPatch(
            (x0 + 0.004, Y + H / 2), (x1 - 0.004, Y + H / 2),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#4a4a4a"))

    ax.text(0.5, 0.20,
            "k ∈ {1, 4, 8, 16}（32 为拓展）；成员共享绝大部分参数，"
            "增加 k 只线性增加极少量适配参数与计算量",
            ha="center", va="center", fontsize=9.5, color="#555555")
    ax.text(0.5, 0.09,
            "复现对象：共享主干 + 成员级缩放；训练时逐成员计算损失再平均，"
            "推理时先 softmax 再平均（分类）",
            ha="center", va="center", fontsize=9.5, color="#555555")

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
