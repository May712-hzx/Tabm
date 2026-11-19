"""
集中存放实验配置。run_exp.py 和 run_crop_ablation.py 共用同一份,
保证两次运行的数据划分完全一致(否则 H3 裁剪结果不可比)。
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "ckpt")
PLOTS_DIR = os.path.join(ROOT, "plots")
CSV_OUT = os.path.join(RESULTS_DIR, "all_result.csv")

DATASETS = ["adult", "california", "covertype"]
SEEDS = [0, 1, 2, 3, 4]
K_LIST = [1, 4, 8, 16]

# H3 裁剪实验从哪个 k 的成员池里裁剪(必须出现在 K_LIST 里, 否则会保存不到权重)
H3_SOURCE_K = 16

# 大数据集下采样上限(控制训练时间预算)。要求全量请设成 None。
MAX_SAMPLES = {
    "adult": None,
    "california": None,
    "covertype": 50000,
}

# 神经网络模型的统一训练预算(公平对比的前提)
N_BLOCKS = 2
D_BLOCK = 256
DROPOUT = 0.1
MAX_EPOCH = 100
PATIENCE = 10
LR = 0.002
WEIGHT_DECAY = 3e-4
BATCH_SIZE = 256

# 树模型预算
N_ESTIMATORS = 100

# H3 裁剪实验要评估的成员子集大小
CROP_SIZES = [4, 8]


def max_samples_of(dataset_name):
    return MAX_SAMPLES.get(dataset_name)
