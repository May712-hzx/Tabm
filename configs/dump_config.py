"""从 src/config.py 导出当前实验配置快照到 configs/。

用途：指导书要求仓库有 configs/，且每个结果能指向唯一配置。
每次运行前导出一次，配置即被固定在 commit 里。

用法（项目根目录）：
    python configs/dump_config.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.config as cfg

FIELDS = [
    "DATASETS", "SEEDS", "K_LIST", "H3_SOURCE_K", "CROP_SIZES", "MAX_SAMPLES",
    "N_BLOCKS", "D_BLOCK", "DROPOUT", "MAX_EPOCH", "PATIENCE",
    "LR", "WEIGHT_DECAY", "BATCH_SIZE", "N_ESTIMATORS",
]


def main():
    snapshot = {f: getattr(cfg, f) for f in FIELDS}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"配置快照 -> {out}")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
