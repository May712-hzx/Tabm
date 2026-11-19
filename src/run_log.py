"""
统一运行日志。

和「实验记录表」的区别（这两个东西容易被混淆）：

  results/experiments_log.md
      ├─ 一行 = 一次「实验配置」的结果（数据集/seed/模型/k → 指标）
      ├─ 由 save_result() 在写 all_result.csv 时自动追加
      └─ 目的是横向比较，给人看结论

  results/logs/<脚本名>_<时间戳>.log   ← 本文件负责
      ├─ 一份 = 一次「脚本运行」的完整过程（命令、环境、stdout、耗时、退出状态）
      ├─ 由 run_log() 在脚本启动时创建，退出时补写结束状态
      └─ 目的是复现排错，给人和机器看过程

用法（在脚本末尾包一层）：

    if __name__ == "__main__":
        from src.run_log import run_log
        with run_log("run_exp"):
            main()

跑完之后：
  - results/logs/run_exp_20261119_001500.log  每次运行的完整 stdout
  - results/logs/index.md                     所有运行的一条索引（命令/开始/耗时/状态/日志路径）
"""

import contextlib
import datetime
import os
import platform
import sys
import time
import traceback

LOG_DIR = os.path.join("results", "logs")
INDEX_PATH = os.path.join(LOG_DIR, "index.md")

_INDEX_HEADER = """# 运行日志索引（自动生成）

> 每运行一次脚本追加一行。日志正文在同目录的 `<脚本名>_<时间戳>.log`。
> 与 `results/experiments_log.md`（一次实验配置一行）互补：这里是「一次运行」一行。

| # | 开始时间 | 脚本 | 命令 | 耗时(s) | 状态 | 日志文件 |
|---|---|---|---|---|---|---|
"""


class _Tee:
    """把输出同时写回原控制台和日志文件。"""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        try:
            self.fh.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        for obj in (self.stream, self.fh):
            try:
                obj.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    # 一些库会访问 fileno / encoding，回退到原流
    def fileno(self):
        return self.stream.fileno()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


def _safe_makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _packages():
    """记录关键库的版本，出问题时可归因到环境变化。"""
    out = {}
    for mod in ("torch", "numpy", "pandas", "sklearn", "xgboost"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "not installed"
    return out


def _append_index(name, path, t0, elapsed, status, argv):
    try:
        _safe_makedirs(LOG_DIR)
        n = 0
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                n = sum(1 for ln in f if ln.startswith("| ") and not ln.startswith("| # "))
        is_new = not os.path.exists(INDEX_PATH) or os.path.getsize(INDEX_PATH) == 0
        with open(INDEX_PATH, "a", encoding="utf-8") as f:
            if is_new:
                f.write(_INDEX_HEADER)
            cmd = "python " + (argv[0] if argv else name) + (" " + " ".join(argv[1:]) if argv and len(argv) > 1 else "")
            f.write(f"| {n + 1} | {t0:%Y-%m-%d %H:%M:%S} | {name} | `{cmd}` | "
                    f"{elapsed:.1f} | {status} | `{os.path.basename(path)}` |\n")
    except Exception:
        pass


@contextlib.contextmanager
def run_log(name, extra=None):
    """
    给一次脚本运行建立日志。退出时无论成功还是异常都会补写结束状态。

    参数：
        name  : 脚本名（用于文件名，例如 "run_exp"）
        extra : dict，附加到日志头部的键值（例如 {"配置": "adult, k=1/4/8/16"}）
    """
    started = datetime.datetime.now()
    argv = list(sys.argv)
    ts = started.strftime("%Y%m%d_%H%M%S")
    _safe_makedirs(LOG_DIR)
    path = os.path.abspath(os.path.join(LOG_DIR, f"{name}_{ts}.log"))

    try:
        fh = open(path, "w", encoding="utf-8")
    except Exception:
        # 连日志文件都建不了（比如只读目录）就退化成不记录，绝不阻断实验
        yield path
        return

    with fh:
        fh.write(f"# 运行日志: {name}\n")
        fh.write(f"开始时间: {started:%Y-%m-%d %H:%M:%S}\n")
        fh.write(f"工作目录: {os.getcwd()}\n")
        fh.write(f"命令: python {' '.join(argv)}\n")
        fh.write(f"Python: {sys.version.split()[0]} @ {platform.platform()}\n")
        fh.write("依赖版本: " + ", ".join(f"{k}={v}" for k, v in _packages().items()) + "\n")
        if extra:
            for k, v in dict(extra).items():
                fh.write(f"{k}: {v}\n")
        fh.write("=" * 78 + "\n\n")
        fh.flush()

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, fh)
        sys.stderr = _Tee(old_err, fh)
        status, elapsed = "OK", 0.0
        t0 = time.time()
        try:
            yield path
        except BaseException as e:
            status = f"FAILED({type(e).__name__})"
            fh.write("\n" + "-" * 78 + "\n" + traceback.format_exc())
            raise
        finally:
            elapsed = time.time() - t0
            sys.stdout, sys.stderr = old_out, old_err
            try:
                fh.write("\n" + "=" * 78 + "\n")
                fh.write(f"结束时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
                fh.write(f"耗时: {elapsed:.1f} s\n")
                fh.write(f"状态: {status}\n")
                fh.flush()
            except Exception:
                pass
            _append_index(name, path, started, elapsed, status, argv)
            try:
                print(f"[run_log] {status}  {elapsed:.1f}s  -> {path}")
            except Exception:
                pass
