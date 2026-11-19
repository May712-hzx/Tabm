"""
PyTorch 训练 / 推理 / 分析工具。

关键点(对应课程指导书):
1. TabM 的损失必须是【每个成员单独算 loss, 再对 k 个 loss 求平均】,
   绝对不能先对 k 个预测求平均再算 loss。
2. 推理时: 分类 -> 先 softmax 得到概率, 再对 k 个概率求平均;
   回归 -> 直接对 k 个预测值求平均。
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularTrainer:
    def __init__(self, model, task_type, device=None, lr=0.002, weight_decay=3e-4,
                 batch_size=256, patience=10, max_epoch=100, verbose=True):
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.task_type = task_type
        self.batch_size = batch_size
        self.patience = patience
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if task_type in ("binclass", "multiclass"):
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"未知 task_type: {task_type}")

        self.best_model = None
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    # ------------------------------------------------------------------ data
    def make_dataloader(self, X, y, shuffle=True):
        X = np.ascontiguousarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).float()
        if self.task_type in ("binclass", "multiclass"):
            y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)).long()
        else:
            y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).float().reshape(-1, 1)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=shuffle)

    @staticmethod
    def split_num_cat(batch_x, n_num_features):
        """dataset.py 输出的列顺序是 [数值..., 类别...]"""
        x_num = batch_x[:, :n_num_features]
        if batch_x.shape[-1] > n_num_features:
            x_cat = batch_x[:, n_num_features:].long()
        else:
            x_cat = None
        return x_num, x_cat

    # ------------------------------------------------------------------ train
    def _run_epochs(self, X_train, y_train, X_val, y_val, n_num_features, tabm_mode):
        train_dl = self.make_dataloader(X_train, y_train, shuffle=True)
        val_dl = self.make_dataloader(X_val, y_val, shuffle=False)

        for epoch in range(self.max_epoch):
            self.model.train()
            it = tqdm(train_dl, leave=False, desc=f"epoch {epoch+1}") if self.verbose else train_dl
            for bx, by in it:
                bx = bx.to(self.device)
                by = by.to(self.device)
                x_num, x_cat = self.split_num_cat(bx, n_num_features)

                self.optimizer.zero_grad()
                pred = self.model(x_num, x_cat)
                if tabm_mode:
                    # (B, k, d_out) -> k 个独立 loss -> 平均
                    k = pred.shape[1]
                    loss = torch.stack(
                        [self.criterion(pred[:, r, :], by) for r in range(k)]
                    ).mean()
                else:
                    loss = self.criterion(pred, by)
                loss.backward()
                self.optimizer.step()

            val_loss = self._eval(val_dl, n_num_features, tabm_mode)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    if self.verbose:
                        print(f"  早停于 epoch {epoch+1}, best epoch={self.best_epoch}, "
                              f"best val_loss={self.best_val_loss:.4f}")
                    break

        if self.best_model is None:
            raise RuntimeError("训练过程中验证损失始终为 NaN, 请检查输入数据 / 学习率")
        self.model.load_state_dict(self.best_model)

    def _eval(self, dataloader, n_num_features, tabm_mode):
        self.model.eval()
        total = 0.0
        cnt = 0
        with torch.no_grad():
            for bx, by in dataloader:
                bx, by = bx.to(self.device), by.to(self.device)
                x_num, x_cat = self.split_num_cat(bx, n_num_features)
                pred = self.model(x_num, x_cat)
                if tabm_mode:
                    k = pred.shape[1]
                    loss = torch.stack(
                        [self.criterion(pred[:, r, :], by) for r in range(k)]
                    ).mean()
                else:
                    loss = self.criterion(pred, by)
                total += float(loss.item()) * bx.shape[0]
                cnt += bx.shape[0]
        return total / max(cnt, 1)

    def train_tabm(self, X_train, y_train, X_val, y_val, n_num_features):
        self._run_epochs(X_train, y_train, X_val, y_val, n_num_features, tabm_mode=True)

    def train_mlp(self, X_train, y_train, X_val, y_val, n_num_features):
        self._run_epochs(X_train, y_train, X_val, y_val, n_num_features, tabm_mode=False)

    # ---------------------------------------------------------------- predict
    def predict_tabm(self, X, n_num_features):
        """分类: 平均概率; 回归: 平均预测值。返回 numpy。"""
        dl = self.make_dataloader(X, np.zeros(len(X)), shuffle=False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for bx, _ in dl:
                bx = bx.to(self.device)
                x_num, x_cat = self.split_num_cat(bx, n_num_features)
                pred = self.model(x_num, x_cat)          # (B, k, d_out)
                if self.task_type in ("binclass", "multiclass"):
                    pred = torch.softmax(pred, dim=-1).mean(dim=1)
                else:
                    pred = pred.mean(dim=1)
                outs.append(pred.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict_mlp(self, X, n_num_features):
        dl = self.make_dataloader(X, np.zeros(len(X)), shuffle=False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for bx, _ in dl:
                bx = bx.to(self.device)
                x_num, x_cat = self.split_num_cat(bx, n_num_features)
                pred = self.model(x_num, x_cat)
                if self.task_type in ("binclass", "multiclass"):
                    pred = torch.softmax(pred, dim=-1)
                outs.append(pred.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def get_raw_member_output(self, X, n_num_features):
        """返回每个成员的原始输出 (N, k, d_out), 不做平均。用于 H3 裁剪 / 多样性分析。"""
        dl = self.make_dataloader(X, np.zeros(len(X)), shuffle=False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for bx, _ in dl:
                bx = bx.to(self.device)
                x_num, x_cat = self.split_num_cat(bx, n_num_features)
                outs.append(self.model(x_num, x_cat).cpu().numpy())
        return np.concatenate(outs, axis=0)

    # ------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path, n_num_features):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "task_type": self.task_type,
            "n_num_features": int(n_num_features),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.task_type = ckpt["task_type"]
        return ckpt.get("n_num_features", None)

    # ---------------------------------------------------------------- metrics
    @staticmethod
    def stable_softmax(x):
        x = np.asarray(x, dtype=np.float64)
        mx = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - mx)
        return e / np.sum(e, axis=-1, keepdims=True)

    def predict_by_crop_members(self, raw_output, member_idx_list):
        """按给定成员下标子集做平均预测。"""
        sub = raw_output[:, list(member_idx_list), :]
        if self.task_type in ("binclass", "multiclass"):
            return np.mean(self.stable_softmax(sub), axis=1)
        return np.mean(sub, axis=1)

    @staticmethod
    def calc_ece(probs, y_true, task_type, n_bins=10):
        """期望校准误差 (Expected Calibration Error)。仅分类任务有效, 回归返回 nan。"""
        if task_type not in ("binclass", "multiclass"):
            return float("nan")
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] < 2:
            return float("nan")
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        y_true = np.asarray(y_true).reshape(-1)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1])
            if not mask.any():
                continue
            bin_conf = conf[mask].mean()
            bin_acc = (pred[mask] == y_true[mask]).mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)
        return float(ece / len(y_true))

    @staticmethod
    def calc_member_correlation(raw_output, task_type):
        """
        成员之间的预测相关性 (H2 多样性指标)。
        - 分类: 用各成员的 argmax 预测类别做皮尔逊相关
        - 回归: 用各成员的连续预测值做皮尔逊相关
        返回值越大 -> 成员越趋同 -> 多样性越低。k<=1 时返回 nan。
        """
        raw = np.asarray(raw_output, dtype=np.float64)
        k = raw.shape[1]
        if k <= 1:
            return float("nan")

        if task_type == "regression":
            vec = raw[..., 0]                       # (N, k)
        else:
            vec = raw.argmax(axis=-1).astype(np.float64)   # (N, k)

        corrs = []
        for i in range(k):
            for j in range(i + 1, k):
                vi, vj = vec[:, i], vec[:, j]
                if vi.std() < 1e-12 or vj.std() < 1e-12:
                    continue
                corrs.append(np.corrcoef(vi, vj)[0, 1])
        if len(corrs) == 0:
            return float("nan")
        return float(np.nanmean(corrs))
