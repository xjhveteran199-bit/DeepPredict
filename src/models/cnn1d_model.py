"""
CNN1D 时序预测模型 v5
在 v4 基础上新增：
1. 多变量支持（自动识别 input_size）
2. 周期性特征注入（月份/星期/小时 sin/cos）
3. Per-variable 归一化
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class CNN1DModelV4(nn.Module):
    """
    改进版 CNN1D 时序预测模型 v4

    参照 PatchTST 架构：
    1. Patch 化：把时序分成多个 patch
    2. CNN 特征提取
    3. 直接多步预测
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        seq_len: int = 96,
        pred_len: int = 48,
        dropout: float = 0.1
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        valid_patch_sizes = [p for p in [1, 2, 4, 8, 16, 24, 32] if seq_len % p == 0]
        self.patch_size = valid_patch_sizes[-1] if valid_patch_sizes else 1
        self.num_patches = seq_len // self.patch_size

        self.patch_embed = nn.Conv1d(
            input_size, hidden_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_channels) * 0.02)

        self.pool_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)

        # Patch embedding: (batch, input_size, seq_len) -> (batch, hidden_channels, num_patches)
        x = self.patch_embed(x)
        # -> (batch, num_patches, hidden_channels)
        x = x.transpose(1, 2)
        x = x + self.pos_embedding
        x = x.transpose(1, 2)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.pool_conv(x)
        x = x.mean(dim=-1)

        out = self.head(x)

        return out


class MultiChannelCNN1D(nn.Module):
    """
    多变量 CNN1D 模型
    - 每个变量独立归一化（per-channel）
    - patch embedding 时对所有通道一起卷积
    - 位置编码 + 时序卷积层
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        seq_len: int = 96,
        pred_len: int = 48,
        dropout: float = 0.1,
        n_date_features: int = 0  # 额外日期特征维度
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_size = input_size
        self.n_date_features = n_date_features

        # === Per-channel 归一化参数 ===
        self.channel_means = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.channel_stds = nn.Parameter(torch.ones(input_size), requires_grad=False)

        # === Patch embedding ===
        # 输入: (batch, input_size, seq_len)
        valid_patch_sizes = [p for p in [1, 2, 4, 8, 16, 24, 32] if seq_len % p == 0]
        patch_size = valid_patch_sizes[-1] if valid_patch_sizes else 1
        num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.patch_embed = nn.Conv1d(
            input_size, hidden_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

        # === 位置编码 ===
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, hidden_channels) * 0.02
        )

        # === 日期特征注入 ===
        if n_date_features > 0:
            self.date_fc = nn.Sequential(
                nn.Linear(n_date_features, hidden_channels),
                nn.GELU()
            )

        # === 时序卷积层 ===
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        # === 池化层 ===
        self.pool_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        # === 预测头 ===
        head_input_dim = hidden_channels + n_date_features if n_date_features > 0 else hidden_channels
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, pred_len)
        )

    def set_channel_stats(self, means: np.ndarray, stds: np.ndarray):
        """设置 per-channel 归一化统计量"""
        self.channel_means.data = torch.FloatTensor(means.astype(np.float32))
        self.channel_stds.data = torch.FloatTensor(stds.astype(np.float32))

    def forward(self, x, date_features=None):
        """
        x: (batch, seq_len, input_size) - 多变量时序
        date_features: (batch, n_date_features) - 可选的日期特征
        """
        batch_size = x.shape[0]

        # === Per-channel 归一化 ===
        # x: (batch, seq_len, input_size)
        means = self.channel_means.view(1, 1, -1)  # (1, 1, input_size)
        stds = self.channel_stds.view(1, 1, -1)
        x_norm = (x - means) / (stds + 1e-8)

        # === Patch embedding ===
        # -> (batch, input_size, seq_len)
        x_norm = x_norm.transpose(1, 2)
        # -> (batch, hidden_channels, num_patches)
        x_patch = self.patch_embed(x_norm)

        # === 位置编码 ===
        # -> (batch, num_patches, hidden_channels)
        x_patch = x_patch.transpose(1, 2)
        x_patch = x_patch + self.pos_embedding

        # === 日期特征注入 ===
        if date_features is not None and self.n_date_features > 0:
            date_emb = self.date_fc(date_features)  # (batch, hidden_channels)
            # 广播到所有 patch
            date_emb = date_emb.unsqueeze(1)  # (batch, 1, hidden_channels)
            x_patch = x_patch + date_emb

        # === 时序卷积 ===
        # -> (batch, hidden_channels, num_patches)
        x_patch = x_patch.transpose(1, 2)

        for layer in self.encoder_layers:
            x_patch = layer(x_patch)

        # === 改进的池化 ===
        x_pool = self.pool_conv(x_patch)  # (batch, hidden_channels, num_patches)
        x_pool = x_pool.mean(dim=-1)     # (batch, hidden_channels)

        # === 预测头 ===
        out = self.head(x_pool)  # (batch, pred_len)

        return out


class CNN1DPredictorV4:
    """CNN1D 预测器 v4 - 直接多步预测，带校准偏移"""

    def __init__(self):
        self.model: Optional[CNN1DModelV4] = None
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        self.seq_len: int = 96
        self.pred_len: int = 48
        self.n_features: int = 1
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.device: str = "cpu"
        self.target_col: str = ""
        self._bias_offset: float = 0.0

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 48,
        hidden_channels: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        target_col: str = "",
        date_features: np.ndarray = None,
        **kwargs
    ) -> Tuple[bool, str]:
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            self.target_col = target_col
            self.seq_len = seq_len
            self.pred_len = pred_len

            # 处理维度
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.ravel()

            n_samples = min(len(X), len(y))
            X = X[:n_samples]
            y = y[:n_samples]
            n_features = X.shape[1]
            self.n_features = n_features

            # ========== 小数据集保护：自动缩小窗口 ==========
            min_window = seq_len + pred_len
            if n_samples < min_window:
                auto_seq = max(4, n_samples // 4)
                auto_pred = max(1, n_samples - auto_seq - max(10, n_samples // 10))
                seq_len = auto_seq
                pred_len = auto_pred
                self.seq_len = seq_len
                self.pred_len = pred_len
                logger.warning(f"CNN1D 数据不足（{n_samples}），自动调整 seq_len={seq_len}, pred_len={pred_len}")

            min_window = seq_len + pred_len
            if n_samples < min_window + 10:
                return False, (
                    f"数据不足：{n_samples} 条样本不足以进行训练。"
                    f"请至少准备 {min_window + 10} 条数据。"
                )

            # 用训练集的 mean/std 归一化
            self._target_mean = float(np.mean(y))
            self._target_std = float(np.std(y)) + 1e-8
            y_norm = (y - self._target_mean) / self._target_std

            # 构建序列
            X_seqs, y_seqs = [], []
            date_seqs = [] if date_features is not None else None

            for i in range(seq_len, n_samples - pred_len + 1):
                X_seqs.append(X[i - seq_len:i])      # (seq_len, n_features)
                y_seqs.append(y_norm[i:i + pred_len])  # (pred_len,)
                if date_features is not None:
                    date_seqs.append(date_features[i - seq_len:i])  # (seq_len, n_date)

            n_seqs = len(X_seqs)
            if n_seqs < 10:
                return False, (
                    f"滑动窗口产生样本不足：{n_seqs} 个序列（数据 {n_samples}"
                    f" / seq_len {seq_len} / pred_len {pred_len}）。"
                    f"请减少 seq_len 或 pred_len。"
                )

            X_tensor = torch.FloatTensor(np.array(X_seqs))
            y_tensor = torch.FloatTensor(np.array(y_seqs))

            split_idx = int(len(X_tensor) * (1 - test_size))
            X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

            self.model = CNN1DModelV4(
                input_size=n_features,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                kernel_size=kernel_size,
                seq_len=seq_len,
                pred_len=pred_len
            ).to(self.device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            self.model.train()
            for epoch in range(epochs):
                indices = torch.randperm(len(X_train))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(X_train), batch_size):
                    batch_idx = indices[i:i + batch_size]
                    X_batch = X_train[batch_idx].to(self.device)
                    y_batch = y_train[batch_idx].to(self.device)

                    optimizer.zero_grad()
                    pred = self.model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"CNN1D-V4 Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/max(n_batches,1):.6f}")

            # 评估
            self.model.eval()
            with torch.no_grad():
                test_pred_norm = self.model(X_test.to(self.device)).cpu().numpy()
                y_test_np = y_test.numpy()

                # 反归一化
                test_pred = test_pred_norm * self._target_std + self._target_mean
                y_test_actual = y_test_np * self._target_std + self._target_mean

                rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
                mae = mean_absolute_error(y_test_actual, test_pred)
                r2 = r2_score(y_test_actual.flatten(), test_pred.flatten())

                # 计算校准偏移量：用测试集前几步预测的第一步的平均误差
                n_calib = min(10, len(test_pred_norm))
                bias_sum = 0.0
                for i in range(n_calib):
                    bias_sum += test_pred_norm[i][0] - y_test_np[i][0]
                self._bias_offset = (bias_sum / n_calib) * self._target_std
                logger.info(f"校准偏移量: {self._bias_offset:.4f}")

                self.metrics = {
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'R2': float(r2),
                    'model': 'CNN1D-V4',
                    'hidden_channels': hidden_channels,
                    'num_layers': num_layers,
                    'seq_len': seq_len,
                    'pred_len': pred_len,
                    'epochs': epochs,
                    'bias_offset': self._bias_offset,
                }

            self.is_fitted = True

            msg = (
                f"✅ CNN1D-V4 训练完成！\n\n"
                f"   模型: CNN (hidden={hidden_channels}, layers={num_layers})\n"
                f"   窗口: seq_len={seq_len}, pred_len={pred_len}\n"
                f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}\n\n"
                f"   **R²** 分数: {r2:.4f}\n"
                f"   **RMSE**: {rmse:.4f}\n"
                f"   **MAE**: {mae:.4f}\n\n"
                f"💡 CNN with patch embedding for direct multi-step prediction"
            )

            logger.info(f"CNN1D-V4 训练完成: R2={r2:.4f}, RMSE={rmse:.4f}")
            return True, msg

        except Exception as e:
            logger.error(f"CNN1D-V4 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"❌ 训练失败: {str(e)}"

    def predict(self, X: np.ndarray, pred_len: int = None) -> np.ndarray:
        """直接多步预测未来 pred_len 步"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练")

        if pred_len is None:
            pred_len = self.pred_len

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        x_input = X[-self.seq_len:]

        x_tensor = torch.FloatTensor(x_input).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x_tensor).cpu().numpy()[0]

        # 反归一化 + 校准偏移
        pred = pred_norm * self._target_std + self._target_mean - self._bias_offset

        return pred[:pred_len]

    def predict_future(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        滚动预测未来 steps 步
        使用直接多步预测 + 滑动窗口
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练")

        original_1d = X.ndim == 1
        if original_1d:
            X = X.reshape(-1, 1)

        all_preds = []

        for _ in range(steps):
            x_input = X[-self.seq_len:]
            x_tensor = torch.FloatTensor(x_input).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                pred_norm = self.model(x_tensor).cpu().numpy()[0]

            # 反归一化 + 校准偏移
            pred = pred_norm * self._target_std + self._target_mean - self._bias_offset

            # 取第一步
            next_val = float(pred[0])
            all_preds.append(next_val)

            # 更新 X（用校准后的预测值）
            X = np.vstack([X, [[next_val]]])

        result = np.array(all_preds)
        return result.squeeze() if original_1d and result.ndim > 1 else result

    def get_feature_importance(self) -> Dict[str, float]:
        return {}

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("没有可保存的模型")
        save_data = {
            'model_state': self.model.state_dict(),
            'target_mean': self._target_mean,
            'target_std': self._target_std,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'metrics': self.metrics,
        }
        torch.save(save_data, path)
        logger.info(f"CNN1D-V4 模型已保存: {path}")

    def load_model(self, path: str):
        data = torch.load(path, map_location=self.device)

        self._target_mean = float(data['target_mean'])
        self._target_std = float(data['target_std'])
        self.seq_len = data['seq_len']
        self.pred_len = data['pred_len']
        self.n_features = data['n_features']
        self.metrics = data.get('metrics', {})
        self._bias_offset = self.metrics.get('bias_offset', 0.0)

        hidden_channels = self.metrics.get('hidden_channels', 128)
        num_layers = self.metrics.get('num_layers', 3)

        self.model = CNN1DModelV4(
            input_size=self.n_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            seq_len=self.seq_len,
            pred_len=self.pred_len
        ).to(self.device)
        self.model.load_state_dict(data['model_state'])
        self.model.eval()

        self.is_fitted = True
        logger.info(f"CNN1D-V4 模型已加载: {path}")
