"""
CNN1D 时序预测模型
基于 PyTorch 实现 1D 卷积神经网络，适合传感器等规则化时序信号
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class CNN1DBlock(nn.Module):
    """单层 1D CNN block: Conv -> BN -> ReLU -> Pooling"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

    def forward(self, x):
        return self.block(x)


class CNN1DModel(nn.Module):
    """
    1D CNN 时序预测模型
    - 多层 1D 卷积提取局部特征
    - 全局平均池化聚合时序信息
    - FC 头输出预测值
    """

    def __init__(
        self,
        input_size: int = 1,      # 输入通道数（特征数）
        hidden_channels: list = None,
        kernel_size: int = 3,
        seq_len: int = 100,
        dropout: float = 0.2
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]

        self.seq_len = seq_len
        self.input_size = input_size

        # 构建卷积层
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(hidden_channels):
            pool_size = 2 if i < len(hidden_channels) - 1 else 1
            layers.append(CNN1DBlock(in_ch, out_ch, kernel_size, pool_size))
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # 计算展平后的维度（需要根据实际计算，或用自适应）
        # 简化：用自适应全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接头
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # Conv1d 需要 (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        x = self.global_pool(x).squeeze(-1)  # (batch, last_hidden)
        x = self.dropout(x)
        out = self.fc(x).squeeze(-1)
        return out


class CNN1DPredictor:
    """
    CNN1D 时序预测器
    支持多通道输入，适合传感器融合场景
    """

    def __init__(self):
        self.model: Optional[CNN1DModel] = None
        self.scaler_X_mean: Optional[np.ndarray] = None
        self.scaler_X_std: Optional[np.ndarray] = None
        self.scaler_y_mean: Optional[np.ndarray] = None
        self.scaler_y_std: Optional[np.ndarray] = None
        self.seq_len: int = 100
        self.n_features: int = 1
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.device: str = "cpu"
        self.target_col: str = ""

    def _normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std, mean, std

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            # data[i:i+self.seq_len] 的 shape 取决于 data 维度：
            # 1D: (seq_len,) -> ok
            # 2D: (seq_len, n_features) -> ok, list of 2D arrays
            X.append(np.array(data[i:i + self.seq_len]))
            y.append(target[i + self.seq_len])
        # 确保 X 是 3D (samples, seq_len, features), y 是 1D 或 2D
        X = np.array(X)
        if len(X.shape) == 2:
            X = X.reshape(-1, self.seq_len, 1)  # (n, seq_len, 1)
        y = np.array(y)
        if len(y.shape) == 2:
            y = y.squeeze(-1)
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        hidden_channels: list = None,
        kernel_size: int = 3,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        target_col: str = "",
        **kwargs
    ) -> Tuple[bool, str]:
        """
        训练 CNN1D 模型

        Args:
            X: 特征数据 (n_samples, n_features)，支持多通道
            y: 目标数据 (n_samples,)
            seq_len: 输入序列长度
            hidden_channels: 卷积通道列表，如 [32, 64, 128]
            kernel_size: 卷积核大小
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            test_size: 测试集比例
        """
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            if hidden_channels is None:
                hidden_channels = [32, 64, 128]

            self.target_col = target_col
            self.seq_len = seq_len

            # 处理维度
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            n_samples = min(len(X), len(y))
            X = X[:n_samples]
            y = y[:n_samples]
            n_features = X.shape[1]
            self.n_features = n_features

            # 归一化
            X_norm, self.scaler_X_mean, self.scaler_X_std = self._normalize(X)
            y_norm, self.scaler_y_mean, self.scaler_y_std = self._normalize(y)
            # 展平 y：y_norm 可能是 (n, n_targets)，取第一列再展平避免列数×样本数膨胀
            if len(y_norm.shape) > 1 and y_norm.shape[1] > 1:
                logger.warning(f"y 有 {y_norm.shape[1]} 列，只取第一列作为预测目标")
                y_norm = y_norm[:, :1]
            y_flat = y_norm.ravel()  # (n,)

            # 构建序列
            X_seq, y_seq = self._create_sequences(X_norm, y_flat)

            # 划分
            split_idx = int(len(X_seq) * (1 - test_size))
            if split_idx < 10:
                return False, f"训练数据不足（{split_idx} 个样本），请减少 seq_len 或增加数据量"

            X_train = torch.FloatTensor(X_seq[:split_idx])
            y_train = torch.FloatTensor(y_seq[:split_idx])
            X_test = torch.FloatTensor(X_seq[split_idx:])
            y_test = torch.FloatTensor(y_seq[split_idx:])

            # 模型
            self.model = CNN1DModel(
                input_size=n_features,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                seq_len=seq_len
            ).to(self.device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            train_losses = []
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
                avg_loss = epoch_loss / max(n_batches, 1)
                train_losses.append(avg_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"CNN1D Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            # 评估
            self.model.eval()
            with torch.no_grad():
                test_pred_norm = self.model(X_test.to(self.device)).cpu().numpy()
                y_test_np = y_test.numpy()

                # 反归一化
                test_pred = test_pred_norm * self.scaler_y_std + self.scaler_y_mean
                y_test_actual = y_test_np * self.scaler_y_std + self.scaler_y_mean

                mse = mean_squared_error(y_test_actual, test_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_actual, test_pred)
                ss_res = np.sum((y_test_actual - test_pred) ** 2)
                ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)

                self.metrics = {
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'R2': float(r2),
                    'model': 'CNN1D',
                    'hidden_channels': hidden_channels,
                    'kernel_size': kernel_size,
                    'seq_len': seq_len,
                    'epochs': epochs,
                    'final_train_loss': float(train_losses[-1])
                }

            self.is_fitted = True

            msg = (
                f"✅ CNN1D 训练完成！\n\n"
                f"   模型: 1D-CNN (channels={hidden_channels})\n"
                f"   窗口: seq_len={seq_len}, kernel={kernel_size}\n"
                f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}\n\n"
                f"   **R²** 分数: {r2:.4f}\n"
                f"   **RMSE**: {rmse:.4f}\n"
                f"   **MAE**: {mae:.4f}\n\n"
                f"💡 CNN1D 擅长时间序列局部特征提取，适合传感器信号"
            )

            logger.info(f"CNN1D 训练完成: R2={r2:.4f}, RMSE={rmse:.4f}")
            return True, msg

        except Exception as e:
            logger.error(f"CNN1D 训练失败: {e}")
            import traceback; logger.error(traceback.format_exc())
            return False, f"❌ CNN1D 训练失败: {str(e)}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        批量预测：给定连续的 seq_len 窗口，预测每个窗口之后 1 步
        X: (n_samples, n_features) 或 (n_samples,)
        返回: (n_windows,) 预测值，每个对应输入窗口之后 1 步
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_needed = self.seq_len
        if len(X) < n_needed:
            raise ValueError(f"需要至少 {n_needed} 个样本，当前只有 {len(X)} 个")

        # 归一化
        x_norm = (X - self.scaler_X_mean) / (self.scaler_X_std + 1e-8)

        # 滑动窗口：每次滑 1 步
        predictions = []
        for i in range(len(x_norm) - n_needed + 1):
            window = x_norm[i:i + n_needed]  # (seq_len, n_features)
            window_t = torch.FloatTensor(window).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)
            with torch.no_grad():
                pred_norm = self.model(window_t).cpu().numpy()[0]  # scalar
            pred = pred_norm * self.scaler_y_std + self.scaler_y_mean
            predictions.append(pred)

        return np.array(predictions)

    def predict_future(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        滚动预测未来 steps 步（自回归）
        用当前窗口预测下一步，然后将预测值加入窗口继续预测
        X: (n_samples, n_features) 或 (n_samples,)，至少 seq_len 个样本
        返回: (steps,) 预测的未来 steps 步
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        # 统一 reshape 为 2D
        original_1d = len(X.shape) == 1
        if original_1d:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        predictions = []

        # 初始化滑动窗口: (seq_len, n_features) numpy array
        cur = X[-self.seq_len:].copy()  # float64, shape: (seq_len, n_features)

        for _ in range(steps):
            # 归一化
            x_norm = (cur - self.scaler_X_mean) / (self.scaler_X_std + 1e-8)
            x_t = torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)

            self.model.eval()
            with torch.no_grad():
                pred_norm = self.model(x_t).cpu().numpy()[0]  # scalar

            # 反归一化 - 使用 .item() 处理 0D numpy array
            pred_raw = pred_norm * self.scaler_y_std + self.scaler_y_mean
            if hasattr(pred_raw, 'item'):
                pred = pred_raw.item()
            else:
                pred = float(pred_raw)
            predictions.append(pred)

            # 滑动窗口: 去掉第一行，追加预测值作为新行
            new_row = np.full((1, n_features), pred, dtype=np.float64)
            cur = np.vstack([cur[1:], new_row])

        if original_1d:
            return np.array(predictions).squeeze()
        return np.array(predictions)

    def get_feature_importance(self) -> Dict[str, float]:
        """1D-CNN 不直接支持特征重要性，返回空"""
        return {}

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("没有可保存的模型")

        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_X_mean': self.scaler_X_mean,
            'scaler_X_std': self.scaler_X_std,
            'scaler_y_mean': self.scaler_y_mean,
            'scaler_y_std': self.scaler_y_std,
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'metrics': self.metrics,
            'target_col': self.target_col
        }, path)
        logger.info(f"CNN1D 模型已保存: {path}")

    def load_model(self, path: str):
        data = torch.load(path, map_location=self.device)

        self.scaler_X_mean = data['scaler_X_mean']
        self.scaler_X_std = data['scaler_X_std']
        self.scaler_y_mean = data['scaler_y_mean']
        self.scaler_y_std = data['scaler_y_std']
        self.seq_len = data['seq_len']
        self.n_features = data['n_features']
        self.metrics = data.get('metrics', {})
        self.target_col = data.get('target_col', '')

        hidden_channels = self.metrics.get('hidden_channels', [32, 64, 128])
        self.model = CNN1DModel(
            input_size=self.n_features,
            hidden_channels=hidden_channels,
            seq_len=self.seq_len
        ).to(self.device)
        self.model.load_state_dict(data['model_state'])
        self.model.eval()

        self.is_fitted = True
        logger.info(f"CNN1D 模型已加载: {path}")
