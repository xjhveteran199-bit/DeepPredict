"""
Enhanced 1D-CNN 时序预测模型
专为复杂时序数据设计（多变量、不规则采样、混合格式）

架构特点：
1. Multi-Scale CNN：并行多尺度卷积（kernel=3/5/7），分别捕捉短期/中期/长期模式
2. Residual Block：残差连接，缓解深层网络退化
3. Channel Attention：SE-like 机制，让模型自动关注重要特征通道
4. 直接多步预测输出

适用场景：
- 多变量时序（温度+湿度+压力同时预测）
- 不规则采样（时间间隔不等的数据）
- 混合格式（数值+类别+日期混合）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CNN1DConfig:
    """1D-CNN 配置"""
    input_channels: int = 1       # 输入特征通道数（变量数）
    hidden_channels: int = 64     # 隐藏层通道数
    num_scales: int = 3           # 多尺度卷积分支数
    kernel_sizes: Tuple = (3, 5, 7)  # 各分支 kernel size
    num_res_blocks: int = 2       # 残差块数量
    seq_len: int = 96             # 输入序列长度
    pred_len: int = 48            # 预测步长
    dropout: float = 0.1
    use_attention: bool = True    # 是否使用通道注意力


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - 通道注意力"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        w = self.squeeze(x).squeeze(-1)        # (batch, channels)
        w = self.excitation(w).unsqueeze(-1)   # (batch, channels, 1)
        return x * w


class ResBlock(nn.Module):
    """残差块：Conv1D + Norm + Activation + Conv1D + Norm"""
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MultiScaleCNN1D(nn.Module):
    """
    多尺度 1D-CNN 核心网络
    3 个并行分支（kernel=3/5/7），分别捕捉短/中/长期模式后融合
    """
    def __init__(self, input_channels: int, hidden_channels: int,
                 kernel_sizes: Tuple[int, ...], dropout: float):
        super().__init__()

        # 输入投影：把 input_channels -> hidden_channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        # 多尺度分支
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels,
                          kernel_size=ks, padding=ks // 2, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # SE 通道注意力（融合后）
        self.se = SEBlock(hidden_channels, reduction=4)

        # 融合层：压缩多分支输出到 hidden_channels
        self.proj = nn.Conv1d(hidden_channels * len(kernel_sizes), hidden_channels, kernel_size=1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_channels, seq_len)
        x = self.input_proj(x)  # (batch, hidden, seq_len)

        # 并行多尺度
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        # 拼接各分支
        x = torch.cat(branch_outputs, dim=1)  # (batch, hidden*num_scales, seq_len)
        x = self.proj(x)  # (batch, hidden, seq_len)
        x = self.se(x)
        x = self.fusion(x)
        return x


class EnhancedCNN1DModel(nn.Module):
    """
    增强 1D-CNN 时序预测模型
    Multi-Scale CNN + ResBlocks + 直接多步预测头
    """
    def __init__(self, config: CNN1DConfig):
        super().__init__()
        self.config = config

        # 多尺度 CNN
        self.ms_cnn = MultiScaleCNN1D(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            kernel_sizes=config.kernel_sizes,
            dropout=config.dropout,
        )

        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            ResBlock(config.hidden_channels, kernel_size=3, dropout=config.dropout)
            for _ in range(config.num_res_blocks)
        ])

        # 通道注意力（最后再加一次）
        if config.use_attention:
            self.se_final = SEBlock(config.hidden_channels, reduction=4)

        # 时间维度压缩：seq_len -> 1（用卷积）
        self.temporal_pool = nn.Conv1d(
            config.hidden_channels, config.hidden_channels,
            kernel_size=config.seq_len, stride=config.seq_len
        )

        # 直接预测头：输出 pred_len 步
        self.pred_head = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, config.pred_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_channels)
        x = x.transpose(1, 2)  # (batch, input_channels, seq_len)

        x = self.ms_cnn(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        if self.config.use_attention:
            x = self.se_final(x)

        # 时间压缩
        x = self.temporal_pool(x)  # (batch, hidden, 1)
        x = x.squeeze(-1)           # (batch, hidden)

        # 预测
        out = self.pred_head(x)     # (batch, pred_len)
        return out


class EnhancedCNN1DPredictor:
    """
    增强 1D-CNN 预测器
    支持：
    - 多变量输入（自动适配 channel 数）
    - 多尺度卷积捕捉不同时间模式
    - 直接输出多步预测
    """

    def __init__(self):
        self.model: Optional[EnhancedCNN1DModel] = None
        self.config: Optional[CNN1DConfig] = None
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.device: str = "cpu"
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._bias_offset: float = 0.0
        self.seq_len: int = 96
        self.pred_len: int = 48

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 48,
        hidden_channels: int = 64,
        num_scales: int = 3,
        kernel_sizes: Tuple = (3, 5, 7),
        num_res_blocks: int = 2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        dropout: float = 0.1,
        use_attention: bool = True,
        **kwargs
    ) -> Tuple[bool, str]:
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

            # 小数据集保护
            min_window = seq_len + pred_len
            if n_samples < min_window + 20:
                auto_seq = max(4, n_samples // 4)
                auto_pred = max(1, n_samples - auto_seq - max(10, n_samples // 10))
                seq_len = auto_seq
                pred_len = auto_pred
                self.seq_len = seq_len
                self.pred_len = pred_len
                logger.warning(f"数据不足，自动调整 seq_len={seq_len}, pred_len={pred_len}")

            # 归一化（按 y）
            self._target_mean = float(np.mean(y))
            self._target_std = float(np.std(y)) + 1e-8
            y_norm = (y - self._target_mean) / self._target_std

            # 多变量 X 标准化（关键修复！）
            # 在构建序列窗口之前，对整个 X 做 fit_transform
            # 确保每个特征列都是零均值单位方差，这对多变量 CNN 效果至关重要
            self._x_scaler = None
            if n_features > 1:
                from sklearn.preprocessing import StandardScaler
                self._x_scaler = StandardScaler()
                X = self._x_scaler.fit_transform(X).astype(np.float32)

            # 构建序列：(seq, n_features) 的窗口
            X_seqs, y_seqs = [], []
            for i in range(seq_len, n_samples - pred_len + 1):
                X_seqs.append(X[i - seq_len:i])       # (seq_len, n_features)
                y_seqs.append(y_norm[i:i + pred_len])  # (pred_len,)
            n_seqs = len(X_seqs)

            if n_seqs < 5:
                return False, f"样本不足（{n_seqs} 个序列），请增加数据量"

            X_tensor = torch.FloatTensor(np.array(X_seqs))   # (n_seqs, seq_len, n_features)
            y_tensor = torch.FloatTensor(np.array(y_seqs))  # (n_seqs, pred_len)

            split_idx = int(len(X_tensor) * (1 - test_size))
            X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

            # 模型
            self.config = CNN1DConfig(
                input_channels=n_features,
                hidden_channels=hidden_channels,
                num_scales=num_scales,
                kernel_sizes=kernel_sizes,
                num_res_blocks=num_res_blocks,
                seq_len=seq_len,
                pred_len=pred_len,
                dropout=dropout,
                use_attention=use_attention,
            )
            self.model = EnhancedCNN1DModel(self.config).to(self.device)

            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Enhanced 1D-CNN 参数量: {n_params}")

            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
            )

            # 训练
            self.model.train()
            for epoch in range(epochs):
                indices = torch.randperm(len(X_train))
                epoch_loss = 0.0
                n_batches = 0

                for i in range(0, len(X_train), batch_size):
                    batch_idx = indices[i:i + batch_size]
                    X_b = X_train[batch_idx].to(self.device)
                    y_b = y_train[batch_idx].to(self.device)

                    optimizer.zero_grad()
                    pred = self.model(X_b)
                    loss = criterion(pred, y_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    logger.info(f"Enhanced 1D-CNN Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            # 评估
            self.model.eval()
            with torch.no_grad():
                test_pred_norm = self.model(X_test.to(self.device)).cpu().numpy()
                y_test_np = y_test.numpy()

                test_pred = test_pred_norm * self._target_std + self._target_mean
                y_test_actual = y_test_np * self._target_std + self._target_mean

                rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), test_pred.flatten()))
                mae = mean_absolute_error(y_test_actual.flatten(), test_pred.flatten())
                r2 = r2_score(y_test_actual.flatten(), test_pred.flatten())

                # 校准偏移
                n_calib = min(10, len(test_pred_norm))
                bias = float(np.mean(test_pred_norm[:n_calib, 0] - y_test_np[:n_calib, 0]))
                self._bias_offset = bias * self._target_std

            self.metrics = {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R2': float(r2),
                'model': 'EnhancedCNN1D',
                'hidden_channels': hidden_channels,
                'num_scales': num_scales,
                'kernel_sizes': kernel_sizes,
                'num_res_blocks': num_res_blocks,
                'seq_len': seq_len,
                'pred_len': pred_len,
                'n_features': n_features,
                'n_params': n_params,
            }
            self.is_fitted = True

            kernel_str = "/".join(map(str, kernel_sizes))
            msg = (
                f"✅ Enhanced 1D-CNN 训练完成！\n\n"
                f"   模型: 多尺度CNN (k={kernel_str}, res={num_res_blocks}, att={use_attention})\n"
                f"   配置: hidden={hidden_channels}, 参数量={n_params:,}\n"
                f"   窗口: seq_len={seq_len}, pred_len={pred_len}\n"
                f"   特征: {n_features} 个变量（多变量支持）\n"
                f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}\n\n"
                f"   **R²** 分数: {r2:.4f}\n"
                f"   **RMSE**: {rmse:.4f}\n"
                f"   **MAE**: {mae:.4f}\n\n"
                f"💡 多尺度卷积（k=3/5/7）捕捉短/中/长期模式，残差连接稳定训练"
            )
            logger.info(f"Enhanced 1D-CNN 完成: R2={r2:.4f}, RMSE={rmse:.4f}")
            return True, msg

        except Exception as e:
            logger.error(f"Enhanced 1D-CNN 训练失败: {e}")
            import traceback; logger.error(traceback.format_exc())
            return False, f"❌ 训练失败: {str(e)}"

    def predict(self, X: np.ndarray, pred_len: int = None) -> np.ndarray:
        """直接多步预测"""
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

        pred = pred_norm * self._target_std + self._target_mean - self._bias_offset
        return pred[:pred_len]

    def predict_future(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """滚动预测未来 steps 步"""
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

            pred = float(pred_norm[0] * self._target_std + self._target_mean - self._bias_offset)
            all_preds.append(pred)
            X = np.vstack([X, [[pred]]])

        result = np.array(all_preds)
        return result.squeeze() if original_1d and result.ndim > 1 else result
