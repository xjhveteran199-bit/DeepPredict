"""
CNN1D 时序预测模型 v4
完全参照 PatchTST 成功模式：
1. 直接多步预测（一次性输出所有 pred_len 步）
2. RevIN 归一化
3. 直接用最后 seq_len 步预测接下来 pred_len 步
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class RevIN:
    """Reversible Instance Normalization - 时序预测专用归一化"""
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.mean = None
        self.std = None
    
    def forward(self, x):
        # x: (batch, seq_len) or (batch, seq_len, channels)
        if self.mean is None:
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
        return (x - self.mean) / self.std
    
    def backward(self, x):
        # x: (batch, seq_len) or (batch, seq_len, channels)
        return x * self.std + self.mean


class CNN1DModelV4(nn.Module):
    """
    改进版 CNN1D 时序预测模型
    
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
        
        # Patch embedding：用卷积做 patch
        # 把 seq_len 分成 patches，每个 patch 大小 = patch_size
        self.patch_size = 16
        self.num_patches = seq_len // self.patch_size
        
        self.patch_embed = nn.Conv1d(
            input_size, hidden_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 多层 CNN 特征提取
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        # 预测头：输出 pred_len 步
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.shape[0]
        
        # 转 (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Patch embedding: (batch, input_size, seq_len) -> (batch, hidden_channels, num_patches)
        x = self.patch_embed(x)
        
        # CNN 编码
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 全局平均池化: (batch, hidden_channels, num_patches) -> (batch, hidden_channels)
        x = x.mean(dim=-1)
        
        # 预测
        out = self.head(x)  # (batch, pred_len)
        
        return out


class CNN1DPredictorV4:
    """CNN1D 预测器 v4 - 直接多步预测，带校准偏移"""

    def __init__(self):
        self.model: Optional[CNN1DModelV4] = None
        self.revin = None
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        self.seq_len: int = 96
        self.pred_len: int = 48
        self.n_features: int = 1
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.device: str = "cpu"
        self.target_col: str = ""
        self._bias_offset: float = 0.0  # 校准偏移量

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

            # 用训练集的 mean/std 归一化
            self._target_mean = float(np.mean(y))
            self._target_std = float(np.std(y)) + 1e-8
            y_norm = (y - self._target_mean) / self._target_std

            # 构建序列
            X_seqs, y_seqs = [], []
            for i in range(seq_len, n_samples - pred_len + 1):
                X_seqs.append(X[i - seq_len:i])      # (seq_len, n_features)
                y_seqs.append(y_norm[i:i + pred_len])  # (pred_len,)
            
            n_seqs = len(X_seqs)
            if n_seqs < 10:
                return False, f"样本不足：{n_seqs}"

            X_tensor = torch.FloatTensor(np.array(X_seqs))
            y_tensor = torch.FloatTensor(np.array(y_seqs))
            
            # 划分
            split_idx = int(len(X_tensor) * (1 - test_size))
            X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

            # 模型
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
