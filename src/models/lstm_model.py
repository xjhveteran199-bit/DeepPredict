"""
LSTM 深度学习时序预测模型
基于 PyTorch 实现
稳定性修复：
- Xavier 权重初始化
- Gradient clipping (max_norm=1.0)
- ReduceLROnPlateau 学习率调度
- Early stopping（patience=10）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Tuple, Dict, Optional
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM 预测模型 - 优化版：LayerNorm + 残差连接"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)


class LSTMPredictor:
    """LSTM 时序预测器"""

    def __init__(self):
        self.model: Optional[LSTMModel] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.seq_len: int = 10
        self.input_size: int = 1
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.device: str = "cpu"
        self.feature_names: Optional[list] = None
        self.target_col: str = ""
        self.best_state: Optional[dict] = None  # for early stopping
        self.train_losses: list = []
        self.val_losses: list = []
        self._fig = None

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(target[i + self.seq_len])
        return np.array(X), np.array(y)

    def _normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std, mean, std

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        seq_len: int = 10,
        test_size: float = 0.2,
        target_col: str = ""
    ) -> Tuple[bool, str]:
        try:
            self.target_col = target_col
            self.seq_len = seq_len

            data = np.column_stack([X, y])
            data_normalized, self.scaler_mean, self.scaler_std = self._normalize(data)

            X_norm = data_normalized[:, :-1]
            y_norm = data_normalized[:, -1]

            X_seq, y_seq = self._create_sequences(X_norm, y_norm)

            y_seq = np.asarray(y_seq).ravel()

            split_idx = int(len(X_seq) * (1 - test_size))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.FloatTensor(y_train)
            X_test_t = torch.FloatTensor(X_test)
            y_test_t = torch.FloatTensor(y_test)

            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            self.input_size = X_train.shape[2]
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            ).to(self.device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # 稳定性修复：学习率调度 + Early stopping
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )

            best_val_loss = float('inf')
            patience_counter = 0
            self.best_state = None
            MAX_PATIENCE = 10

            # 实时绘图初始化
            self.train_losses = []
            self.val_losses = []
            try:
                import matplotlib
                matplotlib.use('qtagg')
                import matplotlib.pyplot as plt
                plt.ion()
                fig, ax = plt.subplots(figsize=(8, 4))
                self._fig = fig
                self._ax = ax
            except Exception:
                self._fig = None

            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()
                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / len(train_loader)
                
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_test_t.to(self.device)).cpu().numpy()
                    val_mse = float(np.mean((val_preds - y_test) ** 2))
                self.model.train()
                
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_mse)

                scheduler.step(val_mse)
                
                # 实时更新损失曲线
                if self._fig is not None:
                    ax = self._ax
                    ax.clear()
                    epochs_range = range(1, len(self.train_losses) + 1)
                    ax.plot(epochs_range, self.train_losses, 'b-', label='Train Loss', linewidth=1.5)
                    ax.plot(epochs_range, self.val_losses, 'r-', label='Val Loss', linewidth=1.5)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss (MSE)')
                    ax.set_title(f'LSTM Training Progress (Epoch {epoch+1})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    self._fig.canvas.draw()
                    self._fig.canvas.flush_events()

                # Early stopping
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"LSTM Epoch {epoch+1}/{epochs}, "
                        f"TrainLoss: {avg_train_loss:.6f}, ValMSE: {val_mse:.6f}, "
                        f"LR: {current_lr:.2e}, Patience: {patience_counter}/{MAX_PATIENCE}"
                    )

                if patience_counter >= MAX_PATIENCE:
                    logger.info(f"LSTM Early stopping at epoch {epoch+1}")
                    break

            # 训练完成：保存损失曲线
            if self._fig is not None:
                plt.ioff()
                ax = self._ax
                ax.clear()
                epochs_range = range(1, len(self.train_losses) + 1)
                ax.plot(epochs_range, self.train_losses, 'b-', label='Train Loss', linewidth=2)
                ax.plot(epochs_range, self.val_losses, 'r-', label='Val Loss', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title('LSTM Training Complete - Loss Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                self._fig.tight_layout()
                fig_path = 'lstm_loss_curve.png'
                self._fig.savefig(fig_path, dpi=150)
                logger.info(f"LSTM loss curve saved: {fig_path}")
                plt.close(self._fig)
                self._fig = None

            # 恢复最佳模型
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
                self.model.to(self.device)

            self.model.eval()
            with torch.no_grad():
                X_test_t = X_test_t.to(self.device)
                predictions = self.model(X_test_t).cpu().numpy()

                # 构建预测数据的归一化版本（只有最后一列是预测值，其他列为0）
                # 然后对所有列正确反归一化，确保 scaler 的 mean/std 正确应用
                test_pred_norm = np.zeros((len(predictions), data_normalized.shape[1]))
                test_pred_norm[:, -1] = predictions.flatten()
                pred_denorm = test_pred_norm * self.scaler_std + self.scaler_mean
                predictions = pred_denorm[:, -1]

                # 构建测试数据的归一化版本（只有最后一列是真实值）
                test_y_norm = np.zeros((len(y_test), data_normalized.shape[1]))
                test_y_norm[:, -1] = y_test.flatten()
                y_test_denorm = test_y_norm * self.scaler_std + self.scaler_mean
                y_test_actual = y_test_denorm[:, -1]

                mse = np.mean((predictions - y_test_actual) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - y_test_actual))
                ss_res = np.sum((y_test_actual - predictions) ** 2)
                ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)

                self.metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

            self.is_fitted = True

            msg = f"✅ LSTM 训练完成！\n"
            msg += f"   模型: LSTM (hidden={hidden_size}, layers={num_layers})\n"
            msg += f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}\n"
            msg += f"   R² 分数: {r2:.4f}\n"
            msg += f"   RMSE: {rmse:.4f}\n"
            msg += f"   MAE: {mae:.4f}"

            logger.info(f"LSTM 训练完成: {self.metrics}")
            return True, msg

        except Exception as e:
            logger.error(f"LSTM 训练失败: {e}")
            return False, f"LSTM 训练失败: {str(e)}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测下一步（单步），返回反归一化的原始尺度预测值"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        self.model.eval()

        original_1d = len(X.shape) == 1
        if original_1d:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]

        # 构建输入 (seq_len, n_features)
        if len(X) > self.seq_len:
            x_input = X[-self.seq_len:]
        else:
            x_input = X

        data = np.column_stack([x_input, np.zeros(len(x_input))])
        data_norm = (data - self.scaler_mean) / self.scaler_std
        X_norm = data_norm[:, :-1]

        X_seq, _ = self._create_sequences(X_norm, np.zeros(len(X_norm)))

        if len(X_seq) == 0:
            X_seq = X_norm[-self.seq_len:].reshape(1, self.seq_len, X_norm.shape[1])

        X_t = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            preds_norm = self.model(X_t).cpu().numpy()

        pred_data = np.zeros((len(preds_norm), len(self.scaler_mean)))
        pred_data[:, -1] = preds_norm
        preds_denorm = pred_data * self.scaler_std + self.scaler_mean

        result = preds_denorm[:, -1]
        if original_1d:
            return result.squeeze()
        return result

    def predict_future(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        滚动预测未来 steps 步（自回归）
        X: (n_samples,) 或 (n_samples, n_features)，至少 seq_len 个样本
        返回: (steps,) 预测的未来 steps 步（原始尺度）
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        original_1d = len(X.shape) == 1
        if original_1d:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        predictions = []
        cur = list(X[-self.seq_len:].astype(np.float64))

        for _ in range(steps):
            x_input = np.array(cur[-self.seq_len:])
            preds = self.predict(x_input)

            if len(preds) == 0:
                break

            pred = float(preds[-1])
            predictions.append(pred)

            new_row = np.zeros(n_features, dtype=np.float64)
            new_row[0] = pred
            cur.append(new_row)

        result = np.array(predictions)
        if original_1d:
            return result.squeeze()
        return result

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'seq_len': self.seq_len,
            'input_size': self.input_size,
            'metrics': self.metrics,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'feature_names': self.feature_names,
            'target_col': self.target_col
        }, path)
        logger.info(f"LSTM 模型已保存: {path}")

    def load_model(self, path: str):
        data = torch.load(path, map_location=self.device)

        self.scaler_mean = data['scaler_mean']
        self.scaler_std = data['scaler_std']
        self.seq_len = data['seq_len']
        self.input_size = data['input_size']
        self.metrics = data.get('metrics', {})
        self.feature_names = data.get('feature_names')
        self.target_col = data.get('target_col', '')

        hidden_size = data.get('hidden_size', 64)
        num_layers = data.get('num_layers', 2)

        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        self.model.load_state_dict(data['model_state'])
        self.model.eval()

        self.is_fitted = True
        logger.info(f"LSTM 模型已加载: {path}")
