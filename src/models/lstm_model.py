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
    """LSTM 预测模型 - 优化版：LayerNorm + 残差连接，支持多目标输出"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

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
            nn.Linear(hidden_size // 2, output_size)
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
        return out


class LSTMPredictor:
    """LSTM 时序预测器，支持多目标输出"""

    def __init__(self):
        self.model: Optional[LSTMModel] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.seq_len: int = 10
        self.input_size: int = 1
        self.output_size: int = 1  # number of target columns
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.device: str = "cpu"
        self.feature_names: Optional[list] = None
        self.target_cols: list = []  # 支持多目标
        self.best_state: Optional[dict] = None  # for early stopping
        # 训练历史：完整指标
        self.train_history: Dict[str, list] = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_r2': [], 'val_r2': []
        }
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
            self.target_cols = [target_col] if isinstance(target_col, str) else list(target_col)
            self.seq_len = seq_len

            # 确保 y 是 2D 数组 (n_samples, n_targets)
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            self.output_size = y_arr.shape[1]

            # 合并 X 和 y 用于联合归一化 (保留最后一列为第一个目标,倒数第二列为第二个...)
            # 真实顺序: [X_features..., y_target1, y_target2, ...]
            data = np.column_stack([X, y_arr])
            data_normalized, self.scaler_mean, self.scaler_std = self._normalize(data)

            # 分离归一化后的 X 和 y
            n_x_cols = X.shape[1]
            X_norm = data_normalized[:, :n_x_cols]
            y_norm = data_normalized[:, n_x_cols:]  # shape: (n_samples, n_targets)

            X_seq, y_seq = self._create_sequences(X_norm, y_norm)
            # y_seq shape: (n_sequences, n_targets)

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
                num_layers=num_layers,
                output_size=self.output_size
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

            # 初始化训练历史
            self.train_history = {
                'epoch': [], 'train_loss': [], 'val_loss': [],
                'train_mae': [], 'val_mae': [],
                'train_r2': [], 'val_r2': []
            }

            # 实时绘图初始化（使用 Agg 后端，无 GUI 环境也能运行）
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                self._fig = fig
                self._ax = ax
                self._plt = plt
            except Exception:
                self._fig = None
                self._plt = None

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
                    val_preds_norm = self.model(X_test_t.to(self.device)).cpu().numpy()
                    val_mse = float(np.mean((val_preds_norm - y_test) ** 2))

                    # 多目标反归一化
                    n_test = len(val_preds_norm)
                    val_preds_denorm = np.zeros((n_test, len(self.scaler_mean)))
                    val_preds_denorm[:, n_x_cols:] = val_preds_norm
                    val_preds_denorm = val_preds_denorm * self.scaler_std + self.scaler_mean
                    val_preds = val_preds_denorm[:, n_x_cols:]

                    y_test_denorm = np.zeros((n_test, len(self.scaler_mean)))
                    y_test_denorm[:, n_x_cols:] = y_test
                    y_test_denorm = y_test_denorm * self.scaler_std + self.scaler_mean
                    y_test_actual = y_test_denorm[:, n_x_cols:]

                    # 多目标 MAE / R²
                    val_mae = float(np.mean(np.abs(val_preds - y_test_actual)))
                    ss_res = np.sum((y_test_actual - val_preds) ** 2)
                    ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
                    val_r2 = float(1 - ss_res / (ss_tot + 1e-8))

                    # 训练集指标
                    train_preds_norm = self.model(X_train_t.to(self.device)).cpu().numpy()
                    train_preds_denorm = np.zeros((len(train_preds_norm), len(self.scaler_mean)))
                    train_preds_denorm[:, n_x_cols:] = train_preds_norm
                    train_preds_denorm = train_preds_denorm * self.scaler_std + self.scaler_mean
                    train_preds = train_preds_denorm[:, n_x_cols:]

                    y_train_denorm = np.zeros((len(y_train), len(self.scaler_mean)))
                    y_train_denorm[:, n_x_cols:] = y_train
                    y_train_denorm = y_train_denorm * self.scaler_std + self.scaler_mean
                    y_train_actual = y_train_denorm[:, n_x_cols:]

                    train_mae = float(np.mean(np.abs(train_preds - y_train_actual)))
                    ss_res_tr = np.sum((y_train_actual - train_preds) ** 2)
                    ss_tot_tr = np.sum((y_train_actual - np.mean(y_train_actual)) ** 2)
                    train_r2 = float(1 - ss_res_tr / (ss_tot_tr + 1e-8))

                self.model.train()

                # 记录训练历史
                self.train_history['epoch'].append(epoch + 1)
                self.train_history['train_loss'].append(avg_train_loss)
                self.train_history['val_loss'].append(val_mse)
                self.train_history['train_mae'].append(train_mae)
                self.train_history['val_mae'].append(val_mae)
                self.train_history['train_r2'].append(train_r2)
                self.train_history['val_r2'].append(val_r2)

                scheduler.step(val_mse)

                # 实时更新损失曲线（仅在有 GUI 时刷新，避免 headless 报错）
                if self._fig is not None and self._plt is not None:
                    try:
                        ax = self._ax
                        ax.clear()
                        epochs_range = self.train_history['epoch']
                        ax.plot(epochs_range, self.train_history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
                        ax.plot(epochs_range, self.train_history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss (MSE)')
                        ax.set_title(f'LSTM Training Progress (Epoch {epoch+1})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        self._fig.canvas.draw()
                        self._fig.canvas.flush_events()
                    except Exception:
                        pass  # headless 环境 canvas.draw 可能失败，忽略

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
                        f"TrainR2: {train_r2:.4f}, ValR2: {val_r2:.4f}, "
                        f"LR: {current_lr:.2e}, Patience: {patience_counter}/{MAX_PATIENCE}"
                    )

                if patience_counter >= MAX_PATIENCE:
                    logger.info(f"LSTM Early stopping at epoch {epoch+1}")
                    break

            # 训练完成：保存损失曲线
            if self._fig is not None and self._plt is not None:
                self._plt.ioff()
                ax = self._ax
                ax.clear()
                epochs_range = self.train_history['epoch']
                ax.plot(epochs_range, self.train_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
                ax.plot(epochs_range, self.train_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title('LSTM Training Complete - Loss Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                self._fig.tight_layout()
                fig_path = 'lstm_loss_curve.png'
                self._fig.savefig(fig_path, dpi=150)
                logger.info(f"LSTM loss curve saved: {fig_path}")
                self._plt.close(self._fig)
                self._fig = None
                self._plt = None

            # 恢复最佳模型
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
                self.model.to(self.device)

            self.model.eval()
            with torch.no_grad():
                X_test_t = X_test_t.to(self.device)
                predictions_norm = self.model(X_test_t).cpu().numpy()

                # 多目标反归一化
                n_test = len(predictions_norm)
                pred_denorm_full = np.zeros((n_test, len(self.scaler_mean)))
                pred_denorm_full[:, n_x_cols:] = predictions_norm
                pred_denorm_full = pred_denorm_full * self.scaler_std + self.scaler_mean
                predictions = pred_denorm_full[:, n_x_cols:]  # (n_test, n_targets)

                y_test_denorm_full = np.zeros((len(y_test), len(self.scaler_mean)))
                y_test_denorm_full[:, n_x_cols:] = y_test
                y_test_denorm_full = y_test_denorm_full * self.scaler_std + self.scaler_mean
                y_test_actual = y_test_denorm_full[:, n_x_cols:]

                # 总体指标（取平均）
                mse = np.mean((predictions - y_test_actual) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - y_test_actual))
                ss_res = np.sum((y_test_actual - predictions) ** 2)
                ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)

                self.metrics = {
                    'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2),
                    'n_targets': self.output_size
                }

            self.is_fitted = True

            target_names = ', '.join(self.target_cols[:3]) + ('...' if len(self.target_cols) > 3 else '')
            msg = f"✅ LSTM 训练完成！\n"
            msg += f"   模型: LSTM (hidden={hidden_size}, layers={num_layers})\n"
            msg += f"   目标列: {target_names} ({self.output_size}个)\n"
            msg += f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}\n"
            msg += f"   R² 分数: {r2:.4f}\n"
            msg += f"   RMSE: {rmse:.4f}\n"
            msg += f"   MAE: {mae:.4f}"

            logger.info(f"LSTM 训练完成: {self.metrics}")
            return True, msg

        except Exception as e:
            logger.error(f"LSTM 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"LSTM 训练失败: {str(e)}"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测下一步（单步），返回反归一化的原始尺度预测值，支持多目标输出"""
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        self.model.eval()

        original_1d = len(X.shape) == 1
        if original_1d:
            X = X.reshape(-1, 1)

        n_x_cols = X.shape[1]
        n_targets = self.output_size

        # 构建输入 (seq_len, n_features)
        if len(X) > self.seq_len:
            x_input = X[-self.seq_len:]
        else:
            x_input = X

        # 用零填充目标列，然后归一化
        data = np.column_stack([x_input, np.zeros((len(x_input), n_targets))])
        data_norm = (data - self.scaler_mean) / self.scaler_std
        X_norm = data_norm[:, :n_x_cols]

        X_seq, _ = self._create_sequences(X_norm, np.zeros((len(X_norm), n_targets)))

        if len(X_seq) == 0:
            X_seq = X_norm[-self.seq_len:].reshape(1, self.seq_len, X_norm.shape[1])

        X_t = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            preds_norm = self.model(X_t).cpu().numpy()  # (n_seq, n_targets)

        # 反归一化：重建完整数据然后提取目标列
        n_seq = len(preds_norm)
        pred_denorm_full = np.zeros((n_seq, len(self.scaler_mean)))
        pred_denorm_full[:, n_x_cols:] = preds_norm
        pred_denorm_full = pred_denorm_full * self.scaler_std + self.scaler_mean
        result = pred_denorm_full[:, n_x_cols:]  # (n_seq, n_targets)

        if original_1d and n_targets == 1:
            return result.squeeze()
        return result

    def predict_future(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        滚动预测未来 steps 步（自回归）
        X: (n_samples,) 或 (n_samples, n_features)，至少 seq_len 个样本
        返回: (steps, n_targets) 预测的未来 steps 步（原始尺度）
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        original_1d = len(X.shape) == 1
        if original_1d:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        n_targets = self.output_size
        predictions = []
        cur = list(X[-self.seq_len:].astype(np.float64))

        for _ in range(steps):
            x_input = np.array(cur[-self.seq_len:])
            preds = self.predict(x_input)

            if len(preds) == 0:
                break

            pred = preds[-1]  # 取最后一个时间步的预测 (n_targets,)
            predictions.append(pred if n_targets > 1 else float(pred))

            # 更新历史：用第一个目标的预测值更新第一列
            # 多目标时，只更新第一个目标（其他目标假设为外生变量）
            new_row = np.zeros(n_features, dtype=np.float64)
            if n_targets >= 1:
                new_row[0] = float(pred[0]) if n_targets > 1 else float(pred)
            cur.append(new_row)

        result = np.array(predictions)  # (steps, n_targets) or (steps,)
        if original_1d and n_targets == 1:
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
            'output_size': self.output_size,
            'metrics': self.metrics,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'feature_names': self.feature_names,
            'target_cols': self.target_cols,
            'train_history': self.train_history
        }, path)
        logger.info(f"LSTM 模型已保存: {path}")

    def load_model(self, path: str):
        data = torch.load(path, map_location=self.device)

        self.scaler_mean = data['scaler_mean']
        self.scaler_std = data['scaler_std']
        self.seq_len = data['seq_len']
        self.input_size = data['input_size']
        self.output_size = data.get('output_size', 1)
        self.metrics = data.get('metrics', {})
        self.feature_names = data.get('feature_names')
        self.target_cols = data.get('target_cols', [])
        self.train_history = data.get('train_history', {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [], 'train_r2': [], 'val_r2': []
        })

        hidden_size = data.get('hidden_size', 64)
        num_layers = data.get('num_layers', 2)

        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=self.output_size
        ).to(self.device)
        self.model.load_state_dict(data['model_state'])
        self.model.eval()

        self.is_fitted = True
        logger.info(f"LSTM 模型已加载: {path}")
