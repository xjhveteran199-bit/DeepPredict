"""
LSTM 深度学习时序预测模型
基于 PyTorch 实现
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
    """LSTM 预测模型"""
    
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
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
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
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
            
            self.model.eval()
            with torch.no_grad():
                X_test_t = X_test_t.to(self.device)
                predictions = self.model(X_test_t).cpu().numpy()
                
                test_data_norm = np.zeros((len(y_test), data_normalized.shape[1]))
                test_data_norm[:, -1] = predictions
                pred_denorm = test_data_norm * self.scaler_std + self.scaler_mean
                predictions = pred_denorm[:, -1]
                
                test_data_norm[:, -1] = y_test
                y_test_denorm = test_data_norm * self.scaler_std + self.scaler_mean
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
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        self.model.eval()
        
        data = np.column_stack([X, np.zeros(len(X))])
        data_normalized = (data - self.scaler_mean) / self.scaler_std
        
        X_norm = data_normalized[:, :-1]
        X_seq, _ = self._create_sequences(X_norm, np.zeros(len(X_norm)))
        
        if len(X_seq) == 0:
            raise ValueError("数据长度不足，无法创建预测序列（需要 > seq_len 条数据）")
        
        X_t = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy()
        
        pred_data = np.zeros((len(predictions), len(self.scaler_mean)))
        pred_data[:, -1] = predictions
        pred_denorm = pred_data * self.scaler_std + self.scaler_mean
        
        return pred_denorm[:, -1]
    
    def save_model(self, path: str):
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
