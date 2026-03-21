"""
PatchTST - 时序预测Transformer模型
基于 "A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers" (ICLR 2023)

核心特点：
- Patch化：类似ViT，将时序切成固定长度片段
- Transformer编码器：全局注意力机制
- RevIN：可逆实例归一化，解决分布偏移
- 通道独立：每个特征独立处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Dict, Optional
import joblib

logger = logging.getLogger(__name__)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    论文: "Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift"
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.norm = nn.Parameter(torch.ones(num_features))
        self.denom = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        x: (batch, seq_len, n_vars)
        """
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _get_statistics(self, x: torch.Tensor):
        dim = list(range(1, x.ndim - 1)) if x.ndim > 2 else [1]
        self.mean = x.mean(dim=dim, keepdim=True)
        self.std = x.std(dim=dim, keepdim=True).add(self.eps)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.std
        return x * self.norm + self.bias
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.bias
        x = x / (self.norm + self.eps)
        return x * self.std + self.mean


class FlattenHead(nn.Module):
    """将序列展平后做回归预测"""
    def __init__(self, n_vars: int, nf: int, target_window: int):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(n_vars * nf, target_window)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_vars, seq_len, nf)
        x = x.permute(0, 1, 3, 2)  # (batch, n_vars, nf, seq_len)
        x = self.flatten(x)  # (batch, n_vars * nf * seq_len)
        return self.linear(x)


class PatchTSTBlock(nn.Module):
    """单层 Transformer 编码器块"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm 自注意力
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class PatchTST(nn.Module):
    """
    PatchTST 模型主体
    - Patch化：Conv1D 将时序划分为固定长度片段
    - Projection：线性投影到 d_model 维度
    - Transformer 编码器
    - RevIN 归一化
    """
    
    def __init__(
        self,
        c_in: int,           # 输入通道数（特征数）
        c_out: int,          # 输出通道数（目标变量数）
        seq_len: int = 96,   # 输入序列长度
        pred_len: int = 96,  # 预测序列长度
        patch_size: int = 16,# 每个patch的长度
        d_model: int = 128,  # 模型维度
        n_heads: int = 4,    # 注意力头数
        n_layers: int = 3,   # Transformer层数
        d_ff: int = 256,     # FFN维度
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.d_model = d_model
        self.c_in = c_in
        self.c_out = c_out
        
        # 计算patch数量
        self.n_patches = (seq_len // patch_size)
        
        # RevIN 归一化
        self.use_revin = use_revin
        self.revin = RevIN(c_in) if use_revin else None
        
        # Patch化层：Conv1D
        # 输入: (batch, c_in, seq_len) -> 输出: (batch, d_model, n_patches)
        self.patch_proj = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 可学习的 CLS token（类似ViT，但不是用于分类而是用于预测）
        # 添加一个可学习的 token 放在序列最前面
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 可学习位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer 编码器层
        self.blocks = nn.ModuleList([
            PatchTSTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # LayerNorm
        self.ln = nn.LayerNorm(d_model)
        
        # 预测头：将每个patch的表示映射到预测值
        # 方式：用最后一个patch的表示预测所有未来步
        self.head = nn.Linear(d_model, pred_len)
        
        # 另一种头：利用所有patch的表示
        self.head_fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, pred_len)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, c_in)  原始时序输入
        返回: (batch, pred_len, c_out)  预测输出
        """
        batch_size = x.size(0)
        
        # RevIN 归一化
        if self.use_revin and self.revin is not None:
            x = self.revin(x, 'norm')
        
        # 转通道维度: (batch, seq_len, c_in) -> (batch, c_in, seq_len)
        x = x.permute(0, 2, 1)
        
        # Patch化: (batch, c_in, seq_len) -> (batch, d_model, n_patches)
        x = self.patch_proj(x)
        
        # 转维度: (batch, d_model, n_patches) -> (batch, n_patches, d_model)
        x = x.permute(0, 2, 1)
        
        # 添加 CLS token: (batch, n_patches, d_model) -> (batch, n_patches+1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.dropout(self.pos_emb)
        
        # 通过 Transformer 编码器
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        
        # 取 CLS token 的表示来预测
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # 预测: (batch, pred_len)
        pred = self.head_fc(cls_output)
        
        # 扩展到 c_out 个目标变量
        pred = pred.unsqueeze(-1).expand(-1, -1, self.c_out)  # (batch, pred_len, c_out)
        
        # RevIN 反归一化
        if self.use_revin and self.revin is not None:
            pred = self.revin(pred, 'denorm')
        
        return pred


class PatchTSTPredictor:
    """
    PatchTST 时序预测器封装
    支持训练、预测、保存、加载
    """
    
    def __init__(self):
        self.model: Optional[PatchTST] = None
        self.device: str = "cpu"
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.seq_len: int = 96
        self.pred_len: int = 96
        self.patch_size: int = 16
        self.n_features: int = 1
        self.target_col: str = ""
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self._target_mean: Optional[np.ndarray] = None
        self._target_std: Optional[np.ndarray] = None
    
    def _normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """标准化数据"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std, mean, std
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 96,
        patch_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.0005,
        test_size: float = 0.2,
        target_col: str = "",
        **kwargs
    ) -> Tuple[bool, str]:
        """
        训练 PatchTST 模型
        
        Args:
            X: 特征数据 (n_samples, n_features)
            y: 目标数据 (n_samples,) 或 (n_samples, n_targets)
            seq_len: 输入序列长度（窗口大小）
            pred_len: 预测序列长度
            patch_size: 每个patch的长度
            d_model: Transformer模型维度
            n_heads: 注意力头数
            n_layers: Transformer层数
            d_ff: FFN隐藏层维度
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            test_size: 测试集比例
        """
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            self.target_col = target_col
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.patch_size = patch_size
            
            # 处理数据维度
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            n_samples = min(len(X), len(y))
            X = X[:n_samples]
            y = y[:n_samples]
            
            n_features = X.shape[1]
            n_targets = y.shape[1] if len(y.shape) > 1 else 1
            self.n_features = n_features
            
            # 归一化
            X_norm, self.scaler_mean, self.scaler_std = self._normalize(X)
            y_norm, self._target_mean, self._target_std = self._normalize(y)
            
            # 构建训练数据：滑动窗口
            X_seqs, y_seqs = [], []
            for i in range(seq_len, n_samples - pred_len):
                X_seqs.append(X_norm[i - seq_len:i])
                y_seqs.append(y_norm[i:i + pred_len])
            
            if len(X_seqs) < 50:
                return False, f"数据不足：只有{len(X_seqs)}个训练样本，需要至少50个。请增加数据量或减少seq_len/pred_len。"
            
            X_tensor = torch.FloatTensor(np.array(X_seqs))
            y_tensor = torch.FloatTensor(np.array(y_seqs))
            
            # 划分训练/测试
            split_idx = int(len(X_tensor) * (1 - test_size))
            if split_idx < 10:
                return False, "训练数据不足，请减少test_size或增加数据量"
            
            X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
            
            # 自动调整 patch_size 使其整除 seq_len
            valid_patch_sizes = [p for p in [2, 4, 8, 16, 24, 32, 48, 64] if seq_len % p == 0]
            if patch_size not in valid_patch_sizes:
                patch_size = valid_patch_sizes[-1] if valid_patch_sizes else 4
                self.patch_size = patch_size
            
            # 创建模型
            self.model = PatchTST(
                c_in=n_features,
                c_out=n_targets,
                seq_len=seq_len,
                pred_len=pred_len,
                patch_size=patch_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                use_revin=True
            ).to(self.device)
            
            # 优化器和学习率调度器
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.MSELoss()
            
            # 训练循环
            self.model.train()
            train_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                n_batches = 0
                
                # Mini-batch训练
                indices = torch.randperm(len(X_train))
                for i in range(0, len(X_train), batch_size):
                    batch_idx = indices[i:i + batch_size]
                    X_batch = X_train[batch_idx].to(self.device)
                    y_batch = y_train[batch_idx].to(self.device)
                    
                    optimizer.zero_grad()
                    pred = self.model(X_batch)  # (batch, pred_len, n_targets)
                    
                    # 计算损失
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                
                scheduler.step()
                avg_loss = epoch_loss / max(n_batches, 1)
                train_losses.append(avg_loss)
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"PatchTST Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # 评估
            self.model.eval()
            with torch.no_grad():
                test_pred = self.model(X_test.to(self.device))
                
                # 反归一化
                test_pred_np = test_pred.cpu().numpy()
                test_pred_denorm = test_pred_np * self._target_std + self._target_mean
                y_test_denorm = y_test.numpy() * self._target_std + self._target_mean
                
                # 计算指标
                y_true = y_test_denorm.flatten()
                y_pred = test_pred_denorm.flatten()
                
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                self.metrics = {
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'R2': float(r2),
                    'model': 'PatchTST',
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'n_layers': n_layers,
                    'seq_len': seq_len,
                    'pred_len': pred_len,
                    'patch_size': patch_size,
                    'epochs': epochs,
                    'final_train_loss': float(train_losses[-1])
                }
            
            self.is_fitted = True
            
            msg = (
                f"✅ PatchTST 训练完成！\n\n"
                f"   模型: PatchTST (Transformer)\n"
                f"   配置: d_model={d_model}, heads={n_heads}, layers={n_layers}\n"
                f"   窗口: seq_len={seq_len}, pred_len={pred_len}, patch={patch_size}\n"
                f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}\n\n"
                f"   **R²** 分数: {r2:.4f}\n"
                f"   **RMSE**: {rmse:.4f}\n"
                f"   **MAE**: {mae:.4f}\n\n"
                f"💡 比 LSTM 更强的全局注意力机制，适合长序列预测"
            )
            
            logger.info(f"PatchTST 训练完成: R2={r2:.4f}, RMSE={rmse:.4f}")
            return True, msg
            
        except Exception as e:
            logger.error(f"PatchTST 训练失败: {e}")
            return False, f"❌ 训练失败: {str(e)}"
    
    def predict(self, X: np.ndarray, pred_len: int = None) -> np.ndarray:
        """
        预测未来序列
        X: (n_samples, n_features) 至少需要 seq_len 个样本
        返回: (pred_len,) 预测的未来序列
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_needed = self.seq_len
        if len(X) < n_needed:
            raise ValueError(f"需要至少 {n_needed} 个样本才能预测，当前只有 {len(X)} 个")
        
        # 取最后 seq_len 个样本
        x_input = X[-n_needed:]
        
        # 归一化
        x_norm = (x_input - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # 转Tensor
        x_tensor = torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)
        
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x_tensor)  # (1, pred_len, n_targets)
        
        pred_np = pred_norm.cpu().numpy()[0]  # (pred_len, n_targets)
        
        # 反归一化
        if len(pred_np.shape) == 1:
            pred_np = pred_np.reshape(-1, 1)
        pred_denorm = pred_np * self._target_std + self._target_mean
        
        return pred_denorm.flatten() if pred_denorm.shape[1] == 1 else pred_denorm
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Transformer不直接支持特征重要性，返回空"""
        return {}
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        torch.save({
            'model_state': self.model.state_dict(),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'patch_size': self.patch_size,
            'n_features': self.n_features,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'target_mean': self._target_mean,
            'target_std': self._target_std,
            'metrics': self.metrics,
            'target_col': self.target_col
        }, path)
        logger.info(f"PatchTST 模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        data = torch.load(path, map_location=self.device)
        
        self.seq_len = data['seq_len']
        self.pred_len = data['pred_len']
        self.patch_size = data['patch_size']
        self.n_features = data['n_features']
        self.scaler_mean = data['scaler_mean']
        self.scaler_std = data['scaler_std']
        self._target_mean = data['target_mean']
        self._target_std = data['target_std']
        self.metrics = data.get('metrics', {})
        self.target_col = data.get('target_col', '')
        
        # 重建模型
        self.model = PatchTST(
            c_in=self.n_features,
            c_out=1,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            patch_size=self.patch_size,
            d_model=self.metrics.get('d_model', 128),
            n_heads=self.metrics.get('n_heads', 4),
            n_layers=self.metrics.get('n_layers', 3),
            use_revin=True
        ).to(self.device)
        self.model.load_state_dict(data['model_state'])
        self.model.eval()
        
        self.is_fitted = True
        logger.info(f"PatchTST 模型已加载: {path}")
