"""
多通道信号解耦模块
将混合多模态信号分离为独立来源信号
支持:
  - FastICA: 独立成分分析，适合线性混合
  - AutoEncoder: 非线性解耦，适合复杂混合
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)


# ============================================================
# FastICA 解耦器 (基于 sklearn)
# ============================================================
class FastICADecoupler:
    """
    基于 FastICA 的线性解耦器
    将 N 通道混合信号分解为 N 个独立信号源
    适用于: 多传感器叠加信号、混合生理信号等
    """

    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.mixers_: Optional[List[np.ndarray]] = []
        self.unmixers_: List[np.ndarray] = []
        self.n_channels_: int = 0
        self.is_fitted: bool = False
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'FastICADecoupler':
        """
        拟合解耦器（无监督学习）

        Args:
            X: 混合信号 (n_samples, n_channels)
        """
        from sklearn.decomposition import FastICA

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_channels = X.shape[1]
        self.n_channels_ = n_channels

        # 归一化
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.scaler_mean) / self.scaler_std

        # 保存每个通道的 ICA 解混矩阵
        self.unmixers_ = []

        # 逐通道提取独立成分
        n_comp = self.n_components or n_channels

        try:
            ica = FastICA(n_components=n_comp, max_iter=500, random_state=42, tol=0.001)
            S_ = ica.fit_transform(X_norm)  # 独立成分
            A_ = ica.mixing_  # 混合矩阵
            self.unmixers_.append(ica.components_)  # 分离矩阵
        except Exception as e:
            logger.warning(f"FastICA failed: {e}, using PCA fallback")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_comp)
            S_ = pca.fit_transform(X_norm)
            self.unmixers_.append(pca.components_)

        self.S_ = S_
        self.is_fitted = True
        logger.info(f"FastICA fitted: {n_comp} components from {n_channels} channels")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """将混合信号转换为独立成分"""
        if not self.is_fitted:
            raise ValueError("先调用 fit() 训练解耦器")

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        X_norm = (X - self.scaler_mean) / (self.scaler_std + 1e-8)

        # 手动 ICA 变换
        S = np.dot(X_norm, self.unmixers_[0].T)
        return S

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        return self.fit(X).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """解耦后的独立信号 = predict"""
        return self.transform(X)


# ============================================================
# AutoEncoder 解耦器 (PyTorch)
# ============================================================
class AutoEncoderDecoder(nn.Module):
    """非线性自编码器解耦器"""

    def __init__(self, n_channels: int, hidden_dim: int = 64, latent_dim: int = None):
        super().__init__()
        if latent_dim is None:
            latent_dim = n_channels

        self.latent_dim = latent_dim
        self.n_channels = n_channels

        # Encoder: 混合信号 -> 隐变量 (移除 BatchNorm1d，因为输入维度就是 n_channels)
        self.encoder = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # Decoder: 隐变量 -> 各个独立信号
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class SignalAutoEncoder:
    """
    基于 AutoEncoder 的非线性信号解耦器
    学习从混合信号到独立源的映射
    """

    def __init__(self, n_channels: int, hidden_dim: int = 64, latent_dim: int = None):
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim or n_channels
        self.device = "cpu"
        self.model: Optional[AutoEncoderDecoder] = None
        self.is_fitted = False

    def _create_segments(self, data: np.ndarray, seg_len: int = 50) -> np.ndarray:
        """将连续信号切分为段"""
        segments = []
        stride = seg_len // 2
        for i in range(0, len(data) - seg_len, stride):
            segments.append(data[i:i + seg_len])
        return np.array(segments)

    def train(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001, seg_len: int = 100, **kwargs) -> Tuple[bool, str]:
        """
        训练自编码器解耦器

        Args:
            X: 多通道混合信号 (n_samples, n_channels)
            epochs: 训练轮数
            batch_size: 批次大小
            seg_len: 每次输入的信号段长度
        """
        try:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)

            n_samples, n_channels = X.shape
            self.n_channels = n_channels

            # 归一化
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-8
            X_norm = (X - self.scaler_mean) / self.scaler_std

            # 构建段
            segs = self._create_segments(X_norm, seg_len)
            if len(segs) < 50:
                return False, f"数据不足（{len(segs)} 个段），请增加数据量"

            # 展平每个段作为一个样本
            X_tensor = torch.FloatTensor(segs)

            # 模型
            self.model = AutoEncoderDecoder(
                n_channels=n_channels,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            ).to(self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.MSELoss()

            self.model.train()
            losses = []

            for epoch in range(epochs):
                indices = torch.randperm(len(X_tensor))
                epoch_loss = 0
                n_batches = 0

                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[indices[i:i + batch_size]].to(self.device)
                    recon, _ = self.model(batch)

                    loss = criterion(recon, batch)
                    # + 稀疏正则：鼓励隐变量独立
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                avg_loss = epoch_loss / max(n_batches, 1)
                losses.append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"AutoEncoder Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            self.is_fitted = True

            msg = (
                f"✅ AutoEncoder 解耦器训练完成！\n\n"
                f"   模型: AutoEncoder (hidden={self.hidden_dim}, latent={self.latent_dim})\n"
                f"   信号段: {len(segs)} 个, 段长={seg_len}\n"
                f"   最终损失: {losses[-1]:.6f}\n\n"
                f"💡 使用 encode() 获取独立信号，decode() 重构混合信号"
            )
            logger.info(f"AutoEncoder 训练完成, final_loss={losses[-1]:.6f}")
            return True, msg

        except Exception as e:
            logger.error(f"AutoEncoder 训练失败: {e}")
            import traceback; logger.error(traceback.format_exc())
            return False, f"❌ AutoEncoder 训练失败: {str(e)}"

    def encode(self, X: np.ndarray) -> np.ndarray:
        """将混合信号编码为独立隐变量"""
        if not self.is_fitted or self.model is None:
            raise ValueError("先训练模型")

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        X_norm = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        X_t = torch.FloatTensor(X_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, z = self.model(X_t)

        return z.cpu().numpy()

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """从隐变量重构混合信号"""
        if not self.is_fitted or self.model is None:
            raise ValueError("先训练模型")

        Z_t = torch.FloatTensor(Z).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model.decode(Z_t)

        return recon.cpu().numpy() * self.scaler_std + self.scaler_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        """解耦 = encode"""
        return self.encode(X)


# ============================================================
# 统一解耦器接口
# ============================================================
class SignalDecoupler:
    """
    多通道信号解耦器 - 统一接口
    支持两种模式:
      - 'ica': FastICA 线性解耦，适合弱混合场景
      - 'autoencoder': 非线性解耦，适合复杂混合
    """

    def __init__(self, method: Literal['ica', 'autoencoder'] = 'ica',
                 n_components: int = None, hidden_dim: int = 64, latent_dim: int = None):
        self.method = method
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.decoupler_: Optional = None
        self.is_fitted = False
        self.n_channels_: int = 0

    def fit(self, X: np.ndarray, **kwargs) -> 'SignalDecoupler':
        """
        训练解耦器（无监督）
        X: 混合信号 (n_samples, n_channels)
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.n_channels_ = X.shape[1]

        if self.method == 'ica':
            self.decoupler_ = FastICADecoupler(n_components=self.n_components)
            self.decoupler_.fit(X)
        elif self.method == 'autoencoder':
            self.decoupler_ = SignalAutoEncoder(
                n_channels=self.n_channels_,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            )
            self.decoupler_.train(X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """将混合信号转换为独立信号"""
        return self.decoupler_.transform(X)

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """拟合并转换"""
        self.fit(X, **kwargs)
        return self.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """解耦后的独立信号"""
        return self.transform(X)

    def get_independent_signals(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取各独立信号通道
        返回: dict，key=通道名，value=信号数组
        """
        sources = self.transform(X)
        if len(sources.shape) == 1:
            sources = sources.reshape(-1, 1)

        result = {}
        for i in range(sources.shape[1]):
            result[f'source_{i+1}'] = sources[:, i]
        return result

    def summary(self) -> str:
        """返回解耦器摘要"""
        method_names = {'ica': 'FastICA (线性)', 'autoencoder': 'AutoEncoder (非线性)'}
        return (
            f"SignalDecoupler ({method_names.get(self.method, self.method)})\n"
            f"  输入通道数: {self.n_channels_}\n"
            f"  输出独立信号: {self.n_components or self.n_channels_}\n"
            f"  已训练: {self.is_fitted}"
        )
