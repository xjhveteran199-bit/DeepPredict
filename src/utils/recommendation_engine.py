"""
推荐引擎
根据数据分析结果 + 用户回答 → 生成最优模型配置
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecommendedConfig:
    """推荐配置"""
    model: str = "GradientBoosting"  # LSTM | PatchTST | EnhancedCNN1D | GradientBoosting

    # 模型参数
    seq_len: int = 48
    pred_len: int = 24
    hidden_size: int = 64
    num_layers: int = 2
    d_model: int = 128
    n_heads: int = 4
    n_layers_trans: int = 3
    d_ff: int = 256
    patch_size: int = 16
    hidden_channels: int = 64
    dropout: float = 0.2

    # 训练参数
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001

    # 模式
    predict_mode: str = "单变量时序(推荐)"  # 单变量 | 多变量
    external_factors: List[str] = field(default_factory=list)  # 外生变量列表

    # 推荐理由
    reason: str = ""
    confidence: float = 0.5  # 推荐置信度 0~1

    # 风险提示
    risk_warnings: List[str] = field(default_factory=list)

    # 可调参数范围
    adjustable_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # {
    #   'seq_len': {'min': 12, 'max': 192, 'step': 4, 'default': 96},
    #   ...
    # }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers_trans': self.n_layers_trans,
            'd_ff': self.d_ff,
            'patch_size': self.patch_size,
            'hidden_channels': self.hidden_channels,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'predict_mode': self.predict_mode,
            'external_factors': self.external_factors,
            'reason': self.reason,
            'confidence': self.confidence,
            'risk_warnings': self.risk_warnings,
            'adjustable_params': self.adjustable_params,
        }


class RecommendationEngine:
    """
    智能推荐引擎

    recommend() → RecommendedConfig
    """

    # 模型选择决策表
    MODEL_THRESHOLDS = {
        # n_samples: (推荐模型, 备选)
        (0, 100): ("GradientBoosting", "LSTM"),
        (100, 500): ("LSTM", "GradientBoosting"),
        (500, float('inf')): ("PatchTST", "LSTM"),
    }

    # 模型特点（用于生成推荐理由）
    MODEL_PROFILES = {
        'LSTM': {
            'strength': '擅长捕捉时序中的中短期依赖关系',
            'weakness': '对长周期季节性捕捉较弱',
            'speed': '中等',
        },
        'PatchTST': {
            'strength': 'Transformer 架构，能捕捉长周期季节性依赖',
            'weakness': '需要较多数据（>500），训练较慢',
            'speed': '较慢',
        },
        'EnhancedCNN1D': {
            'strength': '多尺度卷积捕捉不同时间尺度的模式',
            'weakness': '对长距离依赖捕捉不如 Transformer',
            'speed': '快',
        },
        'GradientBoosting': {
            'strength': '鲁棒性强，训练快，不易过拟合',
            'weakness': '无法捕捉时序动态依赖',
            'speed': '最快',
        },
    }

    def recommend(self, analysis: 'AnalysisResult',
                   user_answers: Dict[str, Any]) -> RecommendedConfig:
        """
        根据数据分析和用户回答生成推荐配置

        Args:
            analysis: DataAnalyzer 的分析结果
            user_answers: 用户问卷答案
                - pred_len: 预测步数（用户选择）
                - external_factors: 外生变量列表
                - priority: "accuracy" | "stability" | "balanced"
                - need_explain: bool
        """
        cfg = RecommendedConfig()

        n = analysis.n_samples
        seasonality = analysis.detected_seasonality

        # ===== 1. 确定预测模式 =====
        external_factors = user_answers.get('external_factors', [])
        has_external = bool(external_factors and external_factors != ['无外部因素（纯自变量）'])
        cfg.predict_mode = "多变量" if has_external else "单变量时序(推荐)"
        cfg.external_factors = external_factors if has_external else []

        # ===== 2. 计算 pred_len（用户选择 vs 数据限制）====
        user_pred = user_answers.get('pred_len', 30)
        max_pred = min(
            max(3, analysis.suggested_pred_len * 2),
            n // 5
        )

        if user_pred > max_pred:
            cfg.risk_warnings.append(
                f"预测步数 {user_pred} 较大，长期预测误差会累积。"
                f"建议控制在 {max_pred} 步以内，或减少 seq_len 以提高精度。"
            )
            cfg.pred_len = int(user_pred)  # 仍然尊重用户选择，但给出警告
        else:
            cfg.pred_len = int(user_pred)

        # ===== 3. 计算 seq_len =====
        min_seq = max(cfg.pred_len + 5, 12)
        max_seq = min(max(48, n // 3), 512)

        if seasonality:
            cfg.seq_len = int(min(max(seasonality, cfg.pred_len * 2), max_seq))
        else:
            cfg.seq_len = int(min(max(analysis.suggested_seq_len, cfg.pred_len * 2), max_seq))

        cfg.seq_len = max(min_seq, cfg.seq_len)

        # ===== 4. 选择模型 =====
        priority = user_answers.get('priority', 'accuracy')

        # 决策逻辑
        if n < 100:
            cfg.model = 'GradientBoosting'
            cfg.reason = f"数据量较小（{n}条），使用轻量模型避免过拟合"
            cfg.confidence = 0.8

        elif has_external:
            # 有外生变量：优先使用能处理多变量的模型
            if n > 500:
                cfg.model = 'PatchTST'
                cfg.reason = "检测到外部驱动因素，PatchTST 能同时建模内生+外生变量"
                cfg.confidence = 0.75
            else:
                cfg.model = 'LSTM'
                cfg.reason = "数据量中等，使用 LSTM 处理多变量时序"
                cfg.confidence = 0.7

        elif seasonality and n > 500:
            # 有季节性 + 数据量大
            if priority == 'accuracy':
                cfg.model = 'PatchTST'
                cfg.reason = f"检测到{seasonality}步季节性周期，Transformer 能有效捕捉长周期依赖"
                cfg.confidence = 0.8
            else:
                cfg.model = 'EnhancedCNN1D'
                cfg.reason = "多尺度 CNN 兼顾季节性模式 + 稳定性"
                cfg.confidence = 0.7

        elif n > 500:
            if priority == 'accuracy':
                cfg.model = 'LSTM'
                cfg.reason = "数据量充足，LSTM 能捕捉复杂时序动态"
                cfg.confidence = 0.75
            else:
                cfg.model = 'GradientBoosting'
                cfg.reason = "稳定性优先，GBDT 鲁棒性强，不易过拟合"
                cfg.confidence = 0.8

        else:
            cfg.model = 'GradientBoosting'
            cfg.reason = "数据量有限，GBDT 是最稳妥的选择"
            cfg.confidence = 0.7

        # ===== 5. 设置模型参数 =====
        self._set_model_params(cfg, n, seasonality)

        # ===== 6. 设置可调参数范围 =====
        cfg.adjustable_params = self._get_adjustable_params(cfg, n)

        # ===== 7. 数据警告 =====
        if analysis.warnings:
            cfg.risk_warnings.extend(analysis.warnings[:3])

        if cfg.model == 'PatchTST' and n < 500:
            cfg.risk_warnings.append("PatchTST 在小数据量（<500）上可能效果不佳，考虑使用 LSTM 或 GradientBoosting")

        if cfg.seq_len > n // 2:
            cfg.risk_warnings.append(f"seq_len={cfg.seq_len} 较大，数据量可能不足以支撑，建议减少 seq_len")

        logger.info(f"推荐完成: model={cfg.model}, seq_len={cfg.seq_len}, pred_len={cfg.pred_len}, reason={cfg.reason}")
        return cfg

    def _set_model_params(self, cfg: RecommendedConfig, n: int, seasonality: Optional[int]):
        """根据模型类型和数据规模设置具体参数"""
        if cfg.model == 'LSTM':
            cfg.hidden_size = 32 if n < 200 else (64 if n < 1000 else 128)
            cfg.num_layers = 1 if n < 200 else 2
            cfg.epochs = min(20, max(10, n // 50))
            cfg.batch_size = max(8, min(32, n // 20))
            cfg.learning_rate = 0.005 if n < 200 else 0.001
            cfg.dropout = 0.3 if n < 200 else 0.2

        elif cfg.model == 'PatchTST':
            cfg.d_model = 32 if n < 500 else (64 if n < 2000 else 128)
            cfg.n_heads = 2 if n < 500 else 4
            cfg.n_layers_trans = 1 if n < 500 else (2 if n < 2000 else 3)
            cfg.d_ff = 64 if n < 500 else (128 if n < 2000 else 256)
            cfg.patch_size = 4 if seasonality and seasonality < 20 else (8 if seasonality and seasonality < 50 else 16)
            cfg.epochs = min(30, max(10, n // 100))
            cfg.batch_size = max(8, min(32, n // 50))
            cfg.learning_rate = 0.005 if n < 500 else 0.001
            cfg.dropout = 0.25 if n < 500 else 0.15

        elif cfg.model == 'EnhancedCNN1D':
            cfg.hidden_channels = 32 if n < 200 else (64 if n < 1000 else 128)
            cfg.hidden_channels = min(cfg.hidden_channels, 64)  # 保守设置
            cfg.epochs = min(30, max(10, n // 50))
            cfg.batch_size = max(8, min(32, n // 20))
            cfg.learning_rate = 0.005 if n < 200 else 0.001
            cfg.dropout = 0.25

        elif cfg.model == 'GradientBoosting':
            cfg.epochs = min(50, max(20, n // 20))
            cfg.batch_size = 32

    def _get_adjustable_params(self, cfg: RecommendedConfig, n: int) -> Dict[str, Dict[str, Any]]:
        """生成可调参数范围"""
        params = {
            'seq_len': {
                'min': 12,
                'max': min(256, n // 2),
                'step': 4,
                'default': cfg.seq_len,
                'label': '输入窗口(seq_len)'
            },
            'pred_len': {
                'min': 3,
                'max': min(cfg.seq_len, n // 3),
                'step': 1,
                'default': cfg.pred_len,
                'label': '预测步数(pred_len)'
            },
        }

        if cfg.model in ('LSTM', 'PatchTST', 'EnhancedCNN1D'):
            params['epochs'] = {
                'min': 5,
                'max': min(100, n // 10),
                'step': 5,
                'default': cfg.epochs,
                'label': '训练轮数(epochs)'
            }
            params['learning_rate'] = {
                'min': 0.0001,
                'max': 0.01,
                'step': 0.0005,
                'default': cfg.learning_rate,
                'label': '学习率'
            }
            params['dropout'] = {
                'min': 0.0,
                'max': 0.5,
                'step': 0.05,
                'default': cfg.dropout,
                'label': 'Dropout率'
            }

        if cfg.model == 'LSTM':
            params['hidden_size'] = {
                'min': 16,
                'max': 256,
                'step': 16,
                'default': cfg.hidden_size,
                'label': 'LSTM隐藏层大小'
            }
            params['num_layers'] = {
                'min': 1,
                'max': 4,
                'step': 1,
                'default': cfg.num_layers,
                'label': 'LSTM层数'
            }

        if cfg.model == 'PatchTST':
            params['d_model'] = {
                'min': 32,
                'max': 256,
                'step': 32,
                'default': cfg.d_model,
                'label': 'Transformer维度'
            }
            params['n_heads'] = {
                'min': 2,
                'max': 8,
                'step': 2,
                'default': cfg.n_heads,
                'label': '注意力头数'
            }

        return params
