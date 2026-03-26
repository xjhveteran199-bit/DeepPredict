"""
数据分析引擎
对解析后的 DataFrame 进行全面分析，为推荐引擎提供数据特征
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """数据分析结果"""
    # 基本信息
    n_samples: int = 0
    n_features: int = 0
    n_numeric_cols: int = 0
    date_col: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None  # (start, end)
    time_step_unit: Optional[str] = None  # 日/小时/分钟/月

    # 数据质量
    missing_summary: Dict[str, int] = field(default_factory=dict)  # col -> n_missing
    duplicate_rows: int = 0

    # 季节性检测
    detected_seasonality: Optional[int] = None  # 周期长度（天/步），None=未检测到
    seasonality_confidence: float = 0.0  # 0~1
    seasonality_label: Optional[str] = None  # "年度(365天)" / "周(7天)" / "月度(30天)" / None

    # 数值列统计
    column_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {
    #   'Temp': {'mean': 15.2, 'std': 8.3, 'min': -5, 'max': 40, 'cv': 0.55},
    #   ...
    # }

    # 相关性矩阵（只包含数值列）
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 推荐
    suggested_target: Optional[str] = None  # 最适合做目标的列
    suggested_target_reason: Optional[str] = None
    suggested_seq_len: int = 48  # 推荐的输入窗口
    suggested_pred_len: int = 24  # 推荐的预测步数
    suggested_features: List[str] = field(default_factory=list)  # 推荐构建的特征名

    # 警告
    warnings: List[str] = field(default_factory=list)
    # ["数据量较小（<100），建议使用轻量模型", "所有数值列相关性>0.9，可能存在共线性"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_numeric_cols': self.n_numeric_cols,
            'date_col': self.date_col,
            'date_range': self.date_range,
            'time_step_unit': self.time_step_unit,
            'missing_summary': self.missing_summary,
            'duplicate_rows': self.duplicate_rows,
            'detected_seasonality': self.detected_seasonality,
            'seasonality_confidence': self.seasonality_confidence,
            'seasonality_label': self.seasonality_label,
            'column_stats': self.column_stats,
            'correlation_matrix': self.correlation_matrix,
            'suggested_target': self.suggested_target,
            'suggested_target_reason': self.suggested_target_reason,
            'suggested_seq_len': self.suggested_seq_len,
            'suggested_pred_len': self.suggested_pred_len,
            'suggested_features': self.suggested_features,
            'warnings': self.warnings,
        }


class DataAnalyzer:
    """数据分析器"""

    # 常见季节性周期
    KNOWN_SEASONALITIES = {
        # 步长周期（按天数近似）
        7: ("周", 0.7),       # 周周期
        12: ("月", 0.6),      # 月周期（约30天，取近似）
        24: ("日/小时", 0.5), # 日周期（小时数据）
        52: ("周(年度)", 0.65),  # 年度周周期
        365: ("年度", 0.75),  # 年度周期
        168: ("周(小时)", 0.5), # 周周期（小时数据）
    }

    def analyze(self, df: pd.DataFrame, date_col: Optional[str] = None) -> AnalysisResult:
        """
        对 DataFrame 进行全面分析

        Args:
            df: 解析后的 DataFrame
            date_col: 日期列名（可选）
        """
        result = AnalysisResult()

        # ===== 基本信息 =====
        result.n_samples = len(df)
        result.n_features = len(df.columns)
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        result.n_numeric_cols = len(numeric_cols)

        # ===== 日期列处理 =====
        if date_col and date_col in df.columns:
            result.date_col = date_col
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                dates = df[date_col].dropna()
                if len(dates) > 1:
                    result.date_range = (str(dates.iloc[0].date()), str(dates.iloc[-1].date()))
                    # 推断时间步单位
                    total_span = (dates.iloc[-1] - dates.iloc[0]).total_seconds()
                    n = len(dates)
                    avg_delta = total_span / max(n - 1, 1)
                    if avg_delta >= 86400 * 25:  # > 25天
                        result.time_step_unit = "月"
                    elif avg_delta >= 3600 * 20:  # > 20小时
                        result.time_step_unit = "日"
                    elif avg_delta >= 60 * 30:  # > 30分钟
                        result.time_step_unit = "小时"
                    else:
                        result.time_step_unit = "分钟"

        # ===== 数据质量 =====
        result.missing_summary = {col: int(df[col].isna().sum())
                                   for col in df.columns
                                   if df[col].isna().sum() > 0}
        result.duplicate_rows = int(df.duplicated().sum())

        if result.n_samples < 50:
            result.warnings.append(f"数据量很小（{result.n_samples}条），建议使用轻量模型和短序列")
        elif result.n_samples < 200:
            result.warnings.append(f"数据量偏小（{result.n_samples}条），建议使用 GradientBoosting")

        # ===== 数值列统计 =====
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            mean = float(series.mean())
            std = float(series.std()) + 1e-8
            result.column_stats[col] = {
                'mean': mean,
                'std': std,
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'cv': float(std / abs(mean)) if mean != 0 else 0.0,  # 变异系数
            }

        # ===== 相关性矩阵 =====
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            for c1 in numeric_cols:
                result.correlation_matrix[c1] = {}
                for c2 in numeric_cols:
                    result.correlation_matrix[c1][c2] = float(corr.loc[c1, c2])

            # 检测高相关性列
            high_corr_pairs = []
            for i, c1 in enumerate(numeric_cols):
                for c2 in numeric_cols[i+1:]:
                    if abs(corr.loc[c1, c2]) > 0.95:
                        high_corr_pairs.append((c1, c2, corr.loc[c1, c2]))
            if high_corr_pairs:
                result.warnings.append(f"发现高相关列（>0.95）：{[f'{c1}~{c2}({r:.2f})' for c1,c2,r in high_corr_pairs[:3]]}")

        # ===== 季节性检测 =====
        if result.date_col and pd.api.types.is_datetime64_any_dtype(df[result.date_col]):
            seasonality, confidence, label = self._detect_seasonality(df, result.date_col)
            result.detected_seasonality = seasonality
            result.seasonality_confidence = confidence
            result.seasonality_label = label

        # ===== 推荐目标列 =====
        suggested_target, reason = self._suggest_target(df, numeric_cols, result.missing_summary)
        result.suggested_target = suggested_target
        result.suggested_target_reason = reason

        # ===== 推荐序列长度 =====
        result.suggested_seq_len, result.suggested_pred_len = self._suggest_window(
            result.n_samples, result.detected_seasonality, result.time_step_unit
        )

        # ===== 推荐特征 =====
        result.suggested_features = self._suggest_features(
            result.detected_seasonality, result.time_step_unit
        )

        logger.info(f"数据分析完成: {result.n_samples}样本, seasonality={result.detected_seasonality}, target={result.suggested_target}")
        return result

    def _detect_seasonality(self, df: pd.DataFrame, date_col: str) -> Tuple[Optional[int], float, Optional[str]]:
        """
        通过 FFT 检测主周期

        返回: (period, confidence, label)
        """
        try:
            dates = pd.to_datetime(df[date_col]).dropna()
            if len(dates) < 2:
                return None, 0.0, None

            # 按时间排序
            values = df.loc[dates.index, df.select_dtypes(include=[np.number]).columns[0]] if df.select_dtypes(include=[np.number]).columns.any() else None
            if values is None or len(values) < 10:
                return None, 0.0, None

            values = values.dropna().values
            if len(values) < 10:
                return None, 0.0, None

            # 时间间隔分析
            if len(dates) > 1:
                deltas = dates.diff().dropna().dt.total_seconds()
                median_delta = deltas.median()

                if median_delta >= 86400 * 20:  # 日以上
                    # 对第一列数值做 FFT
                    signal = values - np.mean(values)
                    n = len(signal)
                    fft_vals = np.fft.rfft(signal)
                    power = np.abs(fft_vals)
                    freqs = np.fft.rfftfreq(n, d=1)

                    # 找主峰（排除直流分量）
                    if len(power) > 1:
                        peak_idx = np.argmax(power[1:]) + 1
                        period = int(round(n / max(freqs[peak_idx], 0.001)))
                        if 2 <= period <= n // 2:
                            confidence = float(power[peak_idx] / (power.sum() + 1e-8))
                            for known, (label, conf) in self.KNOWN_SEASONALITIES.items():
                                if abs(period - known) <= known * 0.15:  # 15%容差
                                    return known, max(confidence, conf), label
                            return period, confidence, f"未知({period}步)"

        except Exception as e:
            logger.warning(f"季节性检测失败: {e}")

        return None, 0.0, None

    def _suggest_target(self, df: pd.DataFrame, numeric_cols: List[str],
                        missing: Dict[str, int]) -> Tuple[Optional[str], Optional[str]]:
        """推荐最适合做预测目标的列"""
        if not numeric_cols:
            return None, None

        candidates = []
        for col in numeric_cols:
            n_miss = missing.get(col, 0)
            miss_rate = n_miss / max(len(df), 1)
            if miss_rate > 0.5:
                continue  # 缺失太多不推荐

            stats = df[col].describe()
            cv = float(stats['std'] / max(abs(stats['mean']), 1e-8)) if stats['mean'] != 0 else 0

            # 目标偏好：有一定变异但不是噪声
            score = cv * (1 - miss_rate)
            candidates.append((col, score, cv, miss_rate))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_col, best_score, best_cv, best_miss = candidates[0]

        if best_cv < 0.01:
            reason = "方差接近0，可能为常数列"
        elif best_cv > 5:
            reason = "变异系数很大，可能为噪声数据"
        elif best_miss > 0.1:
            reason = f"有 {best_miss*100:.1f}% 缺失，已做填充"
        else:
            reason = "方差适中，适合作为预测目标"

        return best_col, reason

    def _suggest_window(self, n_samples: int, seasonality: Optional[int],
                        time_unit: Optional[str]) -> Tuple[int, int]:
        """推荐 seq_len 和 pred_len"""
        if n_samples < 20:
            return max(3, n_samples // 4), max(1, n_samples // 8)

        # seq_len 基础值：数据量的 10%~20%，最大 256
        base_seq = min(max(12, n_samples // 10), 256)

        # 如果检测到季节性，seq_len 至少覆盖一个完整周期的一半
        if seasonality and seasonality > 0:
            base_seq = max(base_seq, seasonality // 2)

        # pred_len：seq_len 的 1/3 ~ 1/2
        base_pred = max(3, base_seq // 3)

        # 限制 pred_len 不超过数据量的 10%
        max_pred = max(3, n_samples // 10)
        base_pred = min(base_pred, max_pred)

        return int(base_seq), int(base_pred)

    def _suggest_features(self, seasonality: Optional[int],
                          time_unit: Optional[str]) -> List[str]:
        """推荐要构建的额外特征"""
        features = []

        # lag 特征（基础）
        features.extend(['lag_1', 'lag_7'])

        # 季节性 lag
        if seasonality:
            features.append(f'lag_{seasonality}')

        # 滚动统计
        features.extend(['rolling_mean_7', 'rolling_std_7'])
        if seasonality:
            features.append(f'rolling_mean_{seasonality}')

        # 差分（去除趋势）
        features.append('diff_1')

        return list(set(features))  # 去重
