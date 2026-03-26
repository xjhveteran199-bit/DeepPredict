"""
自动特征工程
根据数据分析结果自动构建时序特征（lag、滚动统计、季节性编码等）
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """特征构建配置"""
    use_lag: bool = True
    use_rolling: bool = True
    use_seasonal: bool = True
    use_diff: bool = True
    lag_periods: List[int] = None  # [1, 7, 30] 等
    rolling_windows: List[int] = None  # [7, 30] 等
    target_col: str = ""

    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 7]
        if self.rolling_windows is None:
            self.rolling_windows = [7]


class FeatureEngine:
    """
    自动特征工程

    build() → Tuple[DataFrame, List[str]]  # 增强后的 DataFrame + 新增特征名列表
    """

    def __init__(self):
        self._last_features: List[str] = []

    def build(self, df: pd.DataFrame,
              target_col: str,
              date_col: Optional[str],
              seasonality: Optional[int],
              config: Optional[FeatureConfig] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        自动构建时序特征

        Args:
            df: 原始 DataFrame
            target_col: 目标列名
            date_col: 日期列名（可选）
            seasonality: 检测到的季节性周期（步数）
            config: 特征配置

        Returns:
            - 增强后的 DataFrame
            - 新增特征名列表
        """
        if config is None:
            config = FeatureConfig(target_col=target_col)

        df = df.copy()
        new_features = []

        # ===== 1. Lag 特征 =====
        if config.use_lag:
            for lag in config.lag_periods:
                col_name = f'{target_col}_lag_{lag}'
                if target_col in df.columns:
                    df[col_name] = df[target_col].shift(lag)
                    new_features.append(col_name)

        # ===== 2. 滚动统计特征 =====
        if config.use_rolling and target_col in df.columns:
            for window in config.rolling_windows:
                if window < len(df):
                    # 滚动均值
                    mean_col = f'{target_col}_roll_mean_{window}'
                    df[mean_col] = df[target_col].rolling(window=window, min_periods=1).mean()
                    new_features.append(mean_col)

                    # 滚动标准差
                    if window >= 7:
                        std_col = f'{target_col}_roll_std_{window}'
                        df[target_col].rolling(window=window, min_periods=1).std()
                        df[std_col] = df[target_col].rolling(window=window, min_periods=1).std()
                        new_features.append(std_col)

        # ===== 3. 季节性编码（正弦/余弦）=====
        if config.use_seasonal and date_col and date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(dates):
                t = np.arange(len(dates))

                if seasonality and seasonality > 0:
                    for period in [seasonality]:
                        sin_col = f'sin_{period}'
                        cos_col = f'cos_{period}'
                        df[sin_col] = np.sin(2 * np.pi * t / period)
                        df[cos_col] = np.cos(2 * np.pi * t / period)
                        new_features.extend([sin_col, cos_col])

                # 日历特征（总是添加）
                if len(dates) > 0:
                    try:
                        df['hour'] = dates.dt.hour
                        df['day_of_week'] = dates.dt.dayofweek
                        df['day_of_year'] = dates.dt.dayofyear
                        df['month'] = dates.dt.month
                        df['quarter'] = dates.dt.quarter
                        new_features.extend(['hour', 'day_of_week', 'day_of_year', 'month', 'quarter'])
                    except Exception:
                        pass

        # ===== 4. 差分特征 =====
        if config.use_diff and target_col in df.columns:
            df[f'{target_col}_diff_1'] = df[target_col].diff(1)
            new_features.append(f'{target_col}_diff_1')

            if seasonality and seasonality > 1:
                df[f'{target_col}_diff_{seasonality}'] = df[target_col].diff(seasonality)
                new_features.append(f'{target_col}_diff_{seasonality}')

        # ===== 5. 增长率特征 =====
        if config.use_diff and target_col in df.columns:
            df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1).replace([np.inf, -np.inf], np.nan)
            new_features.append(f'{target_col}_pct_change_1')

        # ===== 6. 指数移动平均 =====
        if target_col in df.columns and len(df) >= 7:
            df[f'{target_col}_ema_7'] = df[target_col].ewm(span=7, adjust=False).mean()
            new_features.append(f'{target_col}_ema_7')

        # 移除 NaN 行（lag/diff 产生的前面几行）
        # 不在这里移除，而是在训练时由模型自己处理
        # 但记录下有多少行是 NaN
        nan_info = df[new_features].isna().sum().to_dict()

        self._last_features = new_features
        logger.info(f"特征工程完成：新增 {len(new_features)} 个特征: {new_features[:5]}...")
        return df, new_features

    def get_feature_summary(self, df: pd.DataFrame, new_features: List[str]) -> str:
        """生成特征摘要（用于 UI 显示）"""
        if not new_features:
            return "未添加额外特征（数据量过小）"

        lines = [f"**已自动构建 {len(new_features)} 个特征：**"]
        for f in new_features[:10]:  # 最多显示10个
            if f in df.columns:
                n_nan = int(df[f].isna().sum())
                lines.append(f"  • `{f}`：缺失值 {n_nan} 个")
        if len(new_features) > 10:
            lines.append(f"  ... 共 {len(new_features)} 个特征")
        return "\n".join(lines)

    def suggest_config(self, seasonality: Optional[int],
                       time_unit: Optional[str]) -> FeatureConfig:
        """根据分析结果推荐特征配置"""
        config = FeatureConfig()

        # lag 周期
        config.lag_periods = [1]
        if seasonality:
            config.lag_periods.append(int(seasonality))
        if time_unit == "日":
            config.lag_periods.extend([7, 30])  # 周、月
        elif time_unit == "小时":
            config.lag_periods.extend([24, 168])  # 日、周
        config.lag_periods = sorted(set(config.lag_periods))

        # 滚动窗口
        config.rolling_windows = [7]
        if seasonality:
            config.rolling_windows.append(int(seasonality))
        config.rolling_windows = sorted(set(config.rolling_windows))

        return config
