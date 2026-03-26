"""
数据解耦器 - DataDecoupler
自动识别 CSV 中各列类型，分别预处理后再合并输入模型

支持类型：
  - date: 日期/时间列（转为数值 ordinal 或 unix timestamp）
  - numeric: 数值列（标准化）
  - categorical: 类别列（LabelEncoder / OneHot）
  - text: 文本列（TF-IDF 向量化）

用法：
  decoupler = DataDecoupler()
  decoupler.fit(df, target_col='K')
  X, feature_names = decoupler.transform(df)
  decoupler.inverse_transform(X_forecast)  # 反变换回原始尺度
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """单列分析结果"""
    name: str
    detected_type: str  # date | numeric | categorical | text | unknown
    is_target: bool = False
    nullable: bool = False
    n_unique: int = 0
    n_missing: int = 0
    missing_pct: float = 0.0
    sample_values: List[Any] = field(default_factory=list)
    date_format: str = ""  # 检测到的日期格式
    encoding_method: str = "label"  # label | onehot | ordinal
    stats: Dict[str, float] = field(default_factory=dict)  # mean, std, min, max...


class DataDecoupler:
    """
    数据解耦主类
    fit()  : 分析数据结构，识别每列类型，建立预处理管道
    transform(): 对数据进行解耦处理，返回干净的特征矩阵
    inverse_transform(): 将预测结果反变换回原始尺度
    """

    DATE_PATTERNS = [
        # (regex, format_str, parse_func_name)
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d', 'date'),
        (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d', 'date'),
        (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S', 'datetime'),
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '%Y-%m-%dT%H:%M:%S', 'datetime'),
        (r'^\d{4}年\d{1,2}月\d{1,2}日$', '%Y年%m月%d日', 'date'),
        (r'^\d{4}年\d{1,2}月$', '%Y年%m月', 'date'),
    ]

    DATE_COLUMN_NAMES = [
        'date', 'time', 'datetime', 'timestamp', '时间', '日期', '年', '月', '日',
        'year', 'month', 'day', 'hour', 'minute', 'second',
        'weekday', 'week', 'quarter',
    ]

    def __init__(self):
        self.profiles: Dict[str, ColumnProfile] = {}
        self.target_col: str = ""
        self.feature_names: List[str] = []
        self._fitted: bool = False

        # 变换器
        self._label_encoders: Dict[str, Any] = {}
        self._onehot_cols: List[str] = []
        self._numeric_scalers: Dict[str, Any] = {}  # name -> (mean, std)
        self._date_parsers: Dict[str, Any] = {}

        # 目标列的统计（用于反变换）
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._target_scaler: Optional[Any] = None

    def fit(self, df: pd.DataFrame, target_col: str, config: Optional[Dict] = None) -> "DataDecoupler":
        """
        分析数据，解耦各列，建立预处理管道
        """
        config = config or {}
        self.target_col = target_col
        self.profiles = {}
        self._fitted = True

        df = df.copy()
        n_rows = len(df)

        # ---- 1. 逐列检测类型 ----
        for col in df.columns:
            profile = self._analyze_column(df[col], col, is_target=(col == target_col))
            self.profiles[col] = profile

            # 强制指定类型（用户配置优先）
            if config.get('force_types'):
                forced = config['force_types'].get(col, None)
                if forced:
                    profile.detected_type = forced

        # ---- 2. 对每种类型建立变换器 ----
        self._setup_transforms(df, config)

        logger.info(f"DataDecoupler fit 完成：{len(self.profiles)} 列，类型分布: "
                     f"{self._type_summary()}")
        return self

    def _analyze_column(self, series: pd.Series, name: str, is_target: bool) -> ColumnProfile:
        """分析单列的数据特征"""
        n_rows = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique())
        missing_pct = n_missing / max(n_rows, 1)

        sample = series.dropna().head(5).tolist()
        profile = ColumnProfile(
            name=name,
            detected_type="unknown",
            is_target=is_target,
            nullable=n_missing > 0,
            n_unique=n_unique,
            n_missing=n_missing,
            missing_pct=missing_pct,
            sample_values=sample,
        )

        # 空列
        if n_unique == 0:
            profile.detected_type = "empty"
            return profile

        # 目标列用数值类型
        if is_target:
            profile.detected_type = "numeric"
            if pd.api.types.is_numeric_dtype(series):
                profile.stats = {
                    'mean': float(series.mean()),
                    'std': float(series.std()) + 1e-8,
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                }
            return profile

        # ---- 类型推断 ----

        # 1. 名字推断（强信号）
        col_lower = name.lower().strip()
        if any(kw in col_lower for kw in self.DATE_COLUMN_NAMES):
            profile.detected_type = "date"
            profile.date_format = self._detect_date_format(sample)
            return profile

        # 2. 样本值推断
        str_series = series.astype(str).str.strip()

        # 检查是否全是日期格式
        date_hits = 0
        for val in str_series.head(10):
            if self._looks_like_date(str(val)):
                date_hits += 1
        if date_hits >= 7:  # 70% 以上像日期
            profile.detected_type = "date"
            profile.date_format = self._detect_date_format(sample)
            return profile

        # 数值列
        if pd.api.types.is_numeric_dtype(series):
            profile.detected_type = "numeric"
            profile.stats = {
                'mean': float(series.mean()),
                'std': float(series.std()) + 1e-8,
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
            }
            return profile

        # 布尔/二分类别
        unique_vals = {str(v).lower().strip() for v in series.unique()}
        bool_candidates = {
            {'true', 'false'},
            {'是', '否'},
            {'yes', 'no'},
            {'1', '0'},
            {'男', '女'},
        }
        if unique_vals.issubset(bool_candidates):
            profile.detected_type = "categorical"
            profile.encoding_method = "label"
            return profile

        # 低基数类别（≤ 20 个唯一值）→ 类别
        if n_unique <= 20:
            profile.detected_type = "categorical"
            profile.encoding_method = "onehot" if n_unique <= 5 else "label"
            return profile

        # 高基数类别 → label encoding
        if n_unique <= 100 and not any(self._looks_like_date(str(v)) for v in str_series.head(5)):
            profile.detected_type = "categorical"
            profile.encoding_method = "label"
            return profile

        # 文本列（高基数、文本内容）
        profile.detected_type = "text"
        return profile

    def _looks_like_date(self, val: str) -> bool:
        if not val or len(val) < 4:
            return False
        for pattern, _, _ in self.DATE_PATTERNS:
            if re.match(pattern, val):
                return True
        return False

    def _detect_date_format(self, samples: List[Any]) -> str:
        for val in samples[:5]:
            val_str = str(val).strip()
            for pattern, fmt, _ in self.DATE_PATTERNS:
                if re.match(pattern, val_str):
                    return fmt
        return '%Y-%m-%d'

    def _setup_transforms(self, df: pd.DataFrame, config: Dict):
        """为每列建立变换器"""
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # 目标列：标准化
        if self.target_col in self.profiles:
            p = self.profiles[self.target_col]
            self._target_mean = p.stats.get('mean', 0.0)
            self._target_std = p.stats.get('std', 1.0)

        # 特征列
        for col, profile in self.profiles.items():
            if profile.is_target or profile.detected_type in ("empty", "unknown", "text"):
                continue

            series = df[col].copy()

            if profile.detected_type == "date":
                # 日期 → ordinal（距离参考日期的天数）
                try:
                    ref_date = pd.Timestamp('2000-01-01')
                    parsed = pd.to_datetime(series, errors='coerce')
                    ordinals = (parsed - ref_date).dt.days.fillna(0).astype(float)
                    self._date_parsers[col] = {
                        'ref': ref_date,
                        'mean': float(ordinals.mean()),
                        'std': float(ordinals.std()) + 1e-8,
                    }
                    self._numeric_scalers[col] = (
                        self._date_parsers[col]['mean'],
                        self._date_parsers[col]['std'],
                    )
                except Exception as e:
                    logger.warning(f"日期列 {col} 解析失败: {e}")
                    profile.detected_type = "unknown"

            elif profile.detected_type == "numeric":
                mean = float(series.mean())
                std = float(series.std()) + 1e-8
                self._numeric_scalers[col] = (mean, std)

            elif profile.detected_type == "categorical":
                le = LabelEncoder()
                try:
                    # 处理未知类别
                    vals = series.fillna('__MISSING__').astype(str)
                    le.fit(vals)
                    self._label_encoders[col] = le
                except Exception as e:
                    logger.warning(f"类别列 {col} 编码失败: {e}")

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        将 DataFrame 转换为解耦后的特征矩阵
        返回: (X, feature_names)
        """
        if not self._fitted:
            raise ValueError("请先调用 fit()")

        df = df.copy()
        self.feature_names = []
        columns = []

        for col, profile in self.profiles.items():
            if col not in df.columns:
                df[col] = np.nan

            series = df[col]

            if profile.detected_type == "empty":
                continue

            elif profile.detected_type == "numeric":
                mean, std = self._numeric_scalers.get(col, (0.0, 1.0))
                vals = pd.to_numeric(series, errors='coerce').fillna(mean).values.astype(float)
                normalized = (vals - mean) / std
                columns.append(normalized)
                self.feature_names.append(col)

            elif profile.detected_type == "date":
                ref = self._date_parsers[col]['ref']
                mean = self._date_parsers[col]['mean']
                std = self._date_parsers[col]['std']
                parsed = pd.to_datetime(series, errors='coerce')
                ordinals = (parsed - ref).dt.days.fillna(mean).astype(float)
                normalized = (ordinals - mean) / std
                columns.append(normalized.values)
                self.feature_names.append(col)

            elif profile.detected_type == "categorical":
                le = self._label_encoders.get(col)
                if le is None:
                    continue
                vals = series.fillna('__UNKNOWN__').astype(str)
                # 处理未知类别
                known = set(le.classes_)
                def safe_encode(v):
                    try:
                        if v in known:
                            return le.transform([v])[0]
                        return -1
                    except Exception:
                        return -1
                encoded = vals.apply(safe_encode).values.astype(float)
                columns.append(encoded)
                self.feature_names.append(col)

            elif profile.detected_type == "text":
                # 简单处理：文本长度作为数值特征
                lengths = series.astype(str).str.len().fillna(0).values.astype(float)
                mean_l = float(pd.Series(lengths).mean()) + 1e-8
                columns.append((lengths - mean_l) / mean_l)
                self.feature_names.append(f"{col}_len")

            elif profile.detected_type == "unknown":
                # 尝试当数值处理
                try:
                    vals = pd.to_numeric(series, errors='coerce').fillna(0).values.astype(float)
                    columns.append(vals)
                    self.feature_names.append(col)
                except Exception:
                    pass

        if not columns:
            raise ValueError("解耦后没有可用特征列")

        X = np.column_stack(columns)
        return X, self.feature_names

    def fit_transform(self, df: pd.DataFrame, target_col: str,
                      config: Optional[Dict] = None) -> Tuple[np.ndarray, List[str]]:
        """fit + transform"""
        self.fit(df, target_col, config)
        return self.transform(df)

    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """将归一化后的目标值反变换回原始尺度"""
        y = y_normalized * self._target_std + self._target_mean
        return y

    def get_summary(self) -> str:
        """返回解耦结果摘要"""
        lines = ["**数据解耦结果：**"]
        for col, p in self.profiles.items():
            if p.is_target:
                lines.append(f"  🎯 {col}（目标列）: {p.detected_type}")
            else:
                icon = {"numeric": "🔢", "date": "📅", "categorical": "🏷️",
                        "text": "📝", "unknown": "❓"}.get(p.detected_type, "")
                lines.append(f"  {icon} {col}: {p.detected_type} "
                             f"(唯一值={p.n_unique}, 缺失={p.missing_pct:.0%})")
        return "\n".join(lines)

    def _type_summary(self) -> str:
        from collections import Counter
        types = Counter(p.detected_type for p in self.profiles.values())
        return ", ".join(f"{k}={v}" for k, v in types.items())
