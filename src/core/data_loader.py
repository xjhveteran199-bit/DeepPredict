"""
数据加载模块 v2
负责CSV导入、数据预览、自动列类型识别（数值/类别/日期）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataLoader:
    """CSV数据加载器 v2 - 增强列类型识别"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self._numeric_cols: list = []
        self._categorical_cols: list = []
        self._date_cols: list = []
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._date_parser_results: Dict[str, Any] = {}

    def load_csv(self, file_path: str) -> Tuple[bool, str]:
        """
        加载CSV文件
        Returns: (success, message)
        """
        try:
            # 安全检查：防止路径遍历攻击
            file_path_str = str(file_path)
            if '..' in file_path_str:
                return False, "❌ 路径不允许包含 '..' 遍历符"

            # 解析绝对路径并验证
            try:
                abs_path = Path(file_path).resolve()
            except (OSError, ValueError):
                return False, "❌ 路径解析失败"

            # 定义允许的基础目录（当前工作目录或用户主目录）
            allowed_bases = [Path.cwd(), Path.home(), Path('C:\\')]
            is_allowed = any(str(abs_path).startswith(str(base)) for base in allowed_bases)
            if not is_allowed:
                return False, "❌ 路径不在允许的目录范围内"

            self.file_path = abs_path
            self._label_encoders.clear()
            self._date_parser_results.clear()

            # 自动检测编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None

            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

            self.df = df
            self._analyze_columns()

            logger.info(f"成功加载CSV: {file_path}, 形状: {df.shape}")
            return True, f"成功加载 {len(df)} 行 × {len(df.columns)} 列数据"

        except Exception as e:
            logger.error(f"加载CSV失败: {e}")
            return False, f"加载失败: {str(e)}"

    def _analyze_columns(self):
        """分析列类型：数值/类别/日期"""
        if self.df is None:
            return

        self._numeric_cols = []
        self._categorical_cols = []
        self._date_cols = []

        for col in self.df.columns:
            col_type = self._detect_column_type(col)
            if col_type == 'numeric':
                self._numeric_cols.append(col)
            elif col_type == 'date':
                self._date_cols.append(col)
            else:
                self._categorical_cols.append(col)

        logger.info(f"列类型识别: 数值列={len(self._numeric_cols)}, "
                    f"类别列={len(self._categorical_cols)}, 日期列={len(self._date_cols)}")

    def _detect_column_type(self, col: str) -> str:
        """智能检测列类型"""
        series = self.df[col]

        # === 1. 日期列检测 ===
        if self._is_date_column(col, series):
            # 尝试解析
            parsed = pd.to_datetime(series, errors='coerce', format='mixed')
            not_null_ratio = parsed.notna().sum() / max(len(parsed), 1)
            if not_null_ratio >= 0.8:
                self._date_parser_results[col] = parsed
                return 'date'

        # === 2. 数值列检测 ===
        if series.dtype in ['int64', 'float64', 'int32', 'float32']:
            unique_ratio = series.nunique() / max(series.notna().sum(), 1)
            # 唯一值很多 → 数值型（回归用）
            if unique_ratio > 0.1:
                return 'numeric'
            # 唯一值少且是整数 → 可能是类别编码
            if unique_ratio <= 0.05 and series.dtype in ['int64', 'int32']:
                return 'categorical'
            return 'numeric'

        # === 3. 类别列检测 ===
        # object/category 类型
        unique_count = series.nunique()
        if unique_count <= 50 and unique_count >= 2:
            return 'categorical'

        # 数值但唯一值很少（可能是类别）
        if unique_count <= 20:
            return 'categorical'

        return 'categorical'

    def _is_date_column(self, col: str, series: pd.Series) -> bool:
        """判断是否为日期列"""
        # 1. dtype 是 datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # 2. 字符串列尝试 parse（仅对object/str类型做列名探测）
        if series.dtype == 'object':
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            # 常见日期格式样例
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',      # 2024-01-01
                r'\d{4}/\d{2}/\d{2}',        # 2024/01/01
                r'\d{2}-\d{2}-\d{4}',        # 01-01-2024
                r'\d{4}年\d{1,2}月\d{1,2}日',  # 2024年1月1日
            ]
            import re
            for pattern in date_patterns:
                if sample.astype(str).str.match(pattern).any():
                    return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        if self.df is None:
            return {}

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "numeric_cols": self._numeric_cols,
            "categorical_cols": self._categorical_cols,
            "date_cols": self._date_cols,
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing": self.df.isnull().sum().to_dict(),
            "numeric_stats": self.df[self._numeric_cols].describe().to_dict() if self._numeric_cols else {}
        }

    def get_preview(self, rows: int = 20) -> pd.DataFrame:
        """获取数据预览"""
        if self.df is None:
            return pd.DataFrame()
        return self.df.head(rows)

    def select_target(self, target_col: str) -> Tuple[bool, str]:
        """选择目标列"""
        if self.df is None:
            return False, "未加载数据"

        if target_col not in self.df.columns:
            return False, f"列 '{target_col}' 不存在"

        # 推断任务类型
        if target_col in self._numeric_cols:
            task_type = "regression"
        else:
            unique_count = self.df[target_col].nunique()
            if unique_count <= 10:
                task_type = "classification"
            else:
                task_type = "regression"

        logger.info(f"选择目标列: {target_col}, 推断任务类型: {task_type}")
        return True, f"目标列: {target_col} (推断任务类型: {task_type})"

    def get_feature_matrix(
        self,
        exclude_cols: list = None,
        return_encoded: bool = False
    ) -> pd.DataFrame:
        """
        获取特征矩阵

        Args:
            exclude_cols: 排除的列（通常是目标列）
            return_encoded: True=返回独热编码后的类别列, False=返回原始值
        """
        if self.df is None:
            return pd.DataFrame()

        exclude = set(exclude_cols or [])

        # 1. 数值特征
        numeric_features = [c for c in self._numeric_cols if c not in exclude]

        # 2. 类别特征 → LabelEncoder 编码
        cat_features = [c for c in self._categorical_cols if c not in exclude]
        encoded_cat = {}
        for col in cat_features:
            if col not in self._label_encoders:
                le = LabelEncoder()
                mode_series = self.df[col].mode()
                fill_value = mode_series[0] if not mode_series.empty else 'missing'
                series = self.df[col].fillna(fill_value)
                encoded_cat[col] = le.fit_transform(series.astype(str))
                self._label_encoders[col] = le
            else:
                encoded_cat[col] = self._label_encoders[col].transform(
                    self.df[col].fillna('missing').astype(str)
                )

        # 3. 日期特征 → 提取时间特征
        date_features = [c for c in self._date_cols if c not in exclude]
        date_feature_names = []
        for col in date_features:
            parsed = self._date_parser_results.get(col)
            if parsed is None:
                parsed = pd.to_datetime(self.df[col], errors='coerce')
            date_feats = self._extract_date_features(parsed)
            for feat_name, feat_series in date_feats.items():
                encoded_cat[feat_name] = feat_series
                date_feature_names.append(feat_name)

        # 合并所有特征
        all_features = {}
        if numeric_features:
            all_features.update({c: self.df[c].values for c in numeric_features})
        all_features.update(encoded_cat)

        result_df = pd.DataFrame(all_features)

        logger.info(f"特征矩阵构建完成: {result_df.shape[1]} 特征 "
                    f"(数值={len(numeric_features)}, 类别={len(cat_features)}, "
                    f"日期特征={len(date_feature_names)})")
        return result_df

    def _extract_date_features(self, parsed_series: pd.Series) -> Dict[str, pd.Series]:
        """从日期列提取时间特征"""
        features = {}
        prefix = 'date_'

        try:
            dt = pd.to_datetime(parsed_series, errors='coerce')
            valid = dt.notna()

            features[f'{prefix}year'] = dt.dt.year.fillna(0).astype(int)
            features[f'{prefix}month'] = dt.dt.month.fillna(0).astype(int)
            features[f'{prefix}day'] = dt.dt.day.fillna(0).astype(int)
            features[f'{prefix}dayofweek'] = dt.dt.dayofweek.fillna(0).astype(int)
            features[f'{prefix}dayofyear'] = dt.dt.dayofyear.fillna(0).astype(int)
            features[f'{prefix}quarter'] = dt.dt.quarter.fillna(0).astype(int)
            features[f'{prefix}weekofyear'] = dt.dt.isocalendar().week.fillna(0).astype(int)
            # 安全处理 hour：dt.dt.hour 可能对无效日期返回全 NaT，需检查
            hour_series = dt.dt.hour
            features[f'{prefix}hour'] = hour_series.fillna(0).astype(int) if valid.any() and not hour_series.isna().all() else 0

            # 周期性特征（sin/cos编码）
            if valid.any():
                # 月份周期
                month_sin = np.sin(2 * np.pi * dt.dt.month / 12)
                month_cos = np.cos(2 * np.pi * dt.dt.month / 12)
                features[f'{prefix}month_sin'] = month_sin.fillna(0)
                features[f'{prefix}month_cos'] = month_cos.fillna(0)
                # 星期周期
                dow_sin = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
                dow_cos = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
                features[f'{prefix}dow_sin'] = dow_sin.fillna(0)
                features[f'{prefix}dow_cos'] = dow_cos.fillna(0)
                # 一天中的小时周期（如果有）
                if dt.dt.hour.notna().any():
                    hour_sin = np.sin(2 * np.pi * dt.dt.hour / 24)
                    hour_cos = np.cos(2 * np.pi * dt.dt.hour / 24)
                    features[f'{prefix}hour_sin'] = hour_sin.fillna(0)
                    features[f'{prefix}hour_cos'] = hour_cos.fillna(0)

        except Exception as e:
            logger.warning(f"日期特征提取失败: {e}")

        # 移除 None 值
        features = {k: v for k, v in features.items() if v is not None}
        return features

    def detect_irregular_sampling(self, date_col: str = None) -> Dict[str, Any]:
        """
        检测时序采样是否规则

        Returns:
            dict: {
                'is_regular': bool,
                'interval_mean': float,
                'interval_std': float,
                'cv': float,  # 变异系数 (std/mean)
                'suggested_interval': str,  # '1H', '1D', etc.
                'needs_resampling': bool
            }
        """
        if self.df is None:
            return {'is_regular': False}

        # 找到日期列
        if date_col is None:
            date_cols = [c for c in self._date_cols if c in self.df.columns]
            if not date_cols:
                # 尝试第一列
                date_cols = [self.df.columns[0]]
            date_col = date_cols[0]

        try:
            dt = pd.to_datetime(self.df[date_col], errors='coerce').dropna()
            if len(dt) < 2:
                return {'is_regular': False, 'reason': 'too few samples'}

            intervals = dt.diff().dropna()
            interval_mean = intervals.mean().total_seconds()
            interval_std = intervals.std().total_seconds()
            cv = interval_std / interval_mean if interval_mean > 0 else float('inf')

            # 判断是否规则：CV < 0.1 表示采样间隔较一致
            is_regular = cv < 0.1

            # 推荐采样间隔
            if interval_mean < 3600:  # < 1小时
                suggested = '1T'  # 1分钟
            elif interval_mean < 86400:  # < 1天
                suggested = '1H'  # 1小时
            else:
                suggested = '1D'  # 1天

            result = {
                'is_regular': is_regular,
                'interval_mean_seconds': interval_mean,
                'interval_std_seconds': interval_std,
                'cv': cv,
                'suggested_interval': suggested,
                'needs_resampling': not is_regular,
                'date_col': date_col
            }

            logger.info(f"采样检测: 规则={is_regular}, CV={cv:.4f}, "
                        f"平均间隔={interval_mean:.0f}s, 建议重采样={suggested}")
            return result

        except Exception as e:
            logger.warning(f"采样检测失败: {e}")
            return {'is_regular': None, 'error': str(e)}

    def resample_to_regular(
        self,
        date_col: str,
        freq: str = '1H',
        method: str = 'linear'
    ) -> bool:
        """
        将不规则时序重采样为规则间隔

        Args:
            date_col: 日期列名
            freq: 目标频率，如 '1H', '1D', '30T'
            method: 插值方法 'linear' 或 'ffill'

        Returns:
            是否成功
        """
        if self.df is None:
            return False

        try:
            dt = pd.to_datetime(self.df[date_col], errors='coerce')
            if dt.isna().all():
                return False

            # 设置日期为索引
            df_indexed = self.df.set_index(dt)
            df_indexed = df_indexed[~df_indexed.index.isna()]

            # 选择数值列进行插值
            numeric_to_resample = [
                c for c in self._numeric_cols
                if c != date_col and c in df_indexed.columns
            ]

            # 重采样并插值
            df_resampled = df_indexed[numeric_to_resample].resample(freq).mean()

            if method == 'linear':
                df_resampled = df_resampled.interpolate(method='linear')
            else:
                df_resampled = df_resampled.ffill()

            # 重置索引，恢复日期列
            df_resampled = df_resampled.reset_index()
            df_resampled[date_col] = df_resampled['index']
            df_resampled = df_resampled.drop(columns=['index'])

            # 保留非数值列的原始值（前向填充）
            non_numeric = [c for c in self.df.columns if c not in self._numeric_cols and c != date_col]
            for col in non_numeric:
                if col in self.df.columns:
                    mapping = dict(zip(self.df[date_col], self.df[col]))
                    df_resampled[col] = df_resampled[date_col].map(mapping).ffill()

            self.df = df_resampled
            # 重新分析列类型
            self._analyze_columns()

            logger.info(f"重采样完成: {len(self.df)} 行, 频率={freq}")
            return True

        except Exception as e:
            logger.error(f"重采样失败: {e}")
            return False
