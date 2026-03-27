"""
CSV 解析引擎
支持自动检测（分隔符、表头、日期列、格式）和手动配置解析
"""
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParseConfig:
    """CSV 解析配置"""
    # 模式
    mode: str = "auto"  # auto | manual

    # 表头
    has_header: bool = True
    custom_column_names: Optional[List[str]] = None  # 无表头时手动指定

    # 分隔符
    separator: str = "auto"  # auto | , | ; | \t | space | 其他

    # 日期列
    date_col: Optional[str] = None  # 列名，或 "auto"
    date_format: str = "auto"  # auto | 具体格式字符串

    # 索引列
    index_col: Optional[str] = None  # 列名，或 None

    # 缺失值
    missing_strategy: str = "auto"  # auto | drop | fill_mean | fill_forward | fill_value | none
    missing_fill_value: Any = 0  # fill_value 时使用的值

    # 编码
    encoding: str = "utf-8"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'has_header': self.has_header,
            'custom_column_names': self.custom_column_names,
            'separator': self.separator,
            'date_col': self.date_col,
            'date_format': self.date_format,
            'index_col': self.index_col,
            'missing_strategy': self.missing_strategy,
            'missing_fill_value': self.missing_fill_value,
            'encoding': self.encoding,
        }


class CSVParser:
    """
    CSV 解析器

    auto_detect() → ParseConfig  # 系统自动检测
    parse()       → DataFrame    # 按配置解析
    """

    DATE_PATTERNS = [
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
        (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),
        (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '%Y-%m-%dT%H:%M:%S'),
        (r'^\d{4}年\d{1,2}月\d{1,2}日$', '%Y年%m月%d日'),
        (r'^\d{4}年\d{1,2}月$', '%Y年%m月'),
        (r'^\d{10}$', 'epoch'),  # Unix timestamp
        (r'^\d{13}$', 'epoch_ms'),  # Unix timestamp ms
    ]

    DATE_COLUMN_NAMES = {
        'date', 'time', 'datetime', 'timestamp', '时间', '日期',
        'year', 'month', 'day', 'hour', 'minute', 'second',
        'weekday', 'week', 'quarter',
    }

    def __init__(self):
        self._last_raw_preview: Optional[List[List[str]]] = None

    def _read_raw_lines(self, path: str, n: int = 5) -> List[List[str]]:
        """读取 CSV 原始行（不解析），用于自动检测"""
        # 自动检测编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
        raw_bytes = None
        detected_enc = 'utf-8'
        try:
            with open(path, 'rb') as f:
                raw_bytes = f.read(10000)
            for enc in encodings:
                try:
                    raw_bytes.decode(enc)
                    detected_enc = enc
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
        except Exception:
            pass

        try:
            # 先尝试不同分隔符读取第一行
            for sep in [',', ';', '\t', '|']:
                try:
                    with open(path, 'r', encoding=detected_enc, errors='replace') as f:
                        lines = [f.readline() for _ in range(n + 1)]
                    first_line = lines[0]
                    if sep in first_line:
                        rows = []
                        for line in lines:
                            rows.append([c.strip() for c in line.split(sep)])
                        self._last_raw_preview = rows
                        return rows
                except Exception:
                    continue

            # 最后用默认逗号
            with open(path, 'r', encoding=detected_enc, errors='replace') as f:
                lines = [f.readline() for _ in range(n + 1)]
            rows = []
            for line in lines:
                rows.append([c.strip() for c in line.split(',')])
            self._last_raw_preview = rows
            return rows

        except Exception as e:
            logger.warning(f"读取原始行失败: {e}")
            return []

    def _is_date_value(self, val: str) -> bool:
        """判断单个值是否像日期"""
        if not val or len(val) < 4:
            return False
        for pattern, _ in self.DATE_PATTERNS:
            if re.match(pattern, str(val).strip()):
                return True
        return False

    def _detect_separator(self, path: str) -> str:
        """自动检测分隔符"""
        # 自动检测编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
        detected_enc = 'utf-8'
        try:
            with open(path, 'rb') as f:
                raw = f.read(200)
            for enc in encodings:
                try:
                    raw.decode(enc)
                    detected_enc = enc
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
        except Exception:
            pass

        try:
            with open(path, 'r', encoding=detected_enc, errors='replace') as f:
                first_line = f.readline()

            separators = {',': 0, ';': 0, '\t': 0, '|': 0, ' ': 0}
            for char in first_line:
                if char in separators:
                    separators[char] += 1

            best_sep = max(separators, key=separators.get)
            if separators[best_sep] > 0:
                sep_map = {',': ',', ';': ';', '\t': '\\t', '|': '|', ' ': ' '}
                return sep_map.get(best_sep, ',')
        except Exception as e:
            logger.warning(f"分隔符检测失败: {e}")
        return ','

    def _detect_header(self, first_row: List[str]) -> bool:
        """判断第一行是否是表头（非全数值 = 是表头）"""
        if not first_row:
            return True
        numeric_count = sum(1 for v in first_row if self._is_numeric(v))
        return numeric_count / len(first_row) < 0.5

    def _is_numeric(self, val: str) -> bool:
        """判断字符串是否是数值"""
        try:
            float(str(val).strip())
            return True
        except (ValueError, TypeError):
            return False

    def _detect_date_col(self, rows: List[List[str]], has_header: bool) -> Optional[str]:
        """自动检测日期列"""
        data_rows = rows[1:] if has_header else rows
        if not data_rows:
            return None

        # 方式1：按列名检测
        if has_header and rows:
            header = [str(c).lower().strip() for c in rows[0]]
            for i, col in enumerate(header):
                if col in self.DATE_COLUMN_NAMES:
                    return rows[0][i]  # 返回原始列名

        # 方式2：按值内容检测（统计每列的日期命中率）
        n_check = min(10, len(data_rows))
        best_col = None
        best_score = 0

        n_cols = len(data_rows[0])
        for col_idx in range(n_cols):
            hits = sum(1 for row in data_rows[:n_check]
                       if self._is_date_value(str(row[col_idx]).strip()))
            score = hits / n_check
            if score > best_score and score >= 0.7:
                best_score = score
                best_col = col_idx

        if best_col is not None and has_header and rows:
            return rows[0][best_col]
        elif best_col is not None:
            return f"col_{best_col}"
        return None

    def _detect_date_format(self, rows: List[List[str]], date_col_name: str) -> str:
        """自动检测日期格式"""
        if not rows or not date_col_name:
            return '%Y-%m-%d'

        # 找日期列的索引
        col_idx = None
        if has_header := (rows[0] and date_col_name in rows[0]):
            col_idx = rows[0].index(date_col_name)
        elif date_col_name.startswith('col_'):
            try:
                col_idx = int(date_col_name.split('_')[1])
            except (IndexError, ValueError):
                pass

        if col_idx is None:
            return '%Y-%m-%d'

        # 检查值
        for row in rows[1:min(20, len(rows))]:
            if col_idx < len(row):
                val = str(row[col_idx]).strip()
                for pattern, fmt in self.DATE_PATTERNS:
                    if re.match(pattern, val):
                        return fmt
        return '%Y-%m-%d'

    def _detect_missing_strategy(self, df: pd.DataFrame) -> str:
        """根据缺失率自动选择缺失值策略"""
        total = df.size
        if total == 0:
            return 'none'
        missing_rate = df.isna().sum().sum() / total
        if missing_rate == 0:
            return 'none'
        elif missing_rate < 0.05:
            return 'fill_mean'
        else:
            return 'drop'

    def auto_detect(self, file_path: str) -> Tuple[ParseConfig, Dict[str, Any]]:
        """
        自动检测 CSV 配置

        返回：
          - ParseConfig：检测到的配置
          - Dict：诊断信息（用于 UI 显示）
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取原始行
        rows = self._read_raw_lines(str(path), n=50)
        if not rows:
            raise ValueError("无法读取文件内容")

        # 检测分隔符
        sep = self._detect_separator(str(path))
        sep_display = {'\\t': 'Tab', ',': '逗号', ';': '分号', '|': '竖线', ' ': '空格'}.get(sep, sep)

        # 检测表头
        has_header = self._detect_header(rows[0])
        header_display = "有表头（第一行）" if has_header else "无表头"

        # 预览行（用于 UI）
        preview_rows = rows[1:51] if has_header else rows[:50]

        # 检测日期列
        detected_date_col = self._detect_date_col(rows, has_header)
        detected_date_format = self._detect_date_format(rows, detected_date_col) if detected_date_col else 'auto'

        # 自动检测编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
        detected_enc = 'utf-8'
        try:
            with open(path, 'rb') as f:
                raw = f.read(10000)
            for enc in encodings:
                try:
                    raw.decode(enc)
                    detected_enc = enc
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
        except Exception:
            pass

        # 尝解析看整体信息（只读一次前100行，避免重复I/O）
        try:
            test_df = pd.read_csv(
                str(path),
                sep=sep,
                header=0 if has_header else None,
                encoding=detected_enc,
                encoding_errors='replace',
                nrows=100,
            )
            # 用文件大小估算行数（避免读整个文件），CSV平均每行约200字节
            file_size_bytes = path.stat().st_size
            n_rows_estimate = max(1, int(file_size_bytes / 200))
            n_cols = len(test_df.columns)
            col_names = list(test_df.columns)
            numeric_cols = [c for c in col_names if pd.api.types.is_numeric_dtype(test_df[c])]
            missing_summary = test_df.isna().sum().to_dict()
        except Exception as e:
            n_rows_estimate = "未知"
            n_cols = len(rows[0])
            col_names = rows[0] if has_header else [f"col_{i}" for i in range(n_cols)]
            numeric_cols = []
            missing_summary = {}

        config = ParseConfig(
            mode="auto",
            has_header=has_header,
            separator=sep,
            date_col=detected_date_col,
            date_format=detected_date_format,
            missing_strategy=self._detect_missing_strategy(test_df),
        )

        diagnostics = {
            'file_name': path.name,
            'file_size_mb': round(path.stat().st_size / 1024 / 1024, 2),
            'estimated_rows': n_rows_estimate,
            'detected_cols': n_cols,
            'column_names': col_names,
            'numeric_cols': numeric_cols,
            'detected_separator': sep_display,
            'detected_header': header_display,
            'detected_date_col': detected_date_col,
            'detected_date_format': detected_date_format,
            'missing_summary': {k: v for k, v in missing_summary.items() if v > 0},
            'preview_rows': preview_rows,
            'has_header': has_header,
        }

        return config, diagnostics

    def parse(self, file_path: str, config: ParseConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        按配置解析 CSV

        返回：
          - DataFrame：解析后的数据
          - Dict：解析报告
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        sep = config.separator if config.separator != 'auto' else ','

        # 自动检测编码（当配置为默认utf-8时尝试检测）
        enc = config.encoding
        if enc == 'utf-8':
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
            detected = 'utf-8'
            try:
                with open(path, 'rb') as f:
                    raw = f.read(10000)
                for e in encodings:
                    try:
                        raw.decode(e)
                        detected = e
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                enc = detected
            except Exception:
                pass

        # 读取 CSV
        try:
            if not config.has_header and config.custom_column_names:
                df = pd.read_csv(
                    str(path),
                    sep=sep,
                    header=None,
                    names=config.custom_column_names,
                    encoding=enc,
                    encoding_errors='replace',
                )
            elif not config.has_header:
                df = pd.read_csv(
                    str(path),
                    sep=sep,
                    header=None,
                    encoding=enc,
                    encoding_errors='replace',
                )
                df.columns = [f"col_{i}" for i in range(len(df.columns))]
            else:
                df = pd.read_csv(
                    str(path),
                    sep=sep,
                    header=0,
                    encoding=enc,
                    encoding_errors='replace',
                )
        except Exception as e:
            raise ValueError(f"CSV 解析失败: {e}")

        # 处理日期列
        parse_warnings = []
        if config.date_col and config.date_col != 'auto':
            if config.date_col in df.columns:
                date_format = None if config.date_format == 'auto' else config.date_format
                try:
                    if date_format and date_format not in ('epoch', 'epoch_ms'):
                        df[config.date_col] = pd.to_datetime(df[config.date_col], format=date_format, errors='coerce')
                    elif date_format == 'epoch':
                        df[config.date_col] = pd.to_datetime(df[config.date_col].astype(float), unit='s', errors='coerce')
                    elif date_format == 'epoch_ms':
                        df[config.date_col] = pd.to_datetime(df[config.date_col].astype(float), unit='ms', errors='coerce')
                    else:
                        df[config.date_col] = pd.to_datetime(df[config.date_col], errors='coerce')

                    invalid_dates = df[config.date_col].isna().sum()
                    if invalid_dates > 0:
                        parse_warnings.append(f"日期列 {config.date_col} 有 {invalid_dates} 个无效日期")
                except Exception as e:
                    parse_warnings.append(f"日期解析失败: {e}，已转为字符串")
                    df[config.date_col] = df[config.date_col].astype(str)
            else:
                parse_warnings.append(f"指定的日期列 '{config.date_col}' 不存在，跳过日期解析")

        # 处理缺失值
        missing_before = int(df.isna().sum().sum())
        if config.missing_strategy == 'drop':
            df = df.dropna()
        elif config.missing_strategy == 'fill_mean':
            df = df.fillna(df.select_dtypes(include=[np.number]).mean())
        elif config.missing_strategy == 'fill_forward':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif config.missing_strategy == 'fill_value':
            df = df.fillna(config.missing_fill_value)
        # 'none' / 'auto' → 不处理
        missing_after = int(df.isna().sum().sum())

        # 解析报告
        report = {
            'rows_before': len(df),
            'rows_after': len(df),
            'cols': list(df.columns),
            'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
            'missing_before': missing_before,
            'missing_after': missing_after,
            'missing_handled': config.missing_strategy,
            'warnings': parse_warnings,
            'date_col': config.date_col if (config.date_col and config.date_col != 'auto') else None,
            'parse_success': True,
        }

        logger.info(f"CSV 解析完成: {len(df)} 行 × {len(df.columns)} 列, 缺失值: {missing_before} → {missing_after} ({config.missing_strategy})")
        return df, report
