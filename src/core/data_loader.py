"""
数据加载模块
负责CSV导入、数据预览、基本分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """CSV数据加载器"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self._numeric_cols: list = []
        self._categorical_cols: list = []
        self._date_cols: list = []
    
    def load_csv(self, file_path: str) -> Tuple[bool, str]:
        """
        加载CSV文件
        Returns: (success, message)
        """
        try:
            self.file_path = Path(file_path)
            
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
        """分析列类型"""
        if self.df is None:
            return
        
        self._numeric_cols = []
        self._categorical_cols = []
        self._date_cols = []
        
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                self._numeric_cols.append(col)
            elif self._is_date_column(col):
                self._date_cols.append(col)
            else:
                # 判断是否是类别型（唯一值较少）
                if self.df[col].nunique() < 100:
                    self._categorical_cols.append(col)
                else:
                    self._categorical_cols.append(col)
    
    def _is_date_column(self, col: str) -> bool:
        """简单判断是否为日期列"""
        date_keywords = ['date', 'time', '时间', '日期', 'year', 'month', 'day']
        col_lower = col.lower()
        return any(kw in col_lower for kw in date_keywords)
    
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
        
        # 检查目标列类型
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
    
    def get_feature_matrix(self, exclude_cols: list = None) -> pd.DataFrame:
        """获取特征矩阵"""
        if self.df is None:
            return pd.DataFrame()
        
        exclude = exclude_cols or []
        cols = [c for c in self.df.columns if c not in exclude]
        return self.df[cols]
