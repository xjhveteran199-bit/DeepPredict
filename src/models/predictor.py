"""
预测器模块
负责模型训练、预测和结果输出
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
from pathlib import Path

# LSTM 模型 - 延迟导入，避免 torch DLL 问题
# from models.lstm_model import LSTMPredictor  # 移至使用时导入

logger = logging.getLogger(__name__)


class Predictor:
    """模型预测器"""

    def __init__(self):
        self.model = None
        self.lstm_predictor: Optional[LSTMPredictor] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.task_type: Optional[str] = None
        self.feature_names: Optional[list] = None
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self._is_lstm: bool = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        model_name: str = 'RandomForest',
        model_params: Dict = None,
        test_size: float = 0.2
    ) -> Tuple[bool, str]:
        """
        训练模型

        Returns: (success, message)
        """
        try:
            self.task_type = task_type
            self.feature_names = list(X.columns)
            
            # ===== LSTM 特殊处理 =====
            if model_name == 'LSTM':
                # 延迟导入 torch（避免启动时 DLL 报错）
                from models.lstm_model import LSTMPredictor
                
                self._is_lstm = True
                params = model_params or {}
                
                # 准备数据
                X_array = X.values.astype(np.float32)
                y_array = y.values.astype(np.float32)
                
                target_col = y.name or 'target'
                
                # 创建 LSTM 预测器
                self.lstm_predictor = LSTMPredictor()
                success, msg = self.lstm_predictor.train(
                    X=X_array,
                    y=y_array,
                    hidden_size=params.get('hidden_size', 64),
                    num_layers=params.get('num_layers', 2),
                    epochs=params.get('epochs', 50),
                    batch_size=params.get('batch_size', 32),
                    learning_rate=params.get('learning_rate', 0.001),
                    seq_len=params.get('seq_len', 10),
                    test_size=test_size,
                    target_col=target_col
                )
                
                self.metrics = self.lstm_predictor.metrics
                self.is_fitted = True
                self.feature_names = list(X.columns)
                
                return success, msg
            
            # ===== 普通 sklearn 模型 =====
            self._is_lstm = False

            # 数据预处理
            X_processed, y_processed = self._preprocess(X, y, task_type)

            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=42
            )

            # 创建模型
            self.model = self._create_model(model_name, model_params, task_type)

            # 训练
            logger.info(f"开始训练模型: {model_name}, 训练集大小: {len(X_train)}")
            self.model.fit(X_train, y_train)

            # 评估
            self.metrics = self._evaluate(X_test, y_test, task_type)

            self.is_fitted = True

            msg = f"✅ 训练完成！\n"
            msg += f"   模型: {model_name}\n"
            msg += f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}\n"
            msg += self._format_metrics()

            logger.info(f"训练完成，指标: {self.metrics}")
            return True, msg

        except Exception as e:
            logger.error(f"训练失败: {e}")
            return False, f"训练失败: {str(e)}"

    def _preprocess(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[pd.DataFrame, Any]:
        """数据预处理"""
        X = X.copy()

        # 处理缺失值
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)

        # 处理类别特征
        self.scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        # 对非数值列进行编码
        for col in X.columns:
            if col not in numeric_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # 目标变量处理
        if task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.astype(float)

        return X, y

    def _create_model(self, model_name: str, params: Dict, task_type: str):
        """创建模型实例"""
        params = params or {}

        model_map = {
            ('RandomForest', 'regression'): RandomForestRegressor,
            ('RandomForest', 'classification'): RandomForestClassifier,
            ('RandomForest', 'time_series'): RandomForestRegressor,
            ('GradientBoosting', 'regression'): GradientBoostingRegressor,
            ('GradientBoosting', 'classification'): GradientBoostingClassifier,
            ('GradientBoosting', 'time_series'): GradientBoostingRegressor,
            ('LinearRegression', 'regression'): LinearRegression,
            ('LinearRegression', 'time_series'): LinearRegression,
            ('LogisticRegression', 'classification'): LogisticRegression,
        }

        # 默认处理
        key = (model_name, task_type)
        if key not in model_map:
            # 未知组合，默认用对应task的RandomForest
            if task_type == 'classification':
                return RandomForestClassifier(**params) if params else RandomForestClassifier()
            elif task_type == 'time_series':
                return RandomForestRegressor(**params) if params else RandomForestRegressor()
            else:
                return RandomForestRegressor(**params) if params else RandomForestRegressor()

        model_class = model_map[key]
        return model_class(**params)

    def _evaluate(self, X_test, y_test, task_type: str) -> Dict[str, float]:
        """评估模型"""
        y_pred = self.model.predict(X_test)
        metrics = {}

        if task_type == 'regression' or task_type == 'time_series':
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['MAE'] = mean_absolute_error(y_test, y_pred)
            metrics['R2'] = r2_score(y_test, y_pred)
        else:  # classification
            metrics['Accuracy'] = accuracy_score(y_test, y_pred)
            metrics['Precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['Recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['F1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        return metrics

    def _format_metrics(self) -> str:
        """格式化指标输出"""
        if not self.metrics:
            return ""

        lines = []
        if 'R2' in self.metrics:
            lines.append(f"   • R² 分数: {self.metrics['R2']:.4f}")
            lines.append(f"   • RMSE: {self.metrics['RMSE']:.4f}")
            lines.append(f"   • MAE: {self.metrics['MAE']:.4f}")
        else:
            lines.append(f"   • 准确率: {self.metrics.get('Accuracy', 0):.2%}")
            lines.append(f"   • 精确率: {self.metrics.get('Precision', 0):.2%}")
            lines.append(f"   • 召回率: {self.metrics.get('Recall', 0):.2%}")
            lines.append(f"   • F1分数: {self.metrics.get('F1', 0):.4f}")

        return "\n".join(lines)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """对新数据进行预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先训练模型")

        # LSTM 预测
        if self._is_lstm and self.lstm_predictor is not None:
            X_array = X.values.astype(np.float32)
            return self.lstm_predictor.predict(X_array)

        # sklearn 模型预测
        X = X.copy()

        # 应用相同的预处理
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # 填充缺失值（使用训练集的统计值，这里简化处理）
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)

        # 标准化
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        # 编码非数值列
        for col in X.columns:
            if col not in numeric_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        predictions = self.model.predict(X)

        # 反向转换分类结果
        if self.task_type == 'classification' and self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            return {}

        # LSTM 不支持特征重要性
        if self._is_lstm:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        return {}

    def save_model(self, path: str):
        """保存模型"""
        if self._is_lstm and self.lstm_predictor is not None:
            self.lstm_predictor.save_model(path)
            return

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }, path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        # 尝试加载 torch 模型（LSTM）
        try:
            from models.lstm_model import LSTMPredictor
            if self.lstm_predictor is None:
                self.lstm_predictor = LSTMPredictor()
            self.lstm_predictor.load_model(path)
            self._is_lstm = True
            self.is_fitted = True
            self.metrics = self.lstm_predictor.metrics
            logger.info(f"LSTM 模型已加载: {path}")
            return
        except Exception:
            pass
        
        # 回退到 sklearn 模型
        self._is_lstm = False
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.task_type = data['task_type']
        self.feature_names = data['feature_names']
        self.metrics = data.get('metrics', {})
        self.is_fitted = True
        logger.info(f"模型已加载: {path}")
