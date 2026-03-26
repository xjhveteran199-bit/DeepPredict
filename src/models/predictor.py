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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
import joblib

logger = logging.getLogger(__name__)


class Predictor:
    """模型预测器"""

    def __init__(self):
        self.model = None
        self.lstm_predictor: Optional[Any] = None
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
        try:
            self.task_type = task_type
            self.feature_names = list(X.columns)

            # ===== LSTM 特殊处理 =====
            if model_name == 'LSTM':
                from models.lstm_model import LSTMPredictor
                
                self._is_lstm = True
                params = model_params or {}
                
                X_array = X.values.astype(np.float32)
                y_array = y.values.astype(np.float32)
                target_col = y.name or 'target'
                
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

            # ===== PatchTST 特殊处理 =====
            if model_name == 'PatchTST':
                from models.patchtst_model import PatchTSTPredictor
                
                self._is_lstm = False
                params = model_params or {}
                
                X_array = X.values.astype(np.float32)
                y_array = y.values.astype(np.float32)
                target_col = y.name or 'target'
                
                self.lstm_predictor = PatchTSTPredictor()
                success, msg = self.lstm_predictor.train(
                    X=X_array,
                    y=y_array,
                    seq_len=params.get('seq_len', 96),
                    pred_len=params.get('pred_len', 96),
                    patch_size=params.get('patch_size', 16),
                    d_model=params.get('d_model', 128),
                    n_heads=params.get('n_heads', 4),
                    n_layers=params.get('n_layers', 3),
                    d_ff=params.get('d_ff', 256),
                    epochs=params.get('epochs', 30),
                    batch_size=params.get('batch_size', 32),
                    learning_rate=params.get('learning_rate', 0.0005),
                    test_size=test_size,
                    target_col=target_col
                )
                
                self.metrics = self.lstm_predictor.metrics
                self.is_fitted = True
                self.feature_names = list(X.columns)
                return success, msg

            # ===== sklearn 模型 =====
            self._is_lstm = False
            self.scaler = StandardScaler()
            # 只对有 NaN 的列填充，避免所有列为 NaN 时 median() 返回全 NaN Series
            X_filled = X.copy()
            nan_cols = X.columns[X.isna().any()]
            if len(nan_cols) > 0:
                X_filled[nan_cols] = X[nan_cols].fillna(X[nan_cols].median())
            X_scaled = self.scaler.fit_transform(X_filled)

            if task_type == 'classification':
                self.label_encoder = LabelEncoder()
                y_enc = self.label_encoder.fit_transform(y.astype(str))
            else:
                y_enc = y.astype(float)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_enc, test_size=test_size, random_state=42
            )

            self.model = self._create_model(model_name, model_params, task_type)
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            if task_type == 'classification':
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                self.metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
                metrics_str = f"准确率={acc:.2%} 精确率={prec:.2%} 召回率={rec:.2%} F1={f1:.4f}"
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                self.metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
                metrics_str = f"R²={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}"

            self.is_fitted = True

            msg = f"✅ {model_name} 训练完成！\n   {metrics_str}"
            logger.info(f"训练完成: {self.metrics}")
            return True, msg

        except Exception as e:
            logger.error(f"训练失败: {e}")
            return False, f"训练失败: {str(e)}"

    def _create_model(self, model_name: str, params: Dict, task_type: str):
        params = params or {}
        
        # 过滤掉时序模型参数，避免传给 sklearn 模型
        sklearn_forbidden_keys = {
            'seq_len', 'pred_len', 'hidden_channels', 'hidden_size',
            'kernel_size', 'num_layers', 'patch_size', 'd_model',
            'n_heads', 'n_layers', 'd_ff', 'dropout', 'n_date_features',
            'epochs', 'batch_size', 'learning_rate', 'weight_decay',
            'grad_clip', 'early_stopping', 'seq_len', 'pred_len'
        }
        filtered_params = {k: v for k, v in params.items() 
                         if k not in sklearn_forbidden_keys}
        
        model_map = {
            ('RandomForest', 'regression'): RandomForestRegressor,
            ('RandomForest', 'classification'): RandomForestClassifier,
            ('GradientBoosting', 'regression'): GradientBoostingRegressor,
            ('GradientBoosting', 'classification'): GradientBoostingClassifier,
            ('LinearRegression', 'regression'): LinearRegression,
            ('LogisticRegression', 'classification'): LogisticRegression,
        }
        
        key = (model_name, task_type)
        if key not in model_map:
            if task_type == 'classification':
                return RandomForestClassifier(**filtered_params) if filtered_params else RandomForestClassifier()
            return RandomForestRegressor(**filtered_params) if filtered_params else RandomForestRegressor()
        
        return model_map[key](**filtered_params)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练，请先训练模型")

        if self._is_lstm and self.lstm_predictor is not None:
            X_array = X.values.astype(np.float32)
            return self.lstm_predictor.predict(X_array)

        X_scaled = self.scaler.transform(X.fillna(X.median()))
        pred = self.model.predict(X_scaled)
        if self.task_type == 'classification' and self.label_encoder:
            pred = self.label_encoder.inverse_transform(pred.astype(int))
        return pred

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted or self._is_lstm:
            return {}
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}

    def save_model(self, path: str):
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

    def load_model(self, path: str):
        try:
            from models.lstm_model import LSTMPredictor
            if self.lstm_predictor is None:
                self.lstm_predictor = LSTMPredictor()
            self.lstm_predictor.load_model(path)
            self._is_lstm = True
            self.is_fitted = True
            self.metrics = self.lstm_predictor.metrics
            return
        except Exception:
            pass
        self._is_lstm = False
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data.get('label_encoder')
        self.task_type = data['task_type']
        self.feature_names = data['feature_names']
        self.metrics = data.get('metrics', {})
        self.is_fitted = True
