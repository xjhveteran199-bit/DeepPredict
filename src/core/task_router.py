"""
任务路由模块
根据用户需求描述和数据特征，自动选择合适的模型
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """任务配置"""
    task_type: str  # 'regression' | 'classification' | 'time_series'
    model_name: str
    model_params: Dict
    feature_engineering: List[str]


class TaskRouter:
    """任务路由器 - 分析需求并选择模型"""
    
    # 关键词匹配规则
    KEYWORD_PATTERNS = {
        'time_series': [
            r'时序', r'预测', r'forecast', r'未来', r'趋势', r'trend',
            r'时间序列', r'sequence', r'历史.*预测', r'销售预测',
            r'股票', r'价格预测', r'demand forecast', r'销量'
        ],
        'classification': [
            r'分类', r'classif', r'识别', r'detect', r'判断',
            r'是否', r'归于', r'判定', r'categor'
        ],
        'regression': [
            r'回归', r'regress', r'数值', r'预测值', r'estimat',
            r'连续值', r'产量', r'得分', r'评分'
        ]
    }
    
    # 模型候选池
    MODEL_POOL = {
        'regression': {
            'RandomForest': {
                'model': 'sklearn.ensemble.RandomForestRegressor',
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                '适用': '通用回归，中等规模数据'
            },
            'GradientBoosting': {
                'model': 'sklearn.ensemble.GradientBoostingRegressor',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                '适用': '高精度回归任务'
            },
            'LinearRegression': {
                'model': 'sklearn.linear_model.LinearRegression',
                'params': {},
                '适用': '线性关系明显的数据'
            },
            'XGBoost': {
                'model': 'xgboost.XGBRegressor',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
                '适用': '大规模数据，效果通常较好'
            }
        },
        'classification': {
            'RandomForest': {
                'model': 'sklearn.ensemble.RandomForestClassifier',
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                '适用': '通用分类'
            },
            'GradientBoosting': {
                'model': 'sklearn.ensemble.GradientBoostingClassifier',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                '适用': '高精度分类'
            },
            'LogisticRegression': {
                'model': 'sklearn.linear_model.LogisticRegression',
                'params': {'max_iter': 1000},
                '适用': '二分类，线性可分'
            },
            'XGBoost': {
                'model': 'xgboost.XGBClassifier',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
                '适用': '大规模分类'
            }
        },
        'time_series': {
            'PatchTST': {
                'model': 'PatchTST',
                'params': {
                    'seq_len': 96,
                    'pred_len': 96,
                    'patch_size': 16,
                    'd_model': 128,
                    'n_heads': 4,
                    'n_layers': 3,
                    'd_ff': 256,
                    'epochs': 30,
                    'batch_size': 32,
                    'learning_rate': 0.0005,
                },
                '适用': '⭐ Transformer时序预测，效果最强，适合长序列'
            },
            'LSTM': {
                'model': 'LSTM',
                'params': {'hidden_size': 64, 'num_layers': 2, 'epochs': 50, 'batch_size': 32, 'seq_len': 10, 'learning_rate': 0.001},
                '适用': '深度学习时序预测，效果较好'
            },
            'GradientBoosting': {
                'model': 'sklearn.ensemble.GradientBoostingRegressor',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                '适用': '时序预测，效果较好'
            },
            'RandomForest': {
                'model': 'sklearn.ensemble.RandomForestRegressor',
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                '适用': '时序预测，稳健'
            }
        }
    }
    
    def __init__(self):
        self.current_task: Optional[TaskConfig] = None
    
    def parse_requirement(self, requirement: str, data_info: Dict) -> TaskConfig:
        """
        解析用户需求，返回任务配置
        
        Args:
            requirement: 用户的需求描述（自然语言）
            data_info: 数据摘要信息
        """
        requirement = requirement.lower()
        task_type = self._detect_task_type(requirement, data_info)
        
        # 选择模型
        model_name, model_params = self._select_model(task_type, data_info)
        
        # 特征工程建议
        fe_suggestions = self._get_feature_suggestions(task_type, data_info)
        
        self.current_task = TaskConfig(
            task_type=task_type,
            model_name=model_name,
            model_params=model_params,
            feature_engineering=fe_suggestions
        )
        
        logger.info(f"任务解析结果: type={task_type}, model={model_name}")
        return self.current_task
    
    def _detect_task_type(self, requirement: str, data_info: Dict) -> str:
        """检测任务类型"""
        # 优先根据用户描述判断
        for task_type, patterns in self.KEYWORD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, requirement, re.IGNORECASE):
                    logger.info(f"根据关键词匹配到任务类型: {task_type}")
                    return task_type
        
        # 其次根据数据特征推断
        if data_info.get('numeric_cols') and data_info.get('categorical_cols'):
            # 有数值列和类别列，默认回归
            return 'regression'
        
        # 根据目标列推断
        if 'dtypes' in data_info:
            # 检查是否可能是时序
            if data_info.get('date_cols'):
                return 'time_series'
        
        return 'regression'  # 默认回归
    
    def _select_model(self, task_type: str, data_info: Dict) -> Tuple[str, Dict]:
        """选择最佳模型"""
        models = self.MODEL_POOL.get(task_type, self.MODEL_POOL['regression'])
        
        # 时序任务：数据量足够（>=200）优先 PatchTST，其次 LSTM
        if task_type == 'time_series':
            shape = data_info.get('shape', (0, 0))
            data_size = shape[0] if shape else 0
            
            if data_size >= 200 and 'PatchTST' in models:
                return 'PatchTST', models['PatchTST']['params']
            elif data_size >= 100 and 'LSTM' in models:
                return 'LSTM', models['LSTM']['params']
            elif 'GradientBoosting' in models:
                return 'GradientBoosting', models['GradientBoosting']['params']
            elif 'RandomForest' in models:
                return 'RandomForest', models['RandomForest']['params']
        
        # 回归/分类：优先 GradientBoosting > RandomForest > XGBoost > LinearRegression
        preferred_order = ['GradientBoosting', 'RandomForest', 'XGBoost', 'LinearRegression']
        
        for model_name in preferred_order:
            if model_name in models:
                return model_name, models[model_name]['params']
        
        # 取第一个
        name = list(models.keys())[0]
        return name, models[name]['params']
    
    def _get_feature_suggestions(self, task_type: str, data_info: Dict) -> List[str]:
        """获取特征工程建议"""
        suggestions = []
        
        shape = data_info.get('shape', (0, 0))
        if shape[0] < 100:
            suggestions.append("数据量较小，建议减少模型复杂度或增加数据")
        
        if shape[1] > 50:
            suggestions.append("特征较多，建议进行特征选择或降维")
        
        if task_type == 'time_series':
            suggestions.append("建议检查数据是否按时间排序")
            suggestions.append("可考虑添加时间特征（小时/星期/月份）")
        
        return suggestions
    
    def get_model_info(self, task_type: str) -> List[Dict]:
        """获取某任务类型的所有可用模型"""
        models = self.MODEL_POOL.get(task_type, {})
        return [
            {'name': name, **info}
            for name, info in models.items()
        ]
    
    def explain_task(self) -> str:
        """解释当前任务配置"""
        if not self.current_task:
            return "未配置任务"
        
        return f"""
📊 任务配置详情:
━━━━━━━━━━━━━━━━━━━━
• 任务类型: {self.current_task.task_type}
• 推荐模型: {self.current_task.model_name}
• 模型参数: {self.current_task.model_params}
• 特征工程建议: {', '.join(self.current_task.feature_engineering) if self.current_task.feature_engineering else '无特殊建议'}
━━━━━━━━━━━━━━━━━━━━
        """.strip()
