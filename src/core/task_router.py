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
            'CNN1D': {
                'model': 'CNN1D',
                'params': {
                    'seq_len': 100,
                    'hidden_channels': [32, 64, 128],
                    'kernel_size': 3,
                    'epochs': 50,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                },
                '适用': '⭐ CNN时序预测，擅 长局部特征提取，适合传感器信号'
            },
            'LSTM': {
                'model': 'LSTM',
                'params': {'hidden_size': 64, 'num_layers': 2, 'epochs': 50, 'batch_size': 32, 'seq_len': 30, 'learning_rate': 0.001},
                '适用': '深度学习时序预测，效果较好'
            },
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
                '适用': 'Transformer时序预测，适合长序列'
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
        },
        'decouple': {
            'FastICA': {
                'model': 'FastICA',
                'params': {'n_components': None, 'max_iter': 500},
                '适用': '线性解耦，适合多通道混合信号分离'
            },
            'AutoEncoder': {
                'model': 'AutoEncoder',
                'params': {'hidden_dim': 64, 'latent_dim': None, 'epochs': 50, 'seg_len': 100},
                '适用': '非线性解耦，适合复杂混合信号'
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
        
        # 时序任务：数据量足够（>=200）优先 CNN1D 或 LSTM
        if task_type == 'time_series':
            shape = data_info.get('shape', (0, 0))
            data_size = shape[0] if shape else 0
            
            if data_size >= 200 and 'CNN1D' in models:
                return 'CNN1D', models['CNN1D']['params']
            elif data_size >= 100 and 'LSTM' in models:
                return 'LSTM', models['LSTM']['params']
            elif data_size >= 200 and 'PatchTST' in models:
                return 'PatchTST', models['PatchTST']['params']
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

    def recommend_model(self, data_info: Dict, df=None) -> Dict:
        """
        基于数据分析自动推荐最适配的模型

        分析维度：
        - 样本量 → 模型复杂度上限
        - 特征数 → 单变量/多变量
        - 缺失率 → 数据质量
        - 季节性检测 → 是否需要时序专属模型
        - 趋势复杂度 → 简单趋势/复杂非线性

        Returns:
            {
                'model': str,
                'reason': str,
                'confidence': float,  # 0-1
                'alternatives': list,
                'data_insights': list,
                'params': dict
            }
        """
        import numpy as np

        insights = []
        shape = data_info.get('shape', (0, 0))
        n_samples = shape[0]
        n_features = max(shape[1] - 1, 1)
        numeric_cols = data_info.get('numeric_cols', [])
        is_multivariate = n_features > 1
        missing = data_info.get('missing', {})

        # 缺失率分析
        total_possible = n_samples * shape[1] if shape[1] > 0 else 1
        total_missing = sum(missing.values()) if missing else 0
        missing_rate = total_missing / total_possible
        if missing_rate > 0.1:
            insights.append(f"⚠️ 数据缺失率 {missing_rate:.1%}，建议预处理填充")
        elif missing_rate > 0:
            insights.append(f"✅ 数据缺失率低 ({missing_rate:.1%})")

        has_seasonality = False
        has_trend = False

        if df is not None and len(df) > 50 and numeric_cols:
            target_col = numeric_cols[0]
            series = df[target_col].dropna()
            if len(series) > 50:
                try:
                    mid = len(series) // 2
                    if abs(series[:mid].mean() - series[mid:].mean()) > series.std() * 0.2:
                        has_trend = True
                        insights.append(f"📈 检测到趋势性（均值变化显著）")

                    for lag in [7, 12, 24, 52]:
                        if len(series) > lag * 2:
                            vals1 = series.values[:-lag]
                            vals2 = series.values[lag:]
                            std1, std2 = np.nanstd(vals1), np.nanstd(vals2)
                            if len(vals1) > 1 and len(vals2) > 1 and std1 > 0 and std2 > 0:
                                ac = np.corrcoef(vals1, vals2)[0, 1]
                                if not np.isnan(ac) and abs(ac) > 0.5:
                                    has_seasonality = True
                                    insights.append(f"🔄 检测到季节性（lag={lag}, r={ac:.2f}）")
                                    break
                except Exception:
                    pass

        if not has_seasonality and not has_trend:
            insights.append(f"📊 数据模式较平稳，无明显季节性/趋势")

        # ========== 模型推荐 ==========

        # 1. 数据量极小
        if n_samples < 200:
            insights.append(f"📦 数据量小（{n_samples}），深度学习易过拟合")
            return {
                'model': 'GradientBoosting',
                'reason': f'数据量{n_samples}条，深度学习易过拟合。GradientBoosting稳健且无需大量数据。',
                'confidence': 0.85,
                'alternatives': ['RandomForest', 'LSTM（需增加数据）'],
                'data_insights': insights,
                'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
            }

        # 2. 多变量
        if is_multivariate and n_samples >= 200:
            if has_trend or has_seasonality:
                insights.append(f"🧠 多变量+时序特征 → CNN1D多变量版")
                return {
                    'model': 'CNN1D',
                    'reason': f'多变量（{n_features}个特征）+ 检测到时序特征，CNN1D可同时处理多变量和时序模式。',
                    'confidence': 0.75,
                    'alternatives': ['GradientBoosting', 'LSTM'],
                    'data_insights': insights,
                    'params': {'seq_len': min(96, n_samples // 10), 'pred_len': min(24, n_samples // 20),
                               'hidden_channels': 64, 'num_layers': 2, 'epochs': 30}
                }
            else:
                insights.append(f"📊 多变量静态关系 → GradientBoosting")
                return {
                    'model': 'GradientBoosting',
                    'reason': f'多变量（{n_features}个特征）无明显时序特征，GradientBoosting擅长捕捉变量间静态关系。',
                    'confidence': 0.80,
                    'alternatives': ['RandomForest', 'XGBoost'],
                    'data_insights': insights,
                    'params': {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1}
                }

        # 3. 单变量 + 季节性/趋势
        if not is_multivariate and n_samples >= 200:
            if has_seasonality and has_trend:
                insights.append(f"🔄 季节性+趋势 → PatchTST")
                return {
                    'model': 'PatchTST',
                    'reason': f'检测到季节性+趋势，PatchTST的Transformer架构对长程依赖建模能力强。',
                    'confidence': 0.80,
                    'alternatives': ['CNN1D', 'LSTM'],
                    'data_insights': insights,
                    'params': {'seq_len': 96, 'pred_len': 48, 'patch_size': 16,
                               'd_model': 128, 'n_heads': 4, 'n_layers': 3, 'epochs': 30}
                }
            if has_seasonality:
                insights.append(f"🔄 季节性时序 → CNN1D")
                return {
                    'model': 'CNN1D',
                    'reason': f'检测到季节性，CNN1D Patch架构擅长提取局部周期模式。',
                    'confidence': 0.78,
                    'alternatives': ['PatchTST', 'LSTM'],
                    'data_insights': insights,
                    'params': {'seq_len': min(96, n_samples // 10), 'pred_len': min(24, n_samples // 20),
                               'hidden_channels': 128, 'num_layers': 3, 'epochs': 50}
                }
            if has_trend:
                insights.append(f"📈 趋势性时序 → LSTM")
                return {
                    'model': 'LSTM',
                    'reason': f'检测到趋势，LSTM的记忆单元擅长处理长期趋势依赖。',
                    'confidence': 0.75,
                    'alternatives': ['CNN1D', 'GradientBoosting'],
                    'data_insights': insights,
                    'params': {'hidden_size': 64, 'num_layers': 2, 'seq_len': min(30, n_samples // 20),
                               'epochs': 50, 'learning_rate': 0.001}
                }
            insights.append(f"📊 平稳时序 → CNN1D（默认最优）")
            return {
                'model': 'CNN1D',
                'reason': f'数据无明显季节性/趋势，CNN1D适合快速建模。',
                'confidence': 0.70,
                'alternatives': ['GradientBoosting', 'LSTM'],
                'data_insights': insights,
                'params': {'seq_len': min(48, n_samples // 10), 'pred_len': min(12, n_samples // 20),
                           'hidden_channels': 64, 'num_layers': 2, 'epochs': 30}
            }

        return {
            'model': 'GradientBoosting',
            'reason': '基于数据规模和特征综合评估',
            'confidence': 0.60,
            'alternatives': ['CNN1D', 'RandomForest'],
            'data_insights': insights,
            'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
        }

    def explain_recommendation(self, rec: Dict) -> str:
        """格式化推荐结果"""
        stars = '★' * int(rec['confidence'] * 5) + '☆' * (5 - int(rec['confidence'] * 5))
        lines = [
            f"",
            f"🤖 AI 模型推荐",
            f"{'━' * 36}",
            f"✅ 推荐模型：{rec['model']}",
            f"   置信度：{stars} ({rec['confidence']:.0%})",
            f"",
            f"📝 推荐理由：",
            f"   {rec['reason']}",
            f"",
            f"💡 数据洞察：",
        ]
        for insight in rec.get('data_insights', []):
            lines.append(f"   {insight}")
        lines.append(f"")
        lines.append(f"🔄 备选模型：")
        for alt in rec.get('alternatives', []):
            lines.append(f"   • {alt}")
        return '\n'.join(lines)
