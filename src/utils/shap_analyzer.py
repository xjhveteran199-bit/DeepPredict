"""
SHAP 可解释性分析模块
基于 SHAP (SHapley Additive exPlanations) 提供特征重要性分析
参考：adi6492 论文中的 SHAP 决策解释思路
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP 特征重要性分析器

    支持：
    - Tree-based 模型（RandomForest, GradientBoosting）的精确 SHAP
    - 任意模型的 Kernel SHAP（model-agnostic）
    - 特征重要性排序
    - SHAP 决策图 / beeswarm 图
    """

    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = None
        self.is_fitted = False
        self._method = None  # 'tree' or 'kernel'

    def fit(self, model, X: pd.DataFrame, method: str = 'auto') -> bool:
        """
        计算 SHAP 值

        Args:
            model: 训练好的 sklearn/PyTorch 模型
            X: 特征矩阵 (DataFrame)
            method: 'auto' / 'tree' / 'kernel'
                - 'tree': 使用 TreeExplainer（sklearn 树模型专用，快）
                - 'kernel': 使用 KernelExplainer（任意模型，通用但慢）
                - 'auto': 自动选择

        Returns:
            是否成功
        """
        try:
            self.feature_names = list(X.columns)
            X_array = X.values.astype(np.float32)

            # 自动选择方法
            if method == 'auto':
                method = self._detect_method(model)

            self._method = method
            logger.info(f"SHAP 分析: 使用 {method} 方法")

            if method == 'tree':
                import shap
                # TreeExplainer 适用于 sklearn 树模型
                self.explainer = shap.TreeExplainer(model)
                self.shap_values = self.explainer.shap_values(X_array)
                self.expected_value = self.explainer.expected_value

                # 处理分类问题的 shap_values 形状
                if isinstance(self.shap_values, list):
                    # 多分类：取正类的 SHAP（通常是类别1）
                    self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
                if isinstance(self.expected_value, list):
                    self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]

            elif method == 'kernel':
                import shap

                # 获取模型预测函数
                if hasattr(model, 'predict_proba'):
                    def predict_fn(x):
                        prob = model.predict_proba(x)
                        return prob[:, 1] if prob.shape[1] > 1 else prob.ravel()
                else:
                    predict_fn = model.predict

                # 使用数据集的均值作为背景数据
                background = shap.sample(X_array, 100, random_state=42)
                self.explainer = shap.KernelExplainer(predict_fn, background)
                self.shap_values = self.explainer.shap_values(X_array)

                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
                self.expected_value = self.explainer.expected_value

            else:
                raise ValueError(f"Unknown method: {method}")

            self.is_fitted = True
            logger.info(f"SHAP 计算完成: shape={self.shap_values.shape}")
            return True

        except ImportError as e:
            logger.warning(f"SHAP 库未安装: {e}")
            return False
        except Exception as e:
            logger.error(f"SHAP 分析失败: {e}")
            return False

    def _detect_method(self, model) -> str:
        """检测模型类型，返回合适的 SHAP 方法"""
        model_name = type(model).__name__.lower()
        tree_models = ['randomforestregressor', 'randomforestclassifier',
                       'gradientboostingregressor', 'gradientboostingclassifier',
                       'xgbregressor', 'xgbclassifier', 'lgbmregressor', 'lgbmclassifier',
                       'decisiontreeregressor', 'decisiontreeclassifier']
        if any(t in model_name for t in tree_models):
            return 'tree'
        return 'kernel'

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性（按 SHAP 绝对值均值排序）

        Returns:
            dict: {特征名: 重要性分数}
        """
        if not self.is_fitted or self.shap_values is None:
            return {}

        # 计算每个特征的平均绝对 SHAP 值
        importance = np.abs(self.shap_values).mean(axis=0)
        result = dict(zip(self.feature_names, importance))

        # 归一化到 [0, 1]
        max_val = max(result.values()) if result else 1
        result = {k: round(v / max_val, 4) for k, v in result.items()}

        # 按重要性降序排列
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return result

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取 Top-N 最重要特征"""
        importance = self.get_feature_importance()
        return list(importance.items())[:n]

    def get_local_explanation(self, instance_idx: int = 0) -> Dict[str, float]:
        """
        获取单个样本的 SHAP 解释

        Args:
            instance_idx: 样本索引

        Returns:
            dict: {特征名: SHAP值}
        """
        if not self.is_fitted or self.shap_values is None:
            return {}

        if instance_idx >= len(self.shap_values):
            instance_idx = 0

        shap_row = self.shap_values[instance_idx]
        result = dict(zip(self.feature_names, shap_row))
        # 按绝对值降序
        result = dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))
        return result

    def plot_importance(self, save_path: str = None, figsize: tuple = (10, 6)) -> plt.Figure:
        """
        绘制特征重要性条形图

        Args:
            save_path: 保存路径
            figsize: 图形尺寸
        """
        if not self.is_fitted:
            logger.warning("SHAP 未拟合，无法绘图")
            return None

        importance = self.get_feature_importance()
        if not importance:
            return None

        fig, ax = plt.subplots(figsize=figsize)
        features = list(importance.keys())[:15]  # 最多显示15个
        values = [importance[f] for f in features]

        ax.barh(range(len(features)), values, color='steelblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value| (normalized)')
        ax.set_title('SHAP Feature Importance (Top 15)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_beeswarm(self, save_path: str = None, figsize: tuple = (12, 8),
                      style: str = 'nature', dpi: int = 300) -> plt.Figure:
        """
        绘制 beeswarm 图（展示每个特征对预测的影响分布）

        Args:
            save_path: 保存路径
            figsize: 图形尺寸
            style: scienceplots 样式 ('nature', 'science', 'ieee')
            dpi: 导出分辨率

        Returns:
            matplotlib Figure
        """
        if not self.is_fitted:
            return None

        try:
            import shap
            import matplotlib
            # 保存当前样式
            old_style = plt.get_fignums()
            old_rc = matplotlib.rcParams.copy()

            # 应用 scienceplots 子刊风格
            try:
                import scienceplots
                plt.style.use(['science', style])
            except Exception:
                plt.style.use(['ggplot'])

            fig, ax = plt.subplots(figsize=figsize)
            shap.summary_plot(
                self.shap_values,
                feature_names=self.feature_names,
                show=False,
                max_display=min(15, len(self.feature_names))
            )
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')

            # 恢复原始样式
            matplotlib.rcParams.update(old_rc)
            return fig
        except Exception as e:
            logger.warning(f"Beeswarm 图绘制失败: {e}")
            # 确保恢复样式
            try:
                matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            except Exception:
                pass
            return None

    def plot_decision(self, instance_idx: int = 0, save_path: str = None,
                      figsize: tuple = (14, 6)) -> plt.Figure:
        """
        绘制 SHAP 决策图（参考 adi6492 论文中的决策解释图）

        展示单个样本的 SHAP 值累积过程

        Args:
            instance_idx: 样本索引
            save_path: 保存路径
            figsize: 图形尺寸
        """
        if not self.is_fitted:
            return None

        try:
            import shap
            fig, ax = plt.subplots(figsize=figsize)
            shap.decision_plot(
                self.expected_value if not isinstance(self.expected_value, list) else self.expected_value[0],
                self.shap_values[instance_idx:instance_idx+1],
                feature_names=self.feature_names,
                show=False,
                auto_size=False
            )
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            return fig
        except Exception as e:
            logger.warning(f"决策图绘制失败: {e}")
            return None

    def generate_report(self) -> str:
        """
        生成 SHAP 分析报告文本

        Returns:
            报告字符串（Markdown 格式）
        """
        if not self.is_fitted:
            return "❌ SHAP 未拟合"

        importance = self.get_feature_importance()
        if not importance:
            return "❌ 无法计算特征重要性"

        lines = [
            "## 🔍 SHAP 特征重要性分析报告",
            "",
            f"**分析方法**: {self._method}",
            f"**样本数**: {self.shap_values.shape[0]}",
            f"**特征数**: {self.shap_values.shape[1]}",
            "",
            "### Top 10 最重要特征",
            "",
            "| 排名 | 特征 | 重要性 |",
            "|:---:|:---|:---:|",
        ]

        for i, (feat, val) in enumerate(list(importance.items())[:10], 1):
            bar = "█" * int(val * 20)
            lines.append(f"| {i} | `{feat}` | {val:.4f} {bar} |")

        lines.append("")
        lines.append("### 结论")
        top3 = list(importance.items())[:3]
        lines.append(f"- 最主要的预测因素是 **{top3[0][0]}**（重要性 {top3[0][1]:.4f}）")
        if len(top3) > 1:
            lines.append(f"- 次要因素：**{top3[1][0]}**（重要性 {top3[1][1]:.4f}）")
        if len(top3) > 2:
            lines.append(f"- 第三因素：**{top3[2][0]}**（重要性 {top3[2][1]:.4f}）")

        return "\n".join(lines)

    def save(self, path: str):
        """保存 SHAP 分析结果"""
        import joblib
        joblib.dump({
            'shap_values': self.shap_values,
            'expected_value': self.expected_value,
            'feature_names': self.feature_names,
            'method': self._method,
            'is_fitted': self.is_fitted,
        }, path)

    def load(self, path: str):
        """加载 SHAP 分析结果"""
        import joblib
        data = joblib.load(path)
        self.shap_values = data['shap_values']
        self.expected_value = data['expected_value']
        self.feature_names = data['feature_names']
        self._method = data['method']
        self.is_fitted = data['is_fitted']
