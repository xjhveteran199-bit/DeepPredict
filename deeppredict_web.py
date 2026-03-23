"""
DeepPredict Web 版 - Gradio 界面 v1.04
改进：智能数据分析 + 用户引导
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============ 核心模块 ============

from src.data.data_decoupler import DataDecoupler

class DataLoader:
    def __init__(self):
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.data_structure = None  # 存储数据分析结果
    
    def load_csv(self, file_path):
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            for enc in encodings:
                try:
                    self.df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if self.df is None:
                self.df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            self._analyze()
            return True, f"✅ 加载成功：{self.df.shape[0]}行 × {self.df.shape[1]}列"
        except Exception as e:
            return False, f"❌ {str(e)}"
    
    def _analyze(self):
        """分析数据结构"""
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(self.df.select_dtypes(include=['object']).columns)
        
        # 分析数据结构
        self.data_structure = self._detect_structure()
    
    def _detect_structure(self):
        """检测数据结构类型"""
        n_num = len(self.numeric_cols)
        n_cat = len(self.categorical_cols)
        n_rows = len(self.df)
        
        # 检查是否有重复的时间列（可能是宽表格式的多组时序）
        time_cols = [c for c in self.numeric_cols if 'time' in c.lower() or '日期' in c.lower() or 'date' in c.lower()]
        
        # 检查是否有类似 "K-with", "K-without", "Glu-with" 这种成组的列名
        data_cols = [c for c in self.numeric_cols if c not in time_cols]
        
        # 成组分析：检查是否有相同前缀
        groups = {}
        for col in data_cols:
            # 提取前缀（-前的部分）
            parts = col.split('-')[0] if '-' in col else col
            if parts not in groups:
                groups[parts] = []
            groups[parts].append(col)
        
        structure = {
            'n_rows': n_rows,
            'n_cols': len(self.df.columns),
            'n_numeric': n_num,
            'n_categorical': n_cat,
            'time_cols': time_cols,
            'data_cols': data_cols,
            'groups': groups,
        }
        
        # 判断类型
        if n_num == 0:
            structure['type'] = 'no_numeric'
        elif n_num == 1 or (n_num == 2 and n_cat >= 1):
            structure['type'] = 'single_variable'  # 单变量时序
        elif len(groups) > 1 and all(len(v) >= 2 for v in groups.values()):
            structure['type'] = 'grouped_time_series'  # 成组时序（如 K-xxx, Glu-xxx）
        elif n_num >= 2:
            structure['type'] = 'multi_variable'  # 多变量（特征→目标）
        else:
            structure['type'] = 'unknown'
        
        return structure
    
    def get_info(self):
        if self.df is None:
            return ""
        return (
            f"**数据形状**：{self.df.shape[0]} 行 × {self.df.shape[1]} 列\n\n"
            f"**数值列**（{len(self.numeric_cols)}）：`{', '.join(self.numeric_cols[:5])}{'...' if len(self.numeric_cols)>5 else ''}`\n\n"
            f"**类别列**（{len(self.categorical_cols)}）：`{', '.join(self.categorical_cols[:5])}{'...' if len(self.categorical_cols)>5 else ''}`"
        )
    
    def get_structure_explanation(self):
        """返回数据结构的中文解释"""
        if self.data_structure is None:
            return "请先上传数据"
        
        s = self.data_structure
        lines = []
        
        lines.append(f"**📊 数据结构检测结果**：")
        lines.append("")
        
        if s['type'] == 'single_variable':
            lines.append("✅ **单变量时序数据**（推荐使用 PatchTST/LSTM）")
            lines.append(f"   - 数据点：{s['n_rows']} 条")
            lines.append(f"   - 数值列：{s['data_cols']}")
            lines.append("")
            lines.append("**使用建议**：用该列自己的历史值预测未来值")
        
        elif s['type'] == 'grouped_time_series':
            lines.append("⚠️ **成组时序数据**")
            lines.append(f"   - 数据点：{s['n_rows']} 条")
            lines.append(f"   - 检测到 {len(s['groups'])} 组数据：")
            for group, cols in s['groups'].items():
                lines.append(f"     • {group}: {cols}")
            lines.append("")
            lines.append("**使用建议**：")
            lines.append("   1. 选择其中一组的目标列（如 K-with epifluidics）")
            lines.append("   2. 系统会用该组自己的历史值预测未来")
            lines.append("   3. 同一组内的其他列（如 K-without）可用于多变量预测")
        
        elif s['type'] == 'multi_variable':
            lines.append("📈 **多变量数据**")
            lines.append(f"   - 数据点：{s['n_rows']} 条")
            lines.append(f"   - 数值列：{s['data_cols']}")
            lines.append("")
            lines.append("**使用建议**：选择一个目标列，其他列作为特征")
        
        else:
            lines.append(f"**数据类型**：{s['type']}")
            lines.append(f"**数值列**：{s['numeric_cols']}")
        
        return "\n".join(lines)


class TaskRouter:
    PATTERNS = {
        'time_series': ['时序', '预测', 'forecast', '未来', '趋势', '销售预测', '销量', '流量', '股票', '变化', '走势'],
        'classification': ['分类', '判断', '识别', '是否', '流失', '垃圾邮件', '检测'],
        'regression': ['回归', '数值', '预测值', '产量', '得分', '销售额', '价格']
    }
    
    def parse(self, req, data_info):
        req_lower = req.lower()
        for task_type, keywords in self.PATTERNS.items():
            if any(kw.lower() in req_lower for kw in keywords):
                return task_type
        return 'regression'
    
    def select_model(self, task_type, data_size):
        if task_type == 'time_series':
            if data_size >= 200:
                return 'PatchTST', {'seq_len': 96, 'pred_len': 96, 'patch_size': 16, 'd_model': 128, 'n_heads': 4, 'n_layers': 3, 'd_ff': 256, 'epochs': 30, 'batch_size': 32, 'learning_rate': 0.0005}
            elif data_size >= 100:
                return 'LSTM', {'hidden_size': 64, 'num_layers': 2, 'epochs': 50, 'seq_len': 10}
            return 'GradientBoosting', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
        return 'GradientBoosting', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}


class SklearnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.task_type = None
        self.feature_names = []
        self.is_fitted = False
        self.metrics = {}
    
    def train(self, X, y, task_type, model_name, params):
        try:
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import (
                RandomForestRegressor, RandomForestClassifier,
                GradientBoostingRegressor, GradientBoostingClassifier
            )
            from sklearn.linear_model import LinearRegression, LogisticRegression
            
            self.task_type = task_type
            self.feature_names = list(X.columns)
            
            y_arr = np.asarray(y).ravel()
            if y_arr.ndim != 1:
                return False, f"❌ y 必须是 1D 数组，当前 shape={np.asarray(y).shape}"
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X.fillna(X.median()))
            
            if task_type == 'classification':
                self.label_encoder = LabelEncoder()
                y_enc = self.label_encoder.fit_transform(y_arr.astype(str))
            else:
                y_enc = y_arr.astype(float)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_enc, test_size=0.2, random_state=42
            )
            
            models = {
                ('RandomForest', 'regression'): RandomForestRegressor,
                ('RandomForest', 'classification'): RandomForestClassifier,
                ('GradientBoosting', 'regression'): GradientBoostingRegressor,
                ('GradientBoosting', 'classification'): GradientBoostingClassifier,
                ('LinearRegression', 'regression'): LinearRegression,
                ('LogisticRegression', 'classification'): LogisticRegression,
            }
            
            key = (model_name, task_type)
            model_cls = models.get(key, GradientBoostingRegressor)
            self.model = model_cls(**(params or {}))
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            
            if task_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                self.metrics = {'准确率': acc, '精确率': prec, '召回率': rec, 'F1': f1}
                m_str = f"**准确率**={acc:.2%} **精确率**={prec:.2%} **F1**={f1:.4f}"
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                self.metrics = {'R²': r2, 'RMSE': rmse, 'MAE': mae}
                m_str = f"**R²**={r2:.4f} **RMSE**={rmse:.4f} **MAE**={mae:.4f}"
            
            self.is_fitted = True
            return True, f"✅ {model_name} 训练完成\n{m_str}"
            
        except Exception as e:
            return False, f"❌ {str(e)}"
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("请先训练模型")
        X_scaled = self.scaler.transform(X.fillna(X.median()))
        pred = self.model.predict(X_scaled)
        if self.task_type == 'classification' and self.label_encoder:
            pred = self.label_encoder.inverse_transform(pred.astype(int))
        return pred
    
    def get_importance(self):
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def predict_future(self, last_X: np.ndarray, steps: int = 30) -> np.ndarray:
        """
        滚动预测未来 steps 步（用于时序预测）
        last_X: 最后一个时间点的特征向量 (n_features,)
        返回: (steps,) 预测值
        """
        if not self.is_fitted:
            raise ValueError("请先训练模型")
        preds = []
        x_cur = last_X.copy()
        for _ in range(steps):
            try:
                x_scaled = self.scaler.transform(x_cur.reshape(1, -1))
                p = self.model.predict(x_scaled)[0]
                preds.append(p)
                # 更新：如果是单变量时序（feature_names 只有一列），直接替换
                # 如果是多变量，需要把预测值更新回去（这里简化处理：假设单变量）
                if len(self.feature_names) == 1:
                    x_cur[0] = p
            except Exception:
                break
        return np.array(preds) if preds else np.array([])


# ============ 全局状态 ============
data_loader = DataLoader()
data_decoupler = None  # 数据解耦器
task_router = TaskRouter()
predictor = None
lstm_pred = None


# ============ 图表模板函数 ============

def select_plot_function(chart_requirement, hist, future_preds, target_col, steps_hist, steps_fut, std_val=None):
    """根据用户描述选择图表模板（纯规则匹配，无需 AI）"""
    if std_val is None:
        std_val = np.std(future_preds) * 0.5
    req = (chart_requirement or "").lower()
    if any(kw in req for kw in ["双轴", "双y", "次坐标", "dual", "twin"]):
        return plot_dual_axis(hist, future_preds, target_col, steps_hist, steps_fut)
    elif any(kw in req for kw in ["置信", "误差", "上下界", "区间", "confidence", "band", "uncertainty"]):
        return plot_with_confidence_band(hist, future_preds, target_col, steps_hist, steps_fut, std_val)
    elif any(kw in req for kw in ["散点", "scatter"]):
        return plot_scatter_with_line(hist, future_preds, target_col, steps_hist, steps_fut)
    elif any(kw in req for kw in ["柱状", "bar", "柱形"]):
        return plot_bar_forecast(hist, future_preds, target_col, steps_hist, steps_fut)
    else:
        return plot_standard_forecast(hist, future_preds, target_col, steps_hist, steps_fut)


def plot_standard_forecast(hist, future_preds, target_col, steps_hist, steps_fut):
    """标准折线图：蓝色历史 + 橙色预测"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=2, label='历史数据')
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='预测')
    ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel(target_col, fontsize=11)
    ax.set_title(f'{target_col} 趋势预测', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_dual_axis(hist, future_preds, target_col, steps_hist, steps_fut):
    """双轴图：左轴历史，右轴预测（用于量纲不同场景）"""
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 4))
    color1, color2 = '#3B82F6', '#FF6B2B'
    ax1.set_xlabel('时间步', fontsize=11)
    ax1.set_ylabel(f'{target_col} 历史', color=color1, fontsize=11)
    ax1.plot(steps_hist, hist, color=color1, linewidth=2, label='历史数据')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'{target_col} 预测', color=color2, fontsize=11)
    ax2.plot(steps_fut, future_preds, color=color2, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.set_title(f'{target_col} 双轴趋势预测', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_with_confidence_band(hist, future_preds, target_col, steps_hist, steps_fut, std_val=None):
    """带置信区间的预测图"""
    import matplotlib.pyplot as plt
    if std_val is None:
        std_val = np.std(future_preds) * 0.5
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=2, label='历史数据')
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='预测')
    ax.fill_between(steps_fut,
                    [v - 1.96 * std_val for v in future_preds],
                    [v + 1.96 * std_val for v in future_preds],
                    color='#FF6B2B', alpha=0.2, label='95%置信区间')
    ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel(target_col, fontsize=11)
    ax.set_title(f'{target_col} 趋势预测（含置信区间）', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scatter_with_line(hist, future_preds, target_col, steps_hist, steps_fut):
    """散点+折线组合图"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(steps_hist, hist, color='#3B82F6', s=20, zorder=3, label='历史数据（散点）')
    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=1.5, alpha=0.6)
    ax.scatter(steps_fut, future_preds, color='#FF6B2B', s=30, marker='D', zorder=3, label='预测（散点）')
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--')
    ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel(target_col, fontsize=11)
    ax.set_title(f'{target_col} 趋势预测（散点图）', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bar_forecast(hist, future_preds, target_col, steps_hist, steps_fut):
    """柱状图（历史用柱状，预测用折线）"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.8
    ax.bar(steps_hist, hist, color='#3B82F6', width=bar_width, label='历史数据', alpha=0.8)
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='预测', marker='o', markersize=4)
    ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel(target_col, fontsize=11)
    ax.set_title(f'{target_col} 趋势预测（柱状图）', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


# ============ Gradio 界面 ============

with gr.Blocks(title="DeepPredict v1.04 - 图表定制+下载版") as demo:
    
    # 首页介绍
    with gr.Tab("🏠 首页介绍"):
        gr.Markdown("""
        # 🧠 DeepPredict - 零门槛深度学习预测工具
        
        ### 告别复杂代码，3步完成AI预测！
        
        ---
        
        ## ✨ 核心优势
        
        | 功能 | 说明 |
        |------|------|
        | 🤖 AI自动选模型 | 上传数据后，系统自动推荐最优模型（PatchTST/LSTM/GradientBoosting） |
        | 📊 零代码操作 | 无需编程基础，点点鼠标即可完成时序预测 |
        | 🔒 数据本地处理 | 数据不上传服务器，保护隐私安全 |
        | 📱 支持手机访问 | 响应式设计，随时随地使用 |
        
        ---
        
        ## 🚀 快速开始
        
        1. 点击「**工具使用**」标签页
        2. 上传你的 CSV 数据文件
        3. 选择目标列，点击「开始训练」
        
        ---
        
        ## 💡 适用场景
        
        - 🧬 **生物医学**：细胞培养数据、药物反应预测
        - 📈 **市场分析**：销售预测、流量预测  
        - 🌡️ **环境科学**：空气质量、气候变化预测
        - 🏭 **工业生产**：设备故障预测、质量控制
        
        ---
        
        ## 💰 定价方案
        
        | 版本 | 价格 | 说明 |
        |------|------|------|
        | 个人版 | **免费** | 适合学习和研究 |
        | 预测服务 | **¥5/次** | 单次预测，不限数据量 |
        | 批量服务 | **¥99/月** | 无限次预测 + 优先模型 |
        
        ---
        
        *如有疑问或定制需求，请联系开发者*
        """)
    
    # 工具使用标签
    with gr.Tab("🛠️ 工具使用"):
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## 📂 数据导入与配置")
            
            with gr.Row():
                # === 上传和数据预览 ===
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ 上传数据文件")
                    file_input = gr.File(label="点击上传 CSV 文件", file_types=[".csv"], height=100)
                    
                    gr.Markdown("### 📊 数据预览")
                    data_preview = gr.HTML(label="")
                
                # === 数据分析 ===
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 数据结构分析")
                    data_structure_info = gr.Markdown("**上传数据后自动分析...**")
                    
                    gr.Markdown("### 🔬 数据解耦")
                    data_decouple_info = gr.Markdown("**上传后自动识别列类型...**")
                    
                    gr.Markdown("### 📈 数据摘要")
                    data_info = gr.Markdown("**上传数据后显示摘要**")
            
            gr.Markdown("---")
            
            with gr.Row():
                # === 任务配置 ===
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 任务配置")
                    
                    feature_col = gr.Dropdown(
                        label="① 选择特征列 X（自变量/时间列，可选）",
                        choices=[],
                        info="选择作为时间轴或自变量的列（如日期、序号），不选则自动用行号"
                    )
                    
                    target_col = gr.Dropdown(
                        label="② 选择目标列 Y（要预测什么）",
                        choices=[],
                        info="选择你要预测的目标变量"
                    )
                    
                    predict_mode = gr.Radio(
                        choices=["单变量时序（推荐）", "多变量预测"],
                        value="单变量时序（推荐）",
                        label="③ 预测模式",
                        info="时序数据用单变量，分类/回归可用多变量"
                    )
                    
                    model_select = gr.Dropdown(
                        label="④ 选择模型",
                        choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"],
                        value="自动推荐",
                        info="自动推荐会根据数据情况选择最优模型"
                    )
                    model_recommend = gr.Markdown("")
                    
                    requirement = gr.Textbox(
                        label="⑤ 描述需求（可选）",
                        placeholder="示例：预测K值变化趋势\n预测Glu未来走势\n判断是否流失",
                        lines=2
                    )
                    
                    n_future = gr.Number(
                        label="⑥ 预测未来多少步",
                        value=30,
                        info="训练完成后自动预测未来N步的值，默认30步",
                        precision=0
                    )

                    chart_requirement = gr.Textbox(
                        label="⑦ 描述你想要的图表（可选）",
                        placeholder="示例：双轴图，左边显示温度走势，右边显示湿度\n用红色虚线标注预测区间上下界\n标注关键的拐点和异常值",
                        lines=2
                    )

                # === 结果展示 ===
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 任务配置预览")
                    config_out = gr.Markdown("*配置信息将显示在这里*")
                    
                    gr.Markdown("### 📊 训练结果（模型评价）")
                    result_out = gr.Textbox(label="", lines=6, interactive=False)
                    
                    gr.Markdown("### 📈 未来趋势预测")
                    forecast_plot = gr.Plot(label="趋势预测图（蓝色=历史，橙色=预测）")
                    
                    gr.Markdown("### 🔮 未来预测值")
                    forecast_out = gr.Textbox(label="", lines=10, interactive=False)
                    
                    gr.Markdown("### 💬 趋势结论")
                    summary_out = gr.Markdown("*训练完成后自动生成趋势总结*")
                    
                    gr.Markdown("### 🔍 特征重要性")
                    importance_out = gr.Markdown("")

                    gr.Markdown("### 📥 下载完整结果包")
                    with gr.Row():
                        download_btn = gr.Button("📥 下载结果包（PNG + CSV + JSON）", variant="secondary", size="lg", scale=2)
                        download_file = gr.File(label="点击下载 zip", interactive=False, scale=1)

            gr.Markdown("---")

            with gr.Row():
                train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg", scale=1)
            
            gr.Markdown("---")
            
            with gr.Row():
                gr.Markdown("### 🔮 新数据预测")
                predict_file = gr.File(label="上传新数据（可选）", file_types=[".csv"], scale=1)
                predict_btn = gr.Button("执行预测", variant="secondary", scale=0)
            
            predict_out = gr.HTML(label="预测结果")
            predict_status = gr.Textbox(label="状态", lines=2, interactive=False)
    
    # ========== 事件处理 ==========
    
    def on_file_upload(file):
        global data_loader, predictor, lstm_pred, data_decoupler
        predictor = None
        lstm_pred = None
        data_decoupler = None
        
        if file is None:
            return [None] * 7 + [
                "**请上传 CSV 文件**",
                "**上传数据后自动分析**",
                "**上传后自动识别列类型**",
                gr.update(choices=[]),
                gr.update(choices=[]),
                "**推荐模型**：上传数据后自动推荐",
                gr.update(choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"], value="自动推荐")
            ]
        
        file_path = file.name
        success, msg = data_loader.load_csv(file_path)
        
        if success:
            preview = data_loader.df.head(20).to_html(max_cols=10, classes='table table-striped')
            info = data_loader.get_info()
            structure_info = data_loader.get_structure_explanation()
            
            # 数据解耦
            decouple_info = "无数值列，无需解耦"
            if data_loader.numeric_cols:
                try:
                    data_decoupler = DataDecoupler()
                    default_y = data_loader.numeric_cols[0]
                    data_decoupler.fit(data_loader.df, target_col=default_y)
                    decouple_info = data_decoupler.get_summary()
                except Exception as e:
                    decouple_info = f"解耦分析失败：{str(e)[:80]}"
            
            data_size = data_loader.df.shape[0]
            data_type = data_loader.data_structure['type'] if data_loader.data_structure else 'unknown'
            
            if data_type == 'single_variable' and data_size >= 200:
                recommend = "**推荐模型**：PatchTST（单变量时序，≥200条数据）"
                recommend_value = "PatchTST"
            elif data_type == 'single_variable' and data_size >= 100:
                recommend = "**推荐模型**：LSTM（单变量时序，100-200条数据）"
                recommend_value = "LSTM"
            elif data_type in ['grouped_time_series', 'multi_variable'] and data_size >= 200:
                recommend = "**推荐模型**：EnhancedCNN1D（多变量/复杂数据，推荐）"
                recommend_value = "EnhancedCNN1D"
            elif data_type in ['grouped_time_series', 'multi_variable'] and data_size >= 100:
                recommend = "**推荐模型**：EnhancedCNN1D（多变量，k=3/5/7多尺度卷积）"
                recommend_value = "EnhancedCNN1D"
            else:
                recommend = "**推荐模型**：GradientBoosting（数据量较小或非时序任务）"
                recommend_value = "GradientBoosting"
            
            all_cols = list(data_loader.df.columns)
            numeric_cols = list(data_loader.df.select_dtypes(include=['number']).columns)
            default_x = None
            for col in all_cols:
                if any(kw in col.lower() for kw in ['date', 'time', '日期', '时间', 'timestamp', 'day', 'month', '年', '月', '日']):
                    default_x = col
                    break
            default_y = numeric_cols[0] if numeric_cols else None
            
            return [
                preview, info, structure_info,
                decouple_info,
                gr.update(choices=all_cols, value=default_x),
                gr.update(choices=numeric_cols, value=default_y),
                recommend,
                gr.update(choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"], value=recommend_value)
            ]
        return [None, msg, "**上传失败**"] + [
            "**上传失败**",
            gr.update(choices=[]),
            gr.update(choices=[]),
            "**推荐模型**：上传失败",
            gr.update(choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"], value="自动推荐")
        ]
    
    def on_train(feature_col, target_col, predict_mode, model_select, requirement, n_future, chart_requirement, prog=gr.Progress()):
        global predictor, lstm_pred
        
        if data_loader.df is None:
            return ["❌ 请先上传数据", "", "", None, "", "", ""]
        
        if not target_col:
            return ["❌ 请选择目标列 Y", "", "", None, "", ""]
        
        prog(0.1, desc="解析任务...")
        
        # 解析任务类型
        task_type = task_router.parse(requirement or "", "")
        data_size = len(data_loader.df)
        
        # 根据用户选择或自动推荐决定模型
        if model_select == "自动推荐":
            model_name, params = task_router.select_model(task_type, data_size)
        else:
            model_name = model_select
            # 设置对应的默认参数
            if model_name == 'PatchTST':
                params = {'seq_len': 96, 'pred_len': 96, 'patch_size': 16, 'd_model': 128, 'n_heads': 4, 'n_layers': 3, 'd_ff': 256, 'epochs': 30, 'batch_size': 32, 'learning_rate': 0.0005}
            elif model_name == 'LSTM':
                params = {'hidden_size': 64, 'num_layers': 2, 'epochs': 50, 'seq_len': 10, 'batch_size': 32, 'learning_rate': 0.001}
            elif model_name == 'GradientBoosting':
                params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
                task_type = 'regression'
            elif model_name == 'RandomForest':
                params = {'n_estimators': 100, 'max_depth': 10}
                task_type = 'regression'
            else:
                params = {}
        
        # 获取数据信息
        numeric_cols = list(data_loader.df.select_dtypes(include=['number']).columns)
        if target_col not in numeric_cols:
            return [f"❌ 目标列 [{target_col}] 不是数值列", "", "", None, "", "", ""]
        
        # 根据预测模式决定数据准备方式
        data_type = data_loader.data_structure['type'] if data_loader.data_structure else 'unknown'
        use_univariate = (predict_mode == "单变量时序（推荐）") or (data_type == 'single_variable')
        
        if use_univariate:
            # 单变量时序：用 Y 自己的历史值预测未来
            y = data_loader.df[target_col].values.astype(np.float32)
            if feature_col and feature_col in data_loader.df.columns:
                # 用户指定了 X 时间轴：按 X 排序后做预测
                x_raw = data_loader.df[feature_col].values
                try:
                    # 尝试转为数值排序
                    x_numeric = pd.to_numeric(pd.Series(x_raw), errors='coerce').fillna(0).values.astype(np.float32)
                    sort_idx = np.argsort(x_numeric)
                    y = y[sort_idx]
                    feature_cols_used = f"X={feature_col}（用户指定时间轴）"
                except Exception:
                    y = data_loader.df[target_col].values.astype(np.float32)
                    feature_cols_used = "（X列无法排序，使用行号）"
            else:
                # 未指定 X：用行号（默认行为）
                feature_cols_used = "（无，自变量使用行号）"
            # X 在这里只用于排序，模型只接收 y
            X_for_model = y  # 单变量：X=y
        else:
            # 多变量模式：用其他数值列作为特征（X 列也可以参与）
            feature_cols = [c for c in numeric_cols if c != target_col]
            if feature_col and feature_col in numeric_cols and feature_col != target_col:
                # 用户指定的 X 也作为特征加入
                feature_cols = [feature_col] + [c for c in feature_cols if c != feature_col]
            if not feature_cols:
                # 没有其他特征列，退化为单变量
                X_for_model = y = data_loader.df[target_col].values.astype(np.float32)
                feature_cols_used = "（无，使用自身历史值）"
            else:
                X_for_model = data_loader.df[feature_cols].values.astype(np.float32)
                y = data_loader.df[target_col].values.astype(np.float32)
                feature_cols_used = str(feature_cols)
        
        prog(0.3, desc=f"训练 {model_name}...")
        
        if model_name == 'PatchTST':
            try:
                from src.models.patchtst_model import PatchTSTPredictor
                lstm_pred = PatchTSTPredictor()
                success, msg = lstm_pred.train(
                    X_for_model, y,
                    seq_len=params.get('seq_len', 96),
                    pred_len=params.get('pred_len', 96),
                    patch_size=params.get('patch_size', 16),
                    d_model=params.get('d_model', 128),
                    n_heads=params.get('n_heads', 4),
                    n_layers=params.get('n_layers', 3),
                    d_ff=params.get('d_ff', 256),
                    epochs=params.get('epochs', 30),
                    batch_size=params.get('batch_size', 32),
                    learning_rate=params.get('learning_rate', 0.0005)
                )
                predictor = None
            except Exception as e:
                success = False
                msg = f"❌ PatchTST训练失败: {str(e)}"
                lstm_pred = None
        
        elif model_name == 'EnhancedCNN1D':
            try:
                from src.models.cnn1d_complex import EnhancedCNN1DPredictor
                lstm_pred = EnhancedCNN1DPredictor()
                success, msg = lstm_pred.train(
                    X_for_model, y,
                    seq_len=params.get('seq_len', 96),
                    pred_len=params.get('pred_len', 48),
                    hidden_channels=params.get('hidden_channels', 64),
                    num_scales=params.get('num_scales', 3),
                    kernel_sizes=(3, 5, 7),
                    num_res_blocks=params.get('num_res_blocks', 2),
                    epochs=params.get('epochs', 50),
                    batch_size=params.get('batch_size', 32),
                    learning_rate=params.get('learning_rate', 0.001),
                    dropout=params.get('dropout', 0.1),
                    use_attention=True
                )
                predictor = None
            except Exception as e:
                success = False
                msg = f"❌ EnhancedCNN1D训练失败: {str(e)}"
                lstm_pred = None
        
        elif task_type == 'time_series' and model_name == 'LSTM':
            try:
                from src.models.lstm_model import LSTMPredictor
                lstm_pred = LSTMPredictor()
                success, msg = lstm_pred.train(
                    X_for_model, y,
                    seq_len=params.get('seq_len', 10),
                    hidden_size=params.get('hidden_size', 64),
                    num_layers=params.get('num_layers', 2),
                    epochs=params.get('epochs', 50),
                    batch_size=params.get('batch_size', 32),
                    learning_rate=params.get('learning_rate', 0.001)
                )
                predictor = None
            except Exception as e:
                success = False
                msg = f"❌ LSTM训练失败: {str(e)}"
                lstm_pred = None
        
        else:
            # sklearn 模型需要 DataFrame
            if use_univariate or not feature_cols:
                X_df = data_loader.df[[target_col]]
            else:
                X_df = data_loader.df[feature_cols]
            y_series = data_loader.df[target_col]
            
            predictor = SklearnPredictor()
            success, msg = predictor.train(
                X_df, y_series,
                task_type=task_type,
                model_name=model_name,
                params=params
            )
            lstm_pred = None
        
        prog(0.9, desc="整理结果...")
        
        config = (
            f"**任务类型**：{task_type}\n"
            f"**模型**：{model_name}\n"
            f"**自变量 X**：{feature_col or '（行号）'}\n"
            f"**因变量 Y**：{target_col}\n"
            f"**特征列**：{feature_cols_used}\n"
            f"**参数**：{params}"
        )
        
        importance = ""
        if predictor and predictor.is_fitted:
            imp = predictor.get_importance()
            if imp:
                top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                importance = "\n".join([f"`{k}`: {v:.4f}" for k, v in top10])
        
        # ===== 生成未来预测 ======
        forecast_plot = None
        forecast_text = ""
        summary_text = ""
        n_steps = max(1, int(n_future or 30))
        
        if success and lstm_pred and lstm_pred.is_fitted:
            try:
                # 取最后 seq_len 个样本作为预测起点
                seq_len = lstm_pred.seq_len if hasattr(lstm_pred, 'seq_len') else 96
                X_last = y[-seq_len:] if len(y) >= seq_len else y
                future_preds = lstm_pred.predict_future(X_last, steps=n_steps)
                
                if future_preds is not None and len(future_preds) > 0:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    last_n = min(100, len(y))
                    hist = y[-last_n:]
                    steps_hist = list(range(len(y) - last_n, len(y)))
                    steps_fut = list(range(len(y), len(y) + len(future_preds)))
                    std_val = np.std(future_preds) * 0.5
                    
                    prog(0.65, desc="生成图表...")
                    forecast_plot = select_plot_function(
                        chart_requirement, hist, future_preds,
                        target_col, steps_hist, steps_fut, std_val
                    )
                    plt.close(forecast_plot)
                    
                    # 2. 预测值表格
                    rows = []
                    for i, val in enumerate(future_preds):
                        rows.append(f"  第 {i+1:3d} 步  →  **{val:.4f}**")
                    forecast_text = f"**未来 {len(future_preds)} 步预测值（{target_col}）：**\n\n" + "\n".join(rows[:30])
                    if len(future_preds) > 30:
                        forecast_text += f"\n  ...（共 {len(future_preds)} 步，已截取前30步）"
                    
                    # 3. 自然语言总结
                    first_val = float(future_preds[0])
                    last_val = float(future_preds[-1])
                    change = last_val - first_val
                    pct = (change / abs(first_val) * 100) if first_val != 0 else 0
                    
                    # 判断趋势
                    if change > 0:
                        trend = "📈 **上升趋势**"
                        trend_desc = f"从 {first_val:.4f} 上涨到 {last_val:.4f}，涨幅 **{abs(change):.4f}**（{abs(pct):.1f}%）"
                    elif change < 0:
                        trend = "📉 **下降趋势**"
                        trend_desc = f"从 {first_val:.4f} 下降到 {last_val:.4f}，跌幅 **{abs(change):.4f}**（{abs(pct):.1f}%）"
                    else:
                        trend = "➡️ **基本平稳**"
                        trend_desc = f"基本维持在 {last_val:.4f} 附近"
                    
                    # 波动分析
                    diffs = [future_preds[i+1] - future_preds[i] for i in range(len(future_preds)-1)]
                    volatility = sum(1 for d in diffs if abs(d) > 0.05 * abs(first_val)) / max(1, len(diffs))
                    
                    summary_text = (
                        f"**{target_col} 未来 {n_steps} 步趋势总结**\n\n"
                        f"**趋势方向**：{trend}\n\n"
                        f"{trend_desc}\n\n"
                        f"**预测区间**：{min(future_preds):.4f} ~ {max(future_preds):.4f}\n\n"
                        f"**预测均值**：{sum(future_preds)/len(future_preds):.4f}\n\n"
                        f"**稳定性**：{'波动较小，趋势较稳定' if volatility < 0.3 else '有一定波动，请结合实际判断'}\n\n"
                        f"💡 *以上为模型自动预测结果，仅供参考，实际走势可能受外部因素影响。*"
                    )
            except Exception as e:
                forecast_text = f"趋势预测生成失败：{str(e)}"
                summary_text = "*趋势预测生成失败，请查看上方训练结果*"
        
        # sklearn 模型的未来预测
        elif success and predictor and predictor.is_fitted and predictor.task_type != 'classification':
            try:
                last_X = X_for_model[-1] if len(X_for_model) > 0 else np.array([y[-1]])
                future_preds = predictor.predict_future(last_X, steps=n_steps)
                
                if future_preds is not None and len(future_preds) > 0:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    last_n = min(100, len(y))
                    hist = y[-last_n:]
                    steps_hist = list(range(len(y) - last_n, len(y)))
                    steps_fut = list(range(len(y), len(y) + len(future_preds)))
                    std_val = np.std(future_preds) * 0.5
                    
                    prog(0.65, desc="生成图表...")
                    forecast_plot = select_plot_function(
                        chart_requirement, hist, future_preds,
                        target_col, steps_hist, steps_fut, std_val
                    )
                    plt.close(forecast_plot)
                    
                    rows = [f"  第 {i+1:3d} 步  →  **{val:.4f}**" for i, val in enumerate(future_preds[:30])]
                    forecast_text = f"**未来 {len(future_preds)} 步预测值（{target_col}）：**\n\n" + "\n".join(rows)
                    if len(future_preds) > 30:
                        forecast_text += f"\n  ...（共 {len(future_preds)} 步，已截取前30步）"
                    
                    first_val = float(future_preds[0])
                    last_val = float(future_preds[-1])
                    change = last_val - first_val
                    pct = (change / abs(first_val) * 100) if first_val != 0 else 0
                    
                    if change > 0:
                        trend = "📈 **上升趋势**"
                        trend_desc = f"从 {first_val:.4f} 上涨到 {last_val:.4f}，涨幅 **{abs(change):.4f}**（{abs(pct):.1f}%）"
                    elif change < 0:
                        trend = "📉 **下降趋势**"
                        trend_desc = f"从 {first_val:.4f} 下降到 {last_val:.4f}，跌幅 **{abs(change):.4f}**（{abs(pct):.1f}%）"
                    else:
                        trend = "➡️ **基本平稳**"
                        trend_desc = f"基本维持在 {last_val:.4f} 附近"
                    
                    summary_text = (
                        f"**{target_col} 未来 {n_steps} 步趋势总结**\n\n"
                        f"**趋势方向**：{trend}\n\n"
                        f"{trend_desc}\n\n"
                        f"**预测区间**：{min(future_preds):.4f} ~ {max(future_preds):.4f}\n\n"
                        f"**预测均值**：{sum(future_preds)/len(future_preds):.4f}\n\n"
                        f"💡 *以上为模型自动预测结果，仅供参考。*"
                    )
            except Exception as e:
                forecast_text = f"趋势预测生成失败：{str(e)}"
                summary_text = "*趋势预测生成失败，请查看上方训练结果*"
        
        # ===== 生成结果包（zip） =====
        zip_path = ""
        if forecast_plot is not None and future_preds is not None and len(future_preds) > 0:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import zipfile
                import json
                
                output_dir = Path("outputs") / f"{target_col}_{model_name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 重新生成 fig 以便保存
                std_val = np.std(future_preds) * 0.5
                fig = select_plot_function(
                    chart_requirement, hist, future_preds,
                    target_col, steps_hist, steps_fut, std_val
                )
                fig.savefig(output_dir / "forecast.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # 保存预测数据 CSV
                all_steps = steps_hist + steps_fut
                all_vals = list(hist) + list(future_preds)
                types_ = ['历史'] * len(steps_hist) + ['预测'] * len(steps_fut)
                csv_df = pd.DataFrame({
                    'step': all_steps,
                    'type': types_,
                    target_col: all_vals
                })
                csv_df.to_csv(output_dir / "forecast_data.csv", index=False, encoding='utf-8-sig')
                
                # 保存指标 JSON
                pred_min = float(min(future_preds))
                pred_max = float(max(future_preds))
                pred_mean = float(sum(future_preds) / len(future_preds))
                first_val = float(future_preds[0])
                last_val = float(future_preds[-1])
                change = last_val - first_val
                pct = (change / abs(first_val) * 100) if first_val != 0 else 0
                trend = "上升" if change > 0 else ("下降" if change < 0 else "平稳")
                
                metrics_dict = {
                    'model': model_name,
                    'target': target_col,
                    'n_future': n_steps,
                    'metrics': (predictor.metrics if predictor and predictor.is_fitted
                                else (lstm_pred.metrics if hasattr(lstm_pred, 'metrics') else {})),
                    'pred_range': [pred_min, pred_max],
                    'pred_mean': pred_mean,
                    'trend': trend,
                    'change_pct': round(pct, 2)
                }
                with open(output_dir / "metrics.json", 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
                
                # 打包 zip
                zip_name = f"DeepPredict_{target_col}_{model_name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.zip"
                zip_path = str(output_dir / zip_name)
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(output_dir / "forecast.png", arcname="forecast.png")
                    zf.write(output_dir / "forecast_data.csv", arcname="forecast_data.csv")
                    zf.write(output_dir / "metrics.json", arcname="metrics.json")
                
                prog(0.95, desc="打包完成")
            except Exception as e:
                zip_path = ""
        
        return msg, config, importance, forecast_plot, forecast_text, summary_text, zip_path
    
    def on_download(dummy, prog=gr.Progress()):
        """手动触发下载最新结果包"""
        import zipfile
        from pathlib import Path
        outputs = Path("outputs")
        if not outputs.exists():
            return ""
        zips = sorted(outputs.rglob("DeepPredict_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if zips:
            return str(zips[0])
        return ""
    
    def on_predict(file):
        global predictor, lstm_pred
        
        if predictor is None and (lstm_pred is None or not lstm_pred.is_fitted):
            return "❌ 请先训练模型", None
        
        if file is None:
            return "❌ 请上传预测数据文件", None
        
        try:
            df = pd.read_csv(file.name)
        except Exception as e:
            return f"❌ 读取文件失败: {e}", None
        
        if predictor and predictor.is_fitted:
            cols = [c for c in predictor.feature_names if c in df.columns]
            if not cols:
                return "❌ 新数据中没有匹配的特征列", None
            pred = predictor.predict(df[cols])
            result_df = df.copy()
            result_df['预测值'] = pred
        elif lstm_pred and lstm_pred.is_fitted:
            X = df.values.astype(np.float32)
            pred = lstm_pred.predict(X)
            result_df = df.copy()
            result_df['预测值'] = pred
        
        table = result_df.tail(30).to_html(max_cols=15, classes='table table-striped')
        return "✅ 预测完成", table
    
    # 绑定事件
    file_input.change(
        on_file_upload,
        inputs=[file_input],
        outputs=[data_preview, data_info, data_structure_info, data_decouple_info, feature_col, target_col, model_recommend, model_select]
    )
    train_btn.click(
        on_train,
        inputs=[feature_col, target_col, predict_mode, model_select, requirement, n_future, chart_requirement],
        outputs=[result_out, config_out, importance_out, forecast_plot, forecast_out, summary_out, download_file]
    )
    download_btn.click(
        on_download,
        inputs=[download_file],
        outputs=[download_file]
    )
    predict_btn.click(on_predict, inputs=[predict_file], outputs=[predict_status, predict_out])


if __name__ == "__main__":
    # Railway / HuggingFace Spaces 用 PORT 环境变量
    import os
    port = int(os.environ.get("PORT", 7860))
    print("=" * 60)
    print("  DeepPredict Web 版 v1.04 已启动！")
    print(f"  访问地址: http://0.0.0.0:{port}")
    print("  同一局域网内的手机/电脑都可以访问")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
