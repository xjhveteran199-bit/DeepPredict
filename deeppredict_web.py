"""
DeepPredict Web 版 - Gradio 界面 v1.02
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


# ============ 全局状态 ============
data_loader = DataLoader()
task_router = TaskRouter()
predictor = None
lstm_pred = None


# ============ Gradio 界面 ============

with gr.Blocks(title="DeepPredict v1.03 - 智能数据分析版") as demo:
    
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
                    
                    gr.Markdown("### 📈 数据摘要")
                    data_info = gr.Markdown("**上传数据后显示摘要**")
            
            gr.Markdown("---")
            
            with gr.Row():
                # === 任务配置 ===
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 任务配置")
                    
                    target_col = gr.Dropdown(
                        label="① 选择目标列 Y（要预测什么）",
                        choices=[],
                        info="选择你要预测的目标变量"
                    )
                    
                    predict_mode = gr.Radio(
                        choices=["单变量时序（推荐）", "多变量预测"],
                        value="单变量时序（推荐）",
                        label="② 预测模式",
                        info="时序数据用单变量，分类/回归可用多变量"
                    )
                    
                    model_select = gr.Dropdown(
                        label="③ 选择模型",
                        choices=["自动推荐", "PatchTST", "LSTM", "GradientBoosting", "RandomForest"],
                        value="自动推荐",
                        info="自动推荐会根据数据情况选择最优模型"
                    )
                    model_recommend = gr.Markdown("")
                    
                    requirement = gr.Textbox(
                        label="④ 描述需求（可选）",
                        placeholder="示例：预测K值变化趋势\n预测Glu未来走势\n判断是否流失",
                        lines=2
                    )
                
                # === 结果展示 ===
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 任务配置预览")
                    config_out = gr.Markdown("*配置信息将显示在这里*")
                    
                    gr.Markdown("### 📊 训练结果")
                    result_out = gr.Textbox(label="", lines=8, interactive=False)
                    
                    gr.Markdown("### 🔍 特征重要性")
                    importance_out = gr.Markdown("")
            
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
        global data_loader, predictor, lstm_pred
        predictor = None
        lstm_pred = None
        
        if file is None:
            return None, "**请上传 CSV 文件**", "**上传数据后自动分析**", gr.update(choices=[]), "**推荐模型**：上传数据后自动推荐", gr.update(choices=["自动推荐", "PatchTST", "LSTM", "GradientBoosting", "RandomForest"], value="自动推荐")
        
        file_path = file.name
        success, msg = data_loader.load_csv(file_path)
        
        if success:
            preview = data_loader.df.head(20).to_html(max_cols=10, classes='table table-striped')
            info = data_loader.get_info()
            structure_info = data_loader.get_structure_explanation()
            
            # 根据数据推荐模型
            data_size = data_loader.df.shape[0]
            data_type = data_loader.data_structure['type'] if data_loader.data_structure else 'unknown'
            
            if data_type == 'single_variable' and data_size >= 200:
                recommend = "**推荐模型**：PatchTST（单变量时序，≥200条数据）"
                recommend_value = "PatchTST"
            elif data_type == 'single_variable' and data_size >= 100:
                recommend = "**推荐模型**：LSTM（单变量时序，100-200条数据）"
                recommend_value = "LSTM"
            elif data_type in ['grouped_time_series', 'multi_variable'] and data_size >= 200:
                recommend = "**推荐模型**：PatchTST（多变量时序，≥200条数据）"
                recommend_value = "PatchTST"
            elif data_type in ['grouped_time_series', 'multi_variable'] and data_size >= 100:
                recommend = "**推荐模型**：LSTM（多变量时序，100-200条数据）"
                recommend_value = "LSTM"
            else:
                recommend = "**推荐模型**：GradientBoosting（数据量较小或非时序任务）"
                recommend_value = "GradientBoosting"
            
            numeric_cols = list(data_loader.df.select_dtypes(include=['number']).columns)
            return (
                preview, info, structure_info,
                gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),
                recommend,
                gr.update(choices=["自动推荐", "PatchTST", "LSTM", "GradientBoosting", "RandomForest"], value=recommend_value)
            )
        return None, msg, "**上传失败**", gr.update(choices=[]), "**推荐模型**：上传失败", gr.update(choices=["自动推荐", "PatchTST", "LSTM", "GradientBoosting", "RandomForest"], value="自动推荐")
    
    def on_train(target_col, predict_mode, model_select, requirement, prog=gr.Progress()):
        global predictor, lstm_pred
        
        if data_loader.df is None:
            return "❌ 请先上传数据", "", ""
        
        if not target_col:
            return "❌ 请选择目标列", "", ""
        
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
            return f"❌ 目标列 [{target_col}] 不是数值列", "", ""
        
        # 根据预测模式和数据结构决定特征使用方式
        data_type = data_loader.data_structure['type'] if data_loader.data_structure else 'unknown'
        use_univariate = (predict_mode == "单变量时序（推荐）") or (data_type == 'single_variable')
        
        if use_univariate:
            # 单变量模式：用目标列自己的历史值
            X = data_loader.df[target_col].values.astype(np.float32)
            y = data_loader.df[target_col].values.astype(np.float32)
            feature_cols_used = "（无，使用自身历史值）"
        else:
            # 多变量模式：用其他数值列作为特征
            feature_cols = [c for c in numeric_cols if c != target_col]
            if not feature_cols:
                # 没有其他特征列，退化为单变量
                X = data_loader.df[target_col].values.astype(np.float32)
                y = data_loader.df[target_col].values.astype(np.float32)
                feature_cols_used = "（无，使用自身历史值）"
            else:
                X = data_loader.df[feature_cols].values.astype(np.float32)
                y = data_loader.df[target_col].values.astype(np.float32)
                feature_cols_used = str(feature_cols)
        
        prog(0.3, desc=f"训练 {model_name}...")
        
        if model_name == 'PatchTST':
            try:
                from src.models.patchtst_model import PatchTSTPredictor
                lstm_pred = PatchTSTPredictor()
                success, msg = lstm_pred.train(
                    X, y,
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
        
        elif task_type == 'time_series' and model_name == 'LSTM':
            try:
                from src.models.lstm_model import LSTMPredictor
                lstm_pred = LSTMPredictor()
                success, msg = lstm_pred.train(
                    X, y,
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
            f"**目标列**：{target_col}\n"
            f"**特征列**：{feature_cols_used}\n"
            f"**参数**：{params}"
        )
        
        importance = ""
        if predictor and predictor.is_fitted:
            imp = predictor.get_importance()
            if imp:
                top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                importance = "\n".join([f"`{k}`: {v:.4f}" for k, v in top10])
        
        return msg, config, importance
    
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
    file_input.change(on_file_upload, inputs=[file_input], outputs=[data_preview, data_info, data_structure_info, target_col, model_recommend, model_select])
    train_btn.click(on_train, inputs=[target_col, predict_mode, model_select, requirement], outputs=[result_out, config_out, importance_out])
    predict_btn.click(on_predict, inputs=[predict_file], outputs=[predict_status, predict_out])


if __name__ == "__main__":
    print("=" * 60)
    print("  DeepPredict Web 版 v1.02 已启动！")
    print("  访问地址: http://localhost:7860")
    print("  同一局域网内的手机/电脑都可以访问")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
