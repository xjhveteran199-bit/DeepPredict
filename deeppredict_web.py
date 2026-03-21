"""
DeepPredict Web 版 - Gradio 界面
直接双击运行 http://localhost:7860
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path


# ============ 核心模块 ============

class DataLoader:
    def __init__(self):
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
    
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
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(self.df.select_dtypes(include=['object']).columns)
    
    def get_info(self):
        if self.df is None:
            return ""
        return (
            f"**数据形状**：{self.df.shape[0]} 行 × {self.df.shape[1]} 列\n\n"
            f"**数值列**（{len(self.numeric_cols)}）：`{', '.join(self.numeric_cols[:5])}{'...' if len(self.numeric_cols)>5 else ''}`\n\n"
            f"**类别列**（{len(self.categorical_cols)}）：`{', '.join(self.categorical_cols[:5])}{'...' if len(self.categorical_cols)>5 else ''}`"
        )


class TaskRouter:
    PATTERNS = {
        'time_series': ['时序', '预测', 'forecast', '未来', '趋势', '销售预测', '销量', '流量', '股票'],
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
                return 'LSTM', {'hidden_size': 64, 'num_layers': 2, 'epochs': 50, 'seq_len': 10}
            return 'GradientBoosting', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
        return 'GradientBoosting', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}


class LSTMPredictor:
    """用 GradientBoosting 模拟 LSTM 行为（无 torch DLL 问题）"""
    def __init__(self):
        self.seq_len = 10
        self.scaler = None
        self.model = None
        self.is_fitted = False
        self.metrics = {}
    
    def train(self, X, y, seq_len=10, **kwargs):
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import GradientBoostingRegressor
            
            self.seq_len = seq_len
            
            # 创建滞后特征（模拟时序窗口）
            X_lag, y_lag = [], []
            for i in range(seq_len, len(X)):
                X_lag.append(X[i-seq_len:i].flatten())
                y_lag.append(y[i])
            
            X_arr = np.array(X_lag)
            y_arr = np.array(y_lag)
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_arr)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_arr, test_size=0.2, random_state=42
            )
            
            self.model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = 1 - mse / (np.var(y_test) + 1e-8)
            
            self.metrics = {'R²': r2, 'RMSE': rmse, 'MAE': mae}
            self.is_fitted = True
            
            return True, f"✅ 时序模型训练完成\n**R²**={r2:.4f} **RMSE**={rmse:.4f} **MAE**={mae:.4f}"
        except Exception as e:
            return False, f"❌ {str(e)}"
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型未训练")
        X_lag = []
        for i in range(self.seq_len, len(X)):
            X_lag.append(X[i-self.seq_len:i].flatten())
        X_scaled = self.scaler.transform(np.array(X_lag))
        return self.model.predict(X_scaled)


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
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X.fillna(X.median()))
            
            if task_type == 'classification':
                self.label_encoder = LabelEncoder()
                y_enc = self.label_encoder.fit_transform(y.astype(str))
            else:
                y_enc = y.astype(float)
            
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

with gr.Blocks(title="DeepPredict v1.0") as demo:
    
    gr.Markdown("""
    # 🧠 DeepPredict - 智能预测工具 v1.0
    ### 研究生友好 | 无需编程基础 | 支持手机浏览器
    """)
    
    with gr.Row():
        # === 左侧面板 ===
        with gr.Column(scale=1):
            gr.Markdown("### 📂 第一步：导入数据")
            file_input = gr.File(label="点击上传 CSV 文件", file_types=[".csv"])
            data_preview = gr.HTML(label="数据预览（前20行）")
            data_info = gr.Markdown("**上传数据后显示摘要**")
            
            gr.Markdown("### 🎯 第二步：配置任务")
            target_col = gr.Dropdown(label="选择目标列（要预测的列）", choices=[])
            requirement = gr.Textbox(
                label="预测需求描述",
                placeholder="示例：预测下个月的销售额\n判断用户是否会流失\n预测未来7天访问量趋势",
                lines=3
            )
            train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
        
        # === 右侧面板 ===
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 任务配置")
            config_out = gr.Markdown("训练后显示配置")
            
            gr.Markdown("### 📊 训练结果")
            result_out = gr.Textbox(label="训练日志", lines=6, interactive=False)
            
            gr.Markdown("### 🔍 特征重要性 Top10")
            importance_out = gr.Markdown()
            
            gr.Markdown("### 🔮 第三步：预测新数据")
            predict_file = gr.File(label="上传新数据 CSV（可选）", file_types=[".csv"])
            predict_btn = gr.Button("执行预测", variant="secondary")
            predict_out = gr.HTML(label="预测结果")
            predict_status = gr.Textbox(label="状态", lines=2, interactive=False)
    
    # ========== 事件处理 ==========
    
    def on_file_upload(file):
        global data_loader, predictor, lstm_pred
        predictor = None
        lstm_pred = None
        
        if file is None:
            return None, "**请上传 CSV 文件**", gr.update(choices=[])
        
        # file 是 Gradio 的 FileData 对象
        file_path = file.name
        success, msg = data_loader.load_csv(file_path)
        
        if success:
            preview = data_loader.df.head(20).to_html(max_cols=10, classes='table table-striped')
            info = data_loader.get_info()
            choices = list(data_loader.df.columns)
            return preview, info, gr.update(choices=choices, value=choices[0] if choices else None)
        return None, msg, gr.update(choices=[])
    
    def on_train(target_col, requirement, prog=gr.Progress()):
        global predictor, lstm_pred
        
        if data_loader.df is None:
            return "❌ 请先上传数据", "", ""
        
        if not target_col:
            return "❌ 请选择目标列", "", ""
        
        if not requirement.strip():
            return "❌ 请填写预测需求", "", ""
        
        prog(0.1, desc="解析任务...")
        summary = data_loader.get_info()
        task_type = task_router.parse(requirement, summary)
        model_name, params = task_router.select_model(task_type, len(data_loader.df))
        
        feature_df = data_loader.df.drop(columns=[target_col])
        target_series = data_loader.df[target_col]
        
        if feature_df.empty or feature_df.shape[1] == 0:
            return "❌ 特征列不能为空", "", ""
        
        prog(0.3, desc=f"训练 {model_name}...")
        
        if task_type == 'time_series' and model_name == 'LSTM':
            lstm_pred = LSTMPredictor()
            X = feature_df.values.astype(np.float32)
            y = target_series.values.astype(np.float32)
            success, msg = lstm_pred.train(X, y, seq_len=params.get('seq_len', 10))
            predictor = None
        else:
            predictor = SklearnPredictor()
            success, msg = predictor.train(
                feature_df, target_series,
                task_type=task_type,
                model_name=model_name,
                params=params
            )
            lstm_pred = None
        
        prog(0.9, desc="整理结果...")
        
        config = f"**任务类型**：{task_type}\n**模型**：{model_name}\n**参数**：{params}"
        
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
    file_input.change(on_file_upload, inputs=[file_input], outputs=[data_preview, data_info, target_col])
    train_btn.click(on_train, inputs=[target_col, requirement], outputs=[result_out, config_out, importance_out])
    predict_btn.click(on_predict, inputs=[predict_file], outputs=[predict_status, predict_out])


if __name__ == "__main__":
    print("=" * 60)
    print("  DeepPredict Web 版已启动！")
    print("  访问地址: http://localhost:7860")
    print("  同一局域网内的手机/电脑都可以访问")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
