"""
DeepPredict Web 版 - Gradio 界面 v1.04
改进：智能数据分析 + 用户引导
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# matplotlib 中文字体配置（解决 DejaVu Sans 缺失中文字形问题）
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from gradio.data_classes import ListFiles, FileData

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============ 工具函数 ============
def extract_file_path(file_obj):
    """从 Gradio 6.x File 组件返回值中提取文件路径字符串。
    
    Gradio 6.x 返回 ListFiles/FileData/dict/str 等多种格式，此函数统一处理。
    """
    if file_obj is None:
        return None
    # ListFiles (Gradio 6.x Pydantic root model，行为像 list 但不是 list 的子类)
    if isinstance(file_obj, ListFiles):
        file_obj = file_obj.root  # 取出内部的 list[FileData]
    # 普通列表：取第一个元素（跳过空列表）
    if isinstance(file_obj, list):
        if len(file_obj) == 0:
            return None
        file_obj = file_obj[0]
    # dict（序列化后的 FileData）
    if isinstance(file_obj, dict):
        path = file_obj.get('path', '')
        if path and Path(path).is_file():
            return str(path)
        return None
    # FileData Pydantic 对象
    if hasattr(file_obj, 'path'):
        path = str(file_obj.path)
        if Path(path).is_file():
            return path
        # 如果 path 是目录（Gradio 缓存目录），尝试获取 orig_name
        if hasattr(file_obj, 'orig_name') and file_obj.orig_name:
            # 缓存目录下的实际文件
            return str(Path(path).parent / file_obj.orig_name)
        return None
    # 已经是字符串：检查是否是有效文件
    if isinstance(file_obj, str):
        p = Path(file_obj)
        if p.is_file():
            return str(p)
        return None
    return None


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


# ============ 时间轴构建工具 ============

def build_datetime_steps(df, feature_col, n_hist, n_fut):
    """
    从 datetime 列构建真实时间轴，返回：
    steps_*, xtick_*, date_fmt, first_date, first_future_date
    steps_* = 数字索引（用于绘图定位）
    xtick_* = 格式化日期字符串（用于 X 轴标签）
    """
    import pandas as pd
    import numpy as np

    date_series = None
    date_fmt = "%Y-%m-%d"
    freq = "MS"

    if feature_col and feature_col in df.columns:
        try:
            parsed = pd.to_datetime(df[feature_col], errors='coerce')
            if parsed.notna().sum() > max(5, len(df) * 0.3):
                # 如果解析出来的日期全在1975年之前且跨度<1天，说明是数值列误识别为日期
                parsed_years = parsed.dropna().dt.year
                parsed_range = (parsed.dropna().max() - parsed.dropna().min()).total_seconds()
                if len(parsed_years) > 0 and parsed_years.max() <= 1975 and parsed_range < 86400:
                    # 数值列（分钟/秒）误识别为日期，退回数值模式
                    pass
                else:
                    date_series = parsed
                    deltas = parsed.diff().dropna()
                    if len(deltas) > 0:
                        days_median = deltas.median().days
                        if days_median <= 1:
                            freq = "D"
                            date_fmt = "%Y-%m-%d"
                        elif days_median <= 10:
                            freq = f"{int(days_median)}D"
                            date_fmt = "%Y-%m-%d"
                        elif days_median <= 35:
                            freq = "MS"
                            date_fmt = "%Y-%m"
                        else:
                            freq = "YS"
                            date_fmt = "%Y"
                    else:
                        freq = "MS"
                        date_fmt = "%Y-%m"
        except Exception:
            pass

    last_n = min(n_hist, len(date_series)) if date_series is not None else 0

    if date_series is not None and last_n > 2:
        # 历史
        last_dates = date_series.iloc[-last_n:].reset_index(drop=True)
        start_idx = len(date_series) - last_n
        steps_hist = list(range(start_idx, start_idx + last_n))
        step = max(1, last_n // 12)
        xtick_hist = [d.strftime(date_fmt) if i % step == 0 else "" for i, d in enumerate(last_dates)]

        # 未来
        last_date = date_series.iloc[-1]
        try:
            future_dates = pd.date_range(start=last_date, periods=n_fut + 1, freq=freq)[1:]
        except Exception:
            future_dates = pd.date_range(start=last_date, periods=n_fut + 1, freq="MS")[1:]
        steps_fut = list(range(len(date_series), len(date_series) + n_fut))
        step_fut = max(1, n_fut // 12)
        xtick_fut = [d.strftime(date_fmt) if i % step_fut == 0 else "" for i, d in enumerate(future_dates)]

        first_date = last_dates.iloc[0]
        first_fut = future_dates[0] if len(future_dates) > 0 else None
        return steps_hist, steps_fut, xtick_hist, xtick_fut, date_fmt, first_date, first_fut
    else:
        # ===== 数值/序号模式：直接用时间值作刻度位置，避免索引偏移 =====
        col_vals = None
        if feature_col and feature_col in df.columns:
            try:
                col_vals = pd.to_numeric(df[feature_col], errors='coerce')
            except Exception:
                col_vals = None

        if col_vals is not None and col_vals.notna().sum() > len(df) * 0.5:
            # 推断采样间隔
            vals = col_vals.dropna().values
            val_step = 1.0
            if len(vals) >= 2:
                diffs = np.diff(vals)
                diffs = diffs[diffs > 0]
                if len(diffs) > 0:
                    val_step = float(np.median(diffs))

            last_n = min(n_hist, len(col_vals))
            last_vals = col_vals.iloc[-last_n:].reset_index(drop=True)
            last_time = float(last_vals.iloc[-1])
            first_time = float(last_vals.iloc[0])

            # 历史：用实际时间值作为 X 轴位置
            steps_hist = list(last_vals.values)  # 直接用时间值
            # 均匀取刻度（~6个历史标签，~8个未来标签，避免X轴重叠）
            hist_step = max(1, last_n // 6)  # ~6 labels for historical
            xtick_hist = [f"{last_vals.iloc[i]:.1f}" if i % hist_step == 0 else "" for i in range(last_n)]

            # 未来：也用实际时间值
            future_vals = [last_time + val_step * (i + 1) for i in range(n_fut)]
            steps_fut = future_vals  # 直接用时间值，不再用行号
            fut_step = max(1, n_fut // 8)  # ~8 labels for future
            xtick_fut = [f"{future_vals[i]:.1f}" if i % fut_step == 0 else "" for i in range(n_fut)]

            xlabel = feature_col
            return steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel, last_vals.iloc[0], future_vals[0] if future_vals else None
        else:
            # 完全没有可用列：退化为序号
            steps_hist = list(range(0, n_hist))
            steps_fut = list(range(n_hist, n_hist + n_fut))
            step = max(1, n_hist // 12)
            xtick_hist = [str(i) if i % step == 0 else "" for i in steps_hist]
            step_fut = max(1, n_fut // 12)
            xtick_fut = [str(i) if i % step_fut == 0 else "" for i in steps_fut]
            return steps_hist, steps_fut, xtick_hist, xtick_fut, None, None, None


def _apply_xticks(ax, steps_hist, steps_fut, xtick_hist, xtick_fut, has_datetime):
    """统一设置 X 轴刻度标签（数字或真实日期）"""
    import matplotlib.pyplot as plt
    all_steps = steps_hist + steps_fut
    all_labels = xtick_hist + xtick_fut
    if has_datetime:
        # 只显示部分标签避免拥挤
        ax.set_xticks(all_steps)
        ax.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=8)
    else:
        ax.set_xlabel('时间步', fontsize=11)


# ============ 图表模板函数 ============

def select_plot_function(chart_requirement, hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, std_val=None, xlabel=None):
    """根据用户描述选择图表模板（纯规则匹配，无需 AI）"""
    # 只有 date_fmt 是真正的日期格式字符串（如 "%Y-%m"）才算 datetime 模式
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_datetime = xtick_hist is not None and any(xtick_hist) and is_date_fmt
    req = (chart_requirement or "").lower()
    if any(kw in req for kw in ["双轴", "双y", "次坐标", "dual", "twin"]):
        return plot_dual_axis(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel)
    elif any(kw in req for kw in ["置信", "误差", "上下界", "区间", "confidence", "band", "uncertainty"]):
        return plot_with_confidence_band(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, std_val, xlabel)
    elif any(kw in req for kw in ["散点", "scatter"]):
        return plot_scatter_with_line(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel)
    elif any(kw in req for kw in ["柱状", "bar", "柱形"]):
        return plot_bar_forecast(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel)
    else:
        return plot_standard_forecast(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel)


def plot_standard_forecast(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, xlabel=None):
    """标准折线图：蓝色历史 + 橙色预测，支持真实日期/数值时间轴"""
    import matplotlib.pyplot as plt
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制分隔线（历史与预测的边界）
    if has_xtick and steps_hist and steps_fut:
        # 数值模式或日期模式：分隔线在最后一个历史时间点
        boundary = float(steps_hist[-1])
        ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5, label='预测起点')
    elif not has_xtick:
        ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)

    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=2, label='历史数据')
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='预测')

    # X轴刻度：只显示有标签的位置，彻底避免重叠
    if has_xtick:
        # 历史部分：取所有非空标签的步进
        hist_ticks = [(int(i), xtick_hist[i]) for i in range(len(xtick_hist)) if xtick_hist[i]]
        fut_ticks = [(len(steps_hist) + i, xtick_fut[i]) for i in range(len(xtick_fut)) if xtick_fut[i]]
        all_ticks = hist_ticks + fut_ticks
        if all_ticks:
            tick_pos, tick_lab = zip(*all_ticks)
            plt.xticks(tick_pos, tick_lab, rotation=30, ha='right', fontsize=8)
    elif not is_date_fmt:
        ax.set_xlabel('时间步', fontsize=11)

    if is_date_fmt:
        ax.set_xlabel('日期', fontsize=11)
    elif xlabel:
        ax.set_xlabel(str(xlabel), fontsize=11)

    ax.set_ylabel(target_col, fontsize=11)
    ax.set_title(f'{target_col} 趋势预测', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_dual_axis(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, xlabel=None):
    """双轴图：左轴历史，右轴预测"""
    import matplotlib.pyplot as plt
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1, color2 = '#3B82F6', '#FF6B2B'

    # 分隔线
    if has_xtick and steps_hist and steps_fut:
        boundary = float(steps_hist[-1])
        ax1.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5)

    # X轴刻度（精确控制）
    if has_xtick:
        hist_ticks = [(i, xtick_hist[i]) for i in range(len(xtick_hist)) if xtick_hist[i]]
        fut_ticks = [(len(steps_hist) + i, xtick_fut[i]) for i in range(len(xtick_fut)) if xtick_fut[i]]
        all_ticks = hist_ticks + fut_ticks
        if all_ticks:
            tp, tl = zip(*all_ticks)
            plt.xticks(tp, tl, rotation=45, ha='right', fontsize=8)

    if is_date_fmt:
        ax1.set_xlabel('日期', fontsize=11)
    elif xlabel:
        ax1.set_xlabel(str(xlabel), fontsize=11)
    else:
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


def plot_with_confidence_band(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, std_val=None, xlabel=None):
    """带置信区间的预测图，支持真实日期"""
    import matplotlib.pyplot as plt
    if std_val is None:
        std_val = np.std(future_preds) * 0.5
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=2, label='历史数据')
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='预测')
    ax.fill_between(steps_fut,
                    [v - 1.96 * std_val for v in future_preds],
                    [v + 1.96 * std_val for v in future_preds],
                    color='#FF6B2B', alpha=0.2, label='95%置信区间')
    ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)
    if has_xtick:
        all_steps = steps_hist + steps_fut
        all_labels = xtick_hist + xtick_fut
        ax.set_xticks(all_steps)
        ax.set_xticklabels(all_labels, rotation=35, ha='right', fontsize=8)
    if is_date_fmt:
        ax.set_xlabel('日期', fontsize=11)
    elif xlabel:
        ax.set_xlabel(str(xlabel), fontsize=11)
    else:
        ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel(target_col, fontsize=11)
    ax.set_title(f'{target_col} 趋势预测（含置信区间）', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scatter_with_line(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, xlabel=None):
    """散点+折线组合图，支持真实日期"""
    import matplotlib.pyplot as plt
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(steps_hist, hist, color='#3B82F6', s=20, zorder=3, label='历史数据（散点）')
    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=1.5, alpha=0.6)
    ax.scatter(steps_fut, future_preds, color='#FF6B2B', s=30, marker='D', zorder=3, label='预测（散点）')
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--')
    ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)
    if has_xtick:
        all_steps = steps_hist + steps_fut
        all_labels = xtick_hist + xtick_fut
        ax.set_xticks(all_steps)
        ax.set_xticklabels(all_labels, rotation=35, ha='right', fontsize=8)
    if is_date_fmt:
        ax.set_xlabel('日期', fontsize=11)
    elif xlabel:
        ax.set_xlabel(str(xlabel), fontsize=11)
    else:
        ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel(target_col, fontsize=11)
    ax.set_title(f'{target_col} 趋势预测（散点图）', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bar_forecast(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, xlabel=None):
    """柱状图（历史用柱状，预测用折线），支持真实日期"""
    import matplotlib.pyplot as plt
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax = plt.subplots(figsize=(11, 4))
    bar_width = 0.8
    ax.bar(steps_hist, hist, color='#3B82F6', width=bar_width, label='历史数据', alpha=0.8)
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='预测', marker='o', markersize=4)
    ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)
    if has_xtick:
        all_steps = steps_hist + steps_fut
        all_labels = xtick_hist + xtick_fut
        ax.set_xticks(all_steps)
        ax.set_xticklabels(all_labels, rotation=35, ha='right', fontsize=8)
    if is_date_fmt:
        ax.set_xlabel('日期', fontsize=11)
    elif xlabel:
        ax.set_xlabel(str(xlabel), fontsize=11)
    else:
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
                    
                    gr.Markdown("### 📊 数据预览（均匀采样200行，覆盖全部时间范围）")
                    data_preview = gr.DataFrame(label="", max_height=400, wrap=True, column_widths=None, buttons=['fullscreen', 'copy'], show_search='filter')
                
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
                    
                    with gr.Row():
                        n_future_val = gr.Number(
                            label="⑥ 预测未来时间长度",
                            value=10,
                            info="数值，默认10",
                            precision=1,
                            scale=2
                        )
                        n_future_unit = gr.Dropdown(
                            label="",
                            choices=["分钟", "小时", "天", "秒"],
                            value="分钟",
                            scale=1,
                            min_width=80
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
                    forecast_plot = gr.Image(label="趋势预测图（蓝色=历史，橙色=预测）")
                    
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
        
        file_path = extract_file_path(file)
        if not file_path:
            return [None] * 7 + [
                "**文件路径无效**",
                "**上传失败**",
                "**上传后自动识别列类型**",
                gr.update(choices=[]),
                gr.update(choices=[]),
                "**推荐模型**：上传失败",
                gr.update(choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"], value="自动推荐")
            ]
        success, msg = data_loader.load_csv(file_path)
        
        if success:
            # 均匀采样200行：覆盖完整时间范围（10~50min），而非只取前200行（仅覆盖10~13min）
            n_total = len(data_loader.df)
            n_sample = 200
            if n_total <= n_sample:
                preview = data_loader.df.copy()
            else:
                indices = np.linspace(0, n_total - 1, n_sample, dtype=int)
                preview = data_loader.df.iloc[indices].reset_index(drop=True)
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
    
    def on_train(feature_col, target_col, predict_mode, model_select, requirement, n_future_val, n_future_unit, chart_requirement, prog=gr.Progress()):
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
        
        # ===== 统一保存图表的目录 ======
        # 在任何模型分支之前创建，确保所有代码路径都能统一保存
        import matplotlib
        matplotlib.use('Agg')
        output_dir = Path("outputs") / f"{target_col}_{model_name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # ===== 生成未来预测 ======
        forecast_plot = None
        forecast_text = ""
        summary_text = ""
        # ===== 时间长度 → 步数转换 =====
        time_val = float(n_future_val) if n_future_val is not None else 10.0
        unit = n_future_unit or "分钟"
        # 根据单位换算成"分钟"量级
        if unit == "小时":
            time_min = time_val * 60
        elif unit == "天":
            time_min = time_val * 60 * 24
        elif unit == "秒":
            time_min = time_val / 60.0
        else:  # 分钟
            time_min = time_val
        # 用时间列的采样间隔估算步数
        if feature_col and feature_col in data_loader.df.columns:
            try:
                col_vals = pd.to_numeric(data_loader.df[feature_col], errors='coerce').dropna()
                if len(col_vals) >= 2:
                    intervals = col_vals.diff().dropna()
                    avg_interval = intervals.mean()
                    if avg_interval > 0:
                        n_steps = max(1, int(round(time_min / avg_interval)))
                    else:
                        n_steps = max(1, int(time_val * 10))
                else:
                    n_steps = max(1, int(time_val * 10))
            except Exception:
                n_steps = max(1, int(time_val * 10))
        else:
            n_steps = max(1, int(time_val * 10))
        n_steps = min(n_steps, 2000)  # 上限防止内存问题
        # 预初始化的绘图变量（供 zip 导出使用）
        steps_hist = steps_fut = xtick_hist = xtick_fut = None
        hist = None
        
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
                    # 构建真实时间轴（支持日期列和数值列）
                    d_steps = build_datetime_steps(data_loader.df, feature_col, last_n, len(future_preds))
                    steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel, first_date, first_fut = d_steps
                    std_val = np.std(future_preds) * 0.5
                    
                    prog(0.65, desc="生成图表...")
                    forecast_plot = select_plot_function(
                        chart_requirement, hist, future_preds,
                        target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, std_val, xlabel
                    )
                    # 立即保存为PNG并转换为路径字符串，避免返回matplotlib Figure导致Gradio postprocess错误
                    forecast_plot.savefig(output_dir / "forecast.png", dpi=150, bbox_inches='tight')
                    plt.close(forecast_plot)
                    forecast_plot = str(output_dir / "forecast.png")
                    
                    # 2. 预测值表格（显示真实日期或数值时间轴）
                    rows = []
                    is_datetime_mode = xlabel and '%' in str(xlabel)
                    if is_datetime_mode and first_fut is not None:
                        # 日期时间模式
                        freq = "MS"
                        try:
                            future_dates = pd.date_range(start=first_fut, periods=len(future_preds), freq=freq)
                        except Exception:
                            future_dates = pd.date_range(start=first_fut, periods=len(future_preds), freq="MS")
                        for i, (val, dt) in enumerate(zip(future_preds, future_dates)):
                            rows.append(f"  {dt.strftime(xlabel)}  →  **{val:.4f}**")
                        time_range = f"{first_fut.strftime(xlabel)} ~ {future_dates[-1].strftime(xlabel)}"
                    elif first_fut is not None:
                        # 数值时间轴模式（分钟/秒等）
                        val_step = float(first_fut - first_date) if first_date is not None and first_fut != first_date else 1.0
                        for i, val in enumerate(future_preds):
                            future_val = float(first_fut) + val_step * (i + 1)
                            rows.append(f"  {future_val:.2f}  →  **{val:.4f}**")
                        time_range = f"{float(first_fut):.2f} ~ {float(first_fut) + val_step * len(future_preds):.2f} ({xlabel or '时间'})"
                    else:
                        for i, val in enumerate(future_preds):
                            rows.append(f"  第 {i+1:3d} 步  →  **{val:.4f}**")
                        time_range = f"第 {len(y)} 步 ~ 第 {len(y)+len(future_preds)-1} 步"
                    forecast_text = f"**未来 {len(future_preds)} 步预测值（{target_col}）：{time_range}**\n\n" + "\n".join(rows[:30])
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
                        f"**{target_col} 未来 {n_steps} 步趋势总结**（{time_range}）\n\n"
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
                    # 构建真实时间轴
                    d_steps = build_datetime_steps(data_loader.df, feature_col, last_n, len(future_preds))
                    steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel, first_date, first_fut = d_steps
                    std_val = np.std(future_preds) * 0.5
                    
                    prog(0.65, desc="生成图表...")
                    forecast_plot = select_plot_function(
                        chart_requirement, hist, future_preds,
                        target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, std_val, xlabel
                    )
                    # 立即保存为PNG并转换为路径字符串，避免返回matplotlib Figure导致Gradio postprocess错误
                    forecast_plot.savefig(output_dir / "forecast.png", dpi=150, bbox_inches='tight')
                    plt.close(forecast_plot)
                    forecast_plot = str(output_dir / "forecast.png")
                    
                    # 预测值表格（显示真实日期或数值时间轴）
                    is_datetime_mode = xlabel and '%' in str(xlabel)
                    if is_datetime_mode and first_fut is not None:
                        freq = "MS"
                        try:
                            future_dates = pd.date_range(start=first_fut, periods=len(future_preds), freq=freq)
                        except Exception:
                            future_dates = pd.date_range(start=first_fut, periods=len(future_preds), freq="MS")
                        rows = [f"  {dt.strftime(xlabel)}  →  **{val:.4f}**" for dt, val in zip(future_dates[:30], future_preds[:30])]
                        time_range = f"{first_fut.strftime(xlabel)} ~ {future_dates[-1].strftime(xlabel)}"
                    elif first_fut is not None:
                        val_step = float(first_fut - first_date) if first_date is not None and first_fut != first_date else 1.0
                        rows = [f"  {float(first_fut) + val_step * (i+1):.2f}  →  **{val:.4f}**" for i, val in enumerate(future_preds[:30])]
                        time_range = f"{float(first_fut):.2f} ~ {float(first_fut) + val_step * len(future_preds):.2f} ({xlabel or '时间'})"
                    else:
                        rows = [f"  第 {i+1:3d} 步  →  **{val:.4f}**" for i, val in enumerate(future_preds[:30])]
                        time_range = f"第 {len(y)} 步 ~ 第 {len(y)+len(future_preds)-1} 步"
                    forecast_text = f"**未来 {len(future_preds)} 步预测值（{target_col}）：{time_range}**\n\n" + "\n".join(rows)
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
                        f"**{target_col} 未来 {n_steps} 步趋势总结**（{time_range}）\n\n"
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
                
                # output_dir 已在函数开头统一创建，直接使用已保存的 PNG
                # forecast_plot 已经是 str(output_dir / "forecast.png")
                # 确保 PNG 文件存在（万一代码路径跳过了保存，补救性保存）
                forecast_png_path = str(output_dir / "forecast.png")
                if not Path(forecast_png_path).is_file():
                    fig = select_plot_function(
                        chart_requirement, hist, future_preds,
                        target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, std_val, xlabel
                    )
                    fig.savefig(forecast_png_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    forecast_plot = forecast_png_path
                # 注意：不再重复创建 output_dir，zip 使用已存在的那个
                
                # 保存预测数据 CSV
                all_steps = steps_hist + steps_fut
                all_vals = list(hist) + list(future_preds)
                types_ = ['历史'] * len(steps_hist) + ['预测'] * len(steps_fut)
                # 包含真实日期的 CSV（如果可用）
                if xtick_hist and xtick_fut:
                    all_labels = xtick_hist + xtick_fut
                    csv_df = pd.DataFrame({
                        'date': all_labels,
                        'step': all_steps,
                        'type': types_,
                        target_col: all_vals
                    })
                else:
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
        
        # 修复：更严格的 guard，确保 predictor 和 lstm_pred 都是 None 或未训练时提前返回
        predictor_ready = predictor is not None and predictor.is_fitted
        lstm_ready = lstm_pred is not None and lstm_pred.is_fitted
        if not predictor_ready and not lstm_ready:
            return "❌ 请先训练模型", None
        
        if file is None:
            return "❌ 请上传预测数据文件", None
        
        try:
            file_path = extract_file_path(file)
            if not file_path:
                return "❌ 无法提取文件路径", None
            df = pd.read_csv(file_path)
        except Exception as e:
            return f"❌ 读取文件失败: {e}", None
        
        result_df = None
        if predictor_ready:
            cols = [c for c in predictor.feature_names if c in df.columns]
            if not cols:
                return "❌ 新数据中没有匹配的特征列", None
            pred = predictor.predict(df[cols])
            if pred is None:
                print("预测返回 None")
                return "❌ 预测失败：模型返回空结果", None
            result_df = df.copy()
            result_df['预测值'] = pred
        elif lstm_ready:
            X = df.values.astype(np.float32)
            pred = lstm_pred.predict(X)
            if pred is None:
                print("LSTM 预测返回 None")
                return "❌ 预测失败：模型返回空结果", None
            result_df = df.copy()
            result_df['预测值'] = pred
        
        # 双重保险：result_df 仍未定义则返回错误
        if result_df is None:
            return "❌ 未找到可用的训练模型", None
        
        table = result_df.tail(30).to_html(max_cols=15, classes='table table-striped')
        return "✅ 预测完成", table
    
    # 绑定事件
    file_input.change(
        on_file_upload,
        inputs=[file_input],
        outputs=[data_preview, data_info, data_structure_info, data_decouple_info, feature_col, target_col, model_recommend, model_select],
        queue=False  # 禁用队列避免 Gradio 6.x FileData meta 序列化错误
    )
    train_btn.click(
        on_train,
        inputs=[feature_col, target_col, predict_mode, model_select, requirement, n_future_val, n_future_unit, chart_requirement],
        outputs=[result_out, config_out, importance_out, forecast_plot, forecast_out, summary_out, download_file]
    )
    download_btn.click(
        on_download,
        inputs=[download_file],
        outputs=[download_file]
    )
    predict_btn.click(on_predict, inputs=[predict_file], outputs=[predict_status, predict_out], queue=False)


if __name__ == "__main__":
    # Railway / HuggingFace Spaces 用 PORT 环境变量
    import os
    port = int(os.environ.get("PORT", 7860))
    print("=" * 60)
    print("  DeepPredict Web 版 v1.04 已启动！")
    print(f"  访问地址: http://127.0.0.1:{port}")
    print("  同一局域网内的手机/电脑都可以访问")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        show_error=True
    )
