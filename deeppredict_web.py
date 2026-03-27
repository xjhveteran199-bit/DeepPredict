"""
ChronoML Web 版 - Gradio 界面 v1.5
新增:智能预测向导(5步用户引导流程)、CSV解析引擎、数据分析引擎、推荐引擎、自动特征工程
修复:LSTM matplotlib后端、EnhancedCNN1D预测维度错误、PatchTST predict_future、多处NaN处理
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# matplotlib 中文字体配置(解决 DejaVu Sans 缺失中文字形问题)
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号
mpl.rcParams['savefig.dpi'] = 300  # 300 DPI 高清导出
mpl.rcParams['figure.dpi'] = 150

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from gradio.data_classes import ListFiles, FileData

# 设置 logger
logger = logging.getLogger("ChronoML")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============ 全局状态存储（避免 gr.State DataFrame 序列化问题）============
# gr.State 在 Gradio 6.x 中无法正确序列化 pandas DataFrame，
# 因此 DataFrame 等大对象存于此模块级变量，state 只存简单值。
_wizard_data = {
    'df_parsed': None,
    'file_path': None,
    'analysis': None,
    'recommendation': None,
    'final_config': None,
    'step': 0,
}


# ============ 工具函数 ============
def extract_file_path(file_obj):
    """从 Gradio 6.x File 组件返回值中提取文件路径字符串。

    Gradio 6.x 返回 ListFiles/FileData/dict/str 等多种格式，此函数统一处理。
    返回值永远是一个有效的文件路径字符串，如果不是文件则返回 None。
    """
    if file_obj is None:
        return None

    # ListFiles (Gradio 6.x Pydantic root model)
    if hasattr(file_obj, 'root'):
        file_obj = file_obj.root

    # 普通列表：取第一个有效元素
    if isinstance(file_obj, list):
        for item in file_obj:
            result = extract_file_path(item)
            if result:
                return result
        return None

    # dict（序列化后的 FileData）
    if isinstance(file_obj, dict):
        path = str(file_obj.get('path', ''))
        if path:
            p = Path(path)
            if p.is_file():
                return str(p)
            # path 可能是目录，尝试用 orig_name
            orig_name = file_obj.get('orig_name', '')
            if orig_name:
                real = p / orig_name
                if real.is_file():
                    return str(real)
                real2 = p.parent / orig_name
                if real2.is_file():
                    return str(real2)
        return None

    # FileData Pydantic 对象
    if hasattr(file_obj, 'path'):
        path = str(file_obj.path)
        p = Path(path)
        if p.is_file():
            return path
        # path 可能是 Gradio 缓存目录，尝试用 orig_name
        orig_name = getattr(file_obj, 'orig_name', None)
        if orig_name:
            real = p / orig_name
            if real.is_file():
                return str(real)
            real2 = p.parent / orig_name
            if real2.is_file():
                return str(real2)
        return None

    # 已经是字符串
    if isinstance(file_obj, str):
        p = Path(file_obj)
        if p.is_file():
            return str(p)
        return None

    return None


def _preview_md(df):
    """将 pandas DataFrame 转换为 Markdown 表格字符串，供 gr.Markdown 组件使用。"""
    if df is None or df.empty:
        return "*（无预览数据）*"
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return header + "\n" + sep + "\n" + "\n".join(rows)


# ============ 核心模块 ============

from src.data.data_decoupler import DataDecoupler


def plot_bland_altman(y_true, y_pred, title="Bland-Altman"):
    """Bland-Altman 一致性分析图"""
    from matplotlib.figure import Figure
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    mean_arr = (y_pred_arr + y_true_arr) / 2
    diff_arr = y_pred_arr - y_true_arr
    mean_diff = np.mean(diff_arr)
    std_diff = np.std(diff_arr, ddof=1)
    loa_lo = mean_diff - 1.96 * std_diff
    loa_hi = mean_diff + 1.96 * std_diff
    fig = Figure(figsize=(7, 5), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(mean_arr, diff_arr, alpha=0.6, s=20, color="#4DBBD5")
    ax.axhline(mean_diff, color="#E64B35", lw=2, label="Mean={:.4f}".format(mean_diff))
    ax.axhline(loa_lo, color="#8491B4", lw=1.5, linestyle="--", label="95% LoA=[{:.4f}, {:.4f}]".format(loa_lo, loa_hi))
    ax.axhline(loa_hi, color="#8491B4", lw=1.5, linestyle="--")
    ax.axhline(0, color="gray", lw=1, linestyle=":", alpha=0.7)
    ax.set_xlabel("(Predicted + Actual) / 2", fontsize=11)
    ax.set_ylabel("Predicted - Actual", fontsize=11)
    ax.set_title(title + " - Bland-Altman Analysis", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
            return True, f"✅ 加载成功:{self.df.shape[0]}行 × {self.df.shape[1]}列"
        except Exception as e:
            return False, f"❌ {str(e)}"

    def _analyze(self):
        """分析数据结构"""
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(self.df.select_dtypes(include=['object', 'string']).columns)

        # 分析数据结构
        self.data_structure = self._detect_structure()

    def _detect_structure(self):
        """检测数据结构类型"""
        n_num = len(self.numeric_cols)
        n_cat = len(self.categorical_cols)
        n_rows = len(self.df)

        # 检查是否有重复的时间列(可能是宽表格式的多组时序)
        time_cols = [c for c in self.numeric_cols if 'time' in c.lower() or '日期' in c.lower() or 'date' in c.lower()]

        # 检查是否有类似 "K-with", "K-without", "Glu-with" 这种成组的列名
        data_cols = [c for c in self.numeric_cols if c not in time_cols]

        # 成组分析:检查是否有相同前缀
        groups = {}
        for col in data_cols:
            # 提取前缀(-前的部分)
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
            structure['type'] = 'grouped_time_series'  # 成组时序(如 K-xxx, Glu-xxx)
        elif n_num >= 2:
            structure['type'] = 'multi_variable'  # 多变量(特征→目标)
        else:
            structure['type'] = 'unknown'

        return structure

    def get_info(self):
        if self.df is None:
            return ""
        return (
            f"**数据形状**:{self.df.shape[0]} 行 × {self.df.shape[1]} 列\n\n"
            f"**数值列**({len(self.numeric_cols)}):`{', '.join(self.numeric_cols[:5])}{'...' if len(self.numeric_cols)>5 else ''}`\n\n"
            f"**类别列**({len(self.categorical_cols)}):`{', '.join(self.categorical_cols[:5])}{'...' if len(self.categorical_cols)>5 else ''}`"
        )

    def get_structure_explanation(self):
        """返回数据结构的中文解释"""
        if self.data_structure is None:
            return "请先上传数据"

        s = self.data_structure
        lines = []

        lines.append(f"**📊 数据结构检测结果**:")
        lines.append("")

        if s['type'] == 'single_variable':
            lines.append("✅ **单变量时序数据**(推荐使用 PatchTST/LSTM)")
            lines.append(f"   - 数据点:{s['n_rows']} 条")
            lines.append(f"   - 数值列:{s['data_cols']}")
            lines.append("")
            lines.append("**使用建议**:用该列自己的历史值预测未来值")

        elif s['type'] == 'grouped_time_series':
            lines.append("⚠️ **成组时序数据**")
            lines.append(f"   - 数据点:{s['n_rows']} 条")
            lines.append(f"   - 检测到 {len(s['groups'])} 组数据:")
            for group, cols in s['groups'].items():
                lines.append(f"     • {group}: {cols}")
            lines.append("")
            lines.append("**使用建议**:")
            lines.append("   1. 选择其中一组的目标列(如 K-with epifluidics)")
            lines.append("   2. 系统会用该组自己的历史值预测未来")
            lines.append("   3. 同一组内的其他列(如 K-without)可用于多变量预测")

        elif s['type'] == 'multi_variable':
            lines.append("📈 **多变量数据**")
            lines.append(f"   - 数据点:{s['n_rows']} 条")
            lines.append(f"   - 数值列:{s['data_cols']}")
            lines.append("")
            lines.append("**使用建议**:选择一个目标列,其他列作为特征")

        else:
            lines.append(f"**数据类型**:{s['type']}")
            lines.append(f"**数值列**:{s['numeric_cols']}")

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
        self.train_history = None  # 训练历史(epoch -> accuracy/loss)

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
                return False, f"❌ y 必须是 1D 数组,当前 shape={np.asarray(y).shape}"

            # P2 修复:只对数值列做 fillna,避免 Date/字符串列导致 TypeError
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.shape[1] == 0:
                return False, "❌ 选中特征中没有数值列,无法训练", {}
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_numeric.fillna(X_numeric.median()))

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

            # P0 修复:LogisticRegression 是分类器,不能用于回归
            if model_name == 'LogisticRegression' and task_type == 'regression':
                return False, "❌ LogisticRegression 是分类器,不能用于回归任务。请选择 LinearRegression、Ridge 或 ElasticNet。", {}

            key = (model_name, task_type)
            model_cls = models.get(key)
            self.model = model_cls(**(params or {}))
            # 训练历史追踪
            n_est = params.get("n_estimators", 50) if hasattr(self.model, "n_estimators") else 1
            max_show = min(10, n_est) if n_est > 1 else 1
            self.train_history = {"epoch": list(range(1, max_show + 1))}
            if task_type == "classification":
                from sklearn.metrics import accuracy_score
                if max_show == 1:
                    self.model.fit(X_train, y_train)
                    train_pred = self.model.predict(X_train)
                    self.train_history["accuracy"] = [round(accuracy_score(y_train, train_pred), 6)]
                else:
                    self.train_history["accuracy"] = []
                    for ep in range(max_show):
                        self.model.fit(X_train, y_train)
                        train_pred = self.model.predict(X_train)
                        self.train_history["accuracy"].append(round(accuracy_score(y_train, train_pred), 6))
            else:
                from sklearn.metrics import mean_squared_error
                if max_show == 1:
                    self.model.fit(X_train, y_train)
                    train_pred = self.model.predict(X_train)
                    self.train_history["loss"] = [round(np.sqrt(mean_squared_error(y_train, train_pred)), 6)]
                else:
                    self.train_history["loss"] = []
                    for ep in range(max_show):
                        self.model.fit(X_train, y_train)
                        train_pred = self.model.predict(X_train)
                        self.train_history["loss"].append(round(np.sqrt(mean_squared_error(y_train, train_pred)), 6))

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
                self.metrics = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
                m_str = f"**R2**={r2:.4f} **RMSE**={rmse:.4f} **MAE**={mae:.4f}"

            self.is_fitted = True
            return True, f"✅ {model_name} 训练完成\n{m_str}"

        except Exception as e:
            return False, f"❌ {str(e)}"

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("请先训练模型")
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric.fillna(X_numeric.median()))
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
        滚动预测未来 steps 步(用于时序预测)
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
                # 更新:如果是单变量时序(feature_names 只有一列),直接替换
                # 如果是多变量,需要把预测值更新回去(这里简化处理:假设单变量)
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
    从 datetime 列构建真实时间轴,返回:
    steps_*, xtick_*, date_fmt, first_date, first_future_date
    steps_* = 数字索引(用于绘图定位)
    xtick_* = 格式化日期字符串(用于 X 轴标签)
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
                # 如果解析出来的日期全在1975年之前且跨度<1天,说明是数值列误识别为日期
                parsed_years = parsed.dropna().dt.year
                parsed_range = (parsed.dropna().max() - parsed.dropna().min()).total_seconds()
                if len(parsed_years) > 0 and parsed_years.max() <= 1975 and parsed_range < 86400:
                    # 数值列(分钟/秒)误识别为日期,退回数值模式
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
        # ===== 数值/序号模式:直接用时间值作刻度位置,避免索引偏移 =====
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

            # 历史:用实际时间值作为 X 轴位置
            steps_hist = list(last_vals.values)  # 直接用时间值
            # 均匀取刻度(~6个历史标签,~8个未来标签,避免X轴重叠)
            hist_step = max(1, last_n // 6)  # ~6 labels for historical
            xtick_hist = [f"{last_vals.iloc[i]:.1f}" if i % hist_step == 0 else "" for i in range(last_n)]

            # 未来:也用实际时间值
            future_vals = [last_time + val_step * (i + 1) for i in range(n_fut)]
            steps_fut = future_vals  # 直接用时间值,不再用行号
            fut_step = max(1, n_fut // 8)  # ~8 labels for future
            xtick_fut = [f"{future_vals[i]:.1f}" if i % fut_step == 0 else "" for i in range(n_fut)]

            xlabel = feature_col
            return steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel, last_vals.iloc[0], future_vals[0] if future_vals else None
        else:
            # 完全没有可用列:退化为序号
            steps_hist = list(range(0, n_hist))
            steps_fut = list(range(n_hist, n_hist + n_fut))
            step = max(1, n_hist // 12)
            xtick_hist = [str(i) if i % step == 0 else "" for i in steps_hist]
            step_fut = max(1, n_fut // 12)
            xtick_fut = [str(i) if i % step_fut == 0 else "" for i in steps_fut]
            return steps_hist, steps_fut, xtick_hist, xtick_fut, None, None, None


def _apply_xticks(ax, steps_hist, steps_fut, xtick_hist, xtick_fut, has_datetime):
    """统一设置 X 轴刻度标签(数字或真实日期)"""
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
    """根据用户描述选择图表模板(纯规则匹配,无需 AI)"""
    # 只有 date_fmt 是真正的日期格式字符串(如 "%Y-%m")才算 datetime 模式
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
    """标准折线图:蓝色历史 + 橙色预测,支持真实日期/数值时间轴"""
    import matplotlib.pyplot as plt
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制分隔线(历史与预测的边界)
    if has_xtick and steps_hist and steps_fut:
        # 数值模式或日期模式:分隔线在最后一个历史时间点
        boundary = float(steps_hist[-1])
        ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5, label='预测起点')
    elif not has_xtick:
        ax.axvline(x=len(steps_hist) - 0.5, color='gray', linestyle=':', linewidth=1)

    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=2, label='历史数据')
    ax.plot(steps_fut, future_preds, color='#FF6B2B', linewidth=2, linestyle='--', label='预测')

    # X轴刻度:只显示有标签的位置,彻底避免重叠
    if has_xtick:
        # 历史部分:取所有非空标签的步进
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
    """双轴图:左轴历史,右轴预测"""
    import matplotlib.pyplot as plt
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1, color2 = '#3B82F6', '#FF6B2B'

    # 分隔线
    if has_xtick and steps_hist and steps_fut:
        boundary = float(steps_hist[-1])
        ax1.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5)

    # X轴刻度(精确控制)
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
    """带置信区间的预测图,支持真实日期"""
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
    ax.set_title(f'{target_col} 趋势预测(含置信区间)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scatter_with_line(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, xlabel=None):
    """散点+折线组合图,支持真实日期"""
    import matplotlib.pyplot as plt
    is_date_fmt = xlabel and '%' in str(xlabel)
    has_xtick = xtick_hist is not None and any(xtick_hist)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(steps_hist, hist, color='#3B82F6', s=20, zorder=3, label='历史数据(散点)')
    ax.plot(steps_hist, hist, color='#3B82F6', linewidth=1.5, alpha=0.6)
    ax.scatter(steps_fut, future_preds, color='#FF6B2B', s=30, marker='D', zorder=3, label='预测(散点)')
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
    ax.set_title(f'{target_col} 趋势预测(散点图)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bar_forecast(hist, future_preds, target_col, steps_hist, steps_fut, xtick_hist=None, xtick_fut=None, xlabel=None):
    """柱状图(历史用柱状,预测用折线),支持真实日期"""
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
    ax.set_title(f'{target_col} 趋势预测(柱状图)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


# ============ Gradio 界面 ============

with gr.Blocks(title="ChronoML v1.5 - 零门槛时序预测工具") as demo:

    # 首页介绍
    with gr.Tab("🏠 首页介绍"):
        gr.Markdown("""
        # 🧠 ChronoML - 零门槛时序预测工具

        ### 告别复杂代码,3步完成AI预测!

        ---

        ## ✨ 核心优势

        | 功能 | 说明 |
        |------|------|
        | 🤖 AI自动选模型 | 上传数据后,系统自动推荐最优模型(PatchTST/LSTM/GradientBoosting) |
        | 📊 零代码操作 | 无需编程基础,点点鼠标即可完成时序预测 |
        | 🔒 数据本地处理 | 数据不上传服务器,保护隐私安全 |
        | 📱 支持手机访问 | 响应式设计,随时随地使用 |

        ---

        ## 🚀 快速开始

        1. 点击「**工具使用**」标签页
        2. 上传你的 CSV 数据文件
        3. 选择目标列,点击「开始训练」

        ---

        ## 💡 适用场景

        - 🧬 **生物医学**:细胞培养数据、药物反应预测
        - 📈 **市场分析**:销售预测、流量预测
        - 🌡️ **环境科学**:空气质量、气候变化预测
        - 🏭 **工业生产**:设备故障预测、质量控制

        ---

        ## 💰 定价方案

        | 版本 | 价格 | 说明 |
        |------|------|------|
        | 个人版 | **免费** | 适合学习和研究 |
        | 预测服务 | **¥5/次** | 单次预测,不限数据量 |
        | 批量服务 | **¥99/月** | 无限次预测 + 优先模型 |

        ---

        *如有疑问或定制需求,请联系开发者*
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

                    gr.Markdown("### 📊 数据预览(均匀采样200行,覆盖全部时间范围)")
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
                        label="1 选择特征列 X(自变量/时间列,可选)",
                        choices=[],
                        info="选择作为时间轴或自变量的列(如日期、序号),不选则自动用行号"
                    )

                    target_col = gr.Dropdown(
                        label="2 选择目标列 Y(要预测什么，可多选)",
                        choices=[],
                        info="选择你要预测的目标变量，多选可实现多目标同时预测",
                        multiselect=True
                    )

                    predict_mode = gr.Radio(
                        choices=["单变量时序(推荐)", "多变量预测"],
                        value="单变量时序(推荐)",
                        label="3 预测模式",
                        info="时序数据用单变量,分类/回归可用多变量"
                    )

                    model_select = gr.Dropdown(
                        label="4 选择模型",
                        choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"],
                        value="自动推荐",
                        info="自动推荐会根据数据情况选择最优模型"
                    )
                    model_recommend = gr.Markdown("")

                    requirement = gr.Textbox(
                        label="5 描述需求(可选)",
                        placeholder="示例:预测K值变化趋势\n预测Glu未来走势\n判断是否流失",
                        lines=2
                    )

                    with gr.Row():
                        n_future_val = gr.Number(
                            label="6 预测未来时间长度",
                            value=10,
                            info="数值,默认10",
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
                        label="7 描述你想要的图表(可选)",
                        placeholder="示例:双轴图,左边显示温度走势,右边显示湿度\n用红色虚线标注预测区间上下界\n标注关键的拐点和异常值",
                        lines=2
                    )

                # === 结果展示 ===
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 任务配置预览")
                    config_out = gr.Markdown("*配置信息将显示在这里*")

                    gr.Markdown("### 📊 训练结果(模型评价)")
                    result_out = gr.Textbox(label="", lines=6, interactive=False)

                    gr.Markdown("### 📉 训练历史曲线(Loss & R²)")
                    training_history_plot = gr.Plot(label="训练历史")

                    gr.Markdown("### 📈 未来趋势预测")
                    forecast_plot = gr.Image(label="趋势预测图(蓝色=历史,橙色=预测)")

                    gr.Markdown("### 🔮 未来预测值")
                    forecast_out = gr.Textbox(label="", lines=10, interactive=False)

                    gr.Markdown("### 💬 趋势结论")
                    summary_out = gr.Markdown("*训练完成后自动生成趋势总结*")

                    gr.Markdown("### 🔍 特征重要性")
                    importance_out = gr.Markdown("")

                    gr.Markdown("### 📥 下载完整结果包")
                    with gr.Row():
                        download_btn = gr.Button("📥 下载结果包(PNG + CSV + JSON)", variant="secondary", size="lg", scale=2)
                        download_file = gr.File(label="点击下载 zip", interactive=False, scale=1)

            gr.Markdown("---")

            with gr.Row():
                train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg", scale=1)

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("### 🔮 新数据预测")
                predict_file = gr.File(label="上传新数据(可选)", file_types=[".csv"], scale=1)
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
                "**推荐模型**:上传数据后自动推荐",
                gr.update(choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"], value="自动推荐")
            ]

        # Gradio 6.x + gradio_client 场景: string path 直接传入，不走 FileData 解析
        if isinstance(file, str):
            file_path = file if Path(file).is_file() else None
        else:
            file_path = extract_file_path(file)
        if not file_path:
            return [None] * 7 + [
                "**文件路径无效**",
                "**上传失败**",
                "**上传后自动识别列类型**",
                gr.update(choices=[]),
                gr.update(choices=[]),
                "**推荐模型**:上传失败",
                gr.update(choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"], value="自动推荐")
            ]
        success, msg = data_loader.load_csv(file_path)

        if success:
            # 均匀采样200行:覆盖完整时间范围(10~50min),而非只取前200行(仅覆盖10~13min)
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
            decouple_info = "无数值列,无需解耦"
            if data_loader.numeric_cols:
                try:
                    data_decoupler = DataDecoupler()
                    default_y = data_loader.numeric_cols[0]
                    data_decoupler.fit(data_loader.df, target_col=default_y)
                    decouple_info = data_decoupler.get_summary()
                except Exception as e:
                    decouple_info = f"解耦分析失败:{str(e)[:80]}"

            data_size = data_loader.df.shape[0]
            data_type = data_loader.data_structure['type'] if data_loader.data_structure else 'unknown'

            if data_type == 'single_variable' and data_size >= 200:
                recommend = "**推荐模型**:PatchTST(单变量时序,≥200条数据)"
                recommend_value = "PatchTST"
            elif data_type == 'single_variable' and data_size >= 100:
                recommend = "**推荐模型**:LSTM(单变量时序,100-200条数据)"
                recommend_value = "LSTM"
            elif data_type in ['grouped_time_series', 'multi_variable'] and data_size >= 200:
                recommend = "**推荐模型**:EnhancedCNN1D(多变量/复杂数据,推荐)"
                recommend_value = "EnhancedCNN1D"
            elif data_type in ['grouped_time_series', 'multi_variable'] and data_size >= 100:
                recommend = "**推荐模型**:EnhancedCNN1D(多变量,k=3/5/7多尺度卷积)"
                recommend_value = "EnhancedCNN1D"
            else:
                recommend = "**推荐模型**:GradientBoosting(数据量较小或非时序任务)"
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
            "**推荐模型**:上传失败",
            gr.update(choices=["自动推荐", "PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting", "RandomForest"], value="自动推荐")
        ]

    def plot_training_history(history, target_col, output_dir=None):
        """绘制训练历史曲线（Loss + R²），Nature 配色，300 DPI"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os

            if output_dir is None:
                output_dir = Path(__file__).parent.resolve() / "outputs"

            # Nature 配色方案
            NATURE_BLUE = "#3B82F6"
            NATURE_RED = "#EF4444"
            NATURE_GREEN = "#10B981"
            NATURE_ORANGE = "#F59E0B"
            NATURE_GRAY = "#6B7280"

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

            # 左图：Loss
            ax_loss = axes[0]
            # 支持 PyTorch 格式(train_loss/val_loss)和 sklearn 格式(loss/accuracy)
            train_losses = history.get('train_loss', history.get('loss', []))
            val_losses = history.get('val_loss', [])
            train_r2s = history.get('train_r2', history.get('accuracy', []))
            val_r2s = history.get('val_r2', [])

            epochs = history.get('epoch', list(range(1, len(train_losses) + 1)))
            ax_loss.plot(epochs, train_losses, color=NATURE_BLUE,
                         linewidth=2, marker='o', markersize=3, label='Train Loss')
            if val_losses:
                ax_loss.plot(epochs, val_losses, color=NATURE_RED,
                             linewidth=2, marker='s', markersize=3, label='Val Loss')
            ax_loss.set_xlabel('Epoch', fontsize=12)
            ax_loss.set_ylabel('Loss (MSE)', fontsize=12)
            ax_loss.set_title(f'{target_col} - Training Loss', fontsize=13, fontweight='bold')
            ax_loss.legend(fontsize=10)
            ax_loss.grid(True, alpha=0.3)
            ax_loss.set_facecolor('#FAFAFA')

            # 右图：R²
            ax_r2 = axes[1]
            if train_r2s:
                ax_r2.plot(epochs, train_r2s, color=NATURE_GREEN,
                           linewidth=2, marker='o', markersize=3, label='Train R²')
            if val_r2s:
                ax_r2.plot(epochs, val_r2s, color=NATURE_ORANGE,
                           linewidth=2, marker='s', markersize=3, label='Val R²')
            ax_r2.set_xlabel('Epoch', fontsize=12)
            ax_r2.set_ylabel('R² Score', fontsize=12)
            ax_r2.set_title(f'{target_col} - Accuracy (R²)', fontsize=13, fontweight='bold')
            ax_r2.legend(fontsize=10)
            ax_r2.grid(True, alpha=0.3)
            ax_r2.set_facecolor('#FAFAFA')

            # 如果有 MAE，在右图叠加
            val_maes = history.get('val_mae', [])
            if val_maes and val_r2s:
                ax_mae = ax_r2.twinx()
                ax_mae.plot(epochs, val_maes, color=NATURE_GRAY,
                            linewidth=1.5, linestyle='--', marker='^', markersize=2, label='Val MAE')
                ax_mae.set_ylabel('MAE', fontsize=10, color=NATURE_GRAY)
                ax_mae.tick_params(axis='y', labelcolor=NATURE_GRAY)

            plt.tight_layout()

            save_path = output_dir / 'training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"Training history plot saved: {save_path}")
            return str(save_path)
        except Exception as e:
            logger.warning(f"Failed to plot training history: {e}")
            return None

    def on_train(feature_col, target_col, predict_mode, model_select, requirement, n_future_val, n_future_unit, chart_requirement, prog=gr.Progress()):
        global predictor, lstm_pred

        if data_loader.df is None:
            return ["❌ 请先上传数据", "", "", None, "", "", None, None]

        if not target_col:
            return ["❌ 请选择目标列 Y", "", "", None, "", "", None, None]

        # 处理 target_col 多选情况
        if isinstance(target_col, list):
            target_cols_list = target_col if target_col else []
            target_col_display = ','.join(target_cols_list[:3]) + ('...' if len(target_cols_list) > 3 else '')
        else:
            target_cols_list = [target_col] if target_col else []
            target_col_display = str(target_col) if target_col else ''

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

        # 多选时检查所有目标列是否有效
        if isinstance(target_cols_list, list) and len(target_cols_list) > 1:
            invalid = [c for c in target_cols_list if c not in numeric_cols]
            if invalid:
                return [f"❌ 以下目标列不是数值列: {invalid}", "", "", None, "", "", None, None]
        elif target_cols_list and target_cols_list[0] not in numeric_cols:
            return [f"❌ 目标列 [{target_cols_list[0]}] 不是数值列", "", "", None, "", "", None, None]

        # 根据预测模式决定数据准备方式
        data_type = data_loader.data_structure['type'] if data_loader.data_structure else 'unknown'
        use_univariate = (predict_mode == "单变量时序(推荐)") or (data_type == 'single_variable')

        if use_univariate:
            # 单变量时序:用 Y 自己的历史值预测未来（支持多选）
            y_df = data_loader.df[target_cols_list if isinstance(target_cols_list, list) else [target_cols_list]]
            y = y_df.values.astype(np.float32)
            if feature_col and feature_col in data_loader.df.columns:
                # 用户指定了 X 时间轴:按 X 排序后做预测
                x_raw = data_loader.df[feature_col].values
                try:
                    # 尝试转为数值排序
                    x_numeric = pd.to_numeric(pd.Series(x_raw), errors='coerce').fillna(0).values.astype(np.float32)
                    sort_idx = np.argsort(x_numeric)
                    y = y[sort_idx]
                    feature_cols_used = f"X={feature_col}(用户指定时间轴)"
                except Exception:
                    y = y_df.values.astype(np.float32)
                    feature_cols_used = "(X列无法排序,使用行号)"
            else:
                # 未指定 X:用行号(默认行为)
                feature_cols_used = "(无,自变量使用行号)"
            # X 在这里只用于排序,模型只接收 y
            X_for_model = y  # 单变量:X=y
        else:
            # 多变量模式:用其他数值列作为特征(X 列也可以参与)
            first_target = target_cols_list[0] if isinstance(target_cols_list, list) else target_cols_list
            feature_cols = [c for c in numeric_cols if c not in target_cols_list]
            if feature_col and feature_col in numeric_cols and feature_col not in target_cols_list:
                # 用户指定的 X 也作为特征加入
                feature_cols = [feature_col] + [c for c in feature_cols if c != feature_col]
            if not feature_cols:
                # 没有其他特征列,退化为单变量
                y_df = data_loader.df[target_cols_list if isinstance(target_cols_list, list) else [target_cols_list]]
                X_for_model = y = y_df.values.astype(np.float32)
                feature_cols_used = "(无,使用自身历史值)"
            else:
                X_for_model = data_loader.df[feature_cols].values.astype(np.float32)
                y_df = data_loader.df[target_cols_list if isinstance(target_cols_list, list) else [target_cols_list]]
                y = y_df.values.astype(np.float32)
                feature_cols_used = str(feature_cols)

        prog(0.3, desc=f"训练 {model_name}...")

        if model_name == 'PatchTST':
            try:
                from src.models.patchtst_model import PatchTSTPredictor
                lstm_pred = PatchTSTPredictor()
                first_target = target_cols_list[0] if isinstance(target_cols_list, list) and target_cols_list else target_col
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
                    learning_rate=params.get('learning_rate', 0.0005),
                    target_col=first_target
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
            f"**任务类型**:{task_type}\n"
            f"**模型**:{model_name}\n"
            f"**自变量 X**:{feature_col or '(行号)'}\n"
            f"**因变量 Y**:{target_col}\n"
            f"**特征列**:{feature_cols_used}\n"
            f"**参数**:{params}"
        )

        importance = ""
        if predictor and predictor.is_fitted:
            imp = predictor.get_importance()
            if imp:
                top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                importance = "\n".join([f"`{k}`: {v:.4f}" for k, v in top10])

        # ===== 统一保存图表的目录 ======
        # 在任何模型分支之前创建,确保所有代码路径都能统一保存
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mpl.rcParams['text.usetex'] = False  # 禁用 LaTeX,纯 matplotlib 渲染
        # 使用绝对路径确保 Gradio 6.x 能正确服务文件
        base_dir = Path(__file__).parent.resolve()
        output_dir = base_dir / "outputs" / f"{target_col_display or target_col}_{model_name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
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
        # 预初始化的绘图变量(供 zip 导出使用)
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
                    hist_full = y[-last_n:]
                    # 多目标：取第一个目标用于绘图展示
                    hist_1d = hist_full[:, 0] if hist_full.ndim > 1 else hist_full
                    future_preds_1d = future_preds[:, 0] if future_preds.ndim > 1 else future_preds

                    # 构建真实时间轴(支持日期列和数值列)
                    d_steps = build_datetime_steps(data_loader.df, feature_col, last_n, len(future_preds_1d))
                    steps_hist, steps_fut, xtick_hist, xtick_fut, xlabel, first_date, first_fut = d_steps
                    std_val = np.std(future_preds_1d) * 0.5

                    prog(0.65, desc="生成图表...")
                    first_target = target_cols_list[0] if isinstance(target_cols_list, list) and target_cols_list else (target_col if isinstance(target_col, str) else str(target_col))
                    forecast_plot = select_plot_function(
                        chart_requirement, hist_1d, future_preds_1d,
                        first_target, steps_hist, steps_fut, xtick_hist, xtick_fut, std_val, xlabel
                    )
                    # 立即保存为PNG并转换为路径字符串,避免返回matplotlib Figure导致Gradio postprocess错误
                    forecast_plot.savefig(output_dir / "forecast.png", dpi=300, bbox_inches='tight')
                    plt.close(forecast_plot)
                    forecast_plot = str(output_dir / "forecast.png")

                    # 2. 预测值表格(显示真实日期或数值时间轴)
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
                        # 数值时间轴模式(分钟/秒等)
                        val_step = float(first_fut - first_date) if first_date is not None and first_fut != first_date else 1.0
                        for i, val in enumerate(future_preds):
                            future_val = float(first_fut) + val_step * (i + 1)
                            rows.append(f"  {future_val:.2f}  →  **{val:.4f}**")
                        time_range = f"{float(first_fut):.2f} ~ {float(first_fut) + val_step * len(future_preds):.2f} ({xlabel or '时间'})"
                    else:
                        for i, val in enumerate(future_preds):
                            rows.append(f"  第 {i+1:3d} 步  →  **{val:.4f}**")
                        time_range = f"第 {len(y)} 步 ~ 第 {len(y)+len(future_preds)-1} 步"
                    forecast_text = f"**未来 {len(future_preds)} 步预测值({target_col}):{time_range}**\n\n" + "\n".join(rows[:30])
                    if len(future_preds) > 30:
                        forecast_text += f"\n  ...(共 {len(future_preds)} 步,已截取前30步)"

                    # 3. 自然语言总结
                    first_val = float(future_preds[0])
                    last_val = float(future_preds[-1])
                    change = last_val - first_val
                    pct = (change / abs(first_val) * 100) if first_val != 0 else 0

                    # 判断趋势
                    if change > 0:
                        trend = "📈 **上升趋势**"
                        trend_desc = f"从 {first_val:.4f} 上涨到 {last_val:.4f},涨幅 **{abs(change):.4f}**({abs(pct):.1f}%)"
                    elif change < 0:
                        trend = "📉 **下降趋势**"
                        trend_desc = f"从 {first_val:.4f} 下降到 {last_val:.4f},跌幅 **{abs(change):.4f}**({abs(pct):.1f}%)"
                    else:
                        trend = "➡️ **基本平稳**"
                        trend_desc = f"基本维持在 {last_val:.4f} 附近"

                    # 波动分析
                    diffs = [future_preds[i+1] - future_preds[i] for i in range(len(future_preds)-1)]
                    volatility = sum(1 for d in diffs if abs(d) > 0.05 * abs(first_val)) / max(1, len(diffs))

                    summary_text = (
                        f"**{target_col} 未来 {n_steps} 步趋势总结**({time_range})\n\n"
                        f"**趋势方向**:{trend}\n\n"
                        f"{trend_desc}\n\n"
                        f"**预测区间**:{min(future_preds):.4f} ~ {max(future_preds):.4f}\n\n"
                        f"**预测均值**:{sum(future_preds)/len(future_preds):.4f}\n\n"
                        f"**稳定性**:{'波动较小,趋势较稳定' if volatility < 0.3 else '有一定波动,请结合实际判断'}\n\n"
                        f"💡 *以上为模型自动预测结果,仅供参考,实际走势可能受外部因素影响。*"
                    )
            except Exception as e:
                forecast_text = f"趋势预测生成失败:{str(e)}"
                summary_text = "*趋势预测生成失败,请查看上方训练结果*"

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
                    # 立即保存为PNG并转换为路径字符串,避免返回matplotlib Figure导致Gradio postprocess错误
                    forecast_plot.savefig(output_dir / "forecast.png", dpi=300, bbox_inches='tight')
                    plt.close(forecast_plot)
                    forecast_plot = str(output_dir / "forecast.png")

                    # 预测值表格(显示真实日期或数值时间轴)
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
                    forecast_text = f"**未来 {len(future_preds)} 步预测值({target_col}):{time_range}**\n\n" + "\n".join(rows)
                    if len(future_preds) > 30:
                        forecast_text += f"\n  ...(共 {len(future_preds)} 步,已截取前30步)"

                    first_val = float(future_preds[0])
                    last_val = float(future_preds[-1])
                    change = last_val - first_val
                    pct = (change / abs(first_val) * 100) if first_val != 0 else 0

                    if change > 0:
                        trend = "📈 **上升趋势**"
                        trend_desc = f"从 {first_val:.4f} 上涨到 {last_val:.4f},涨幅 **{abs(change):.4f}**({abs(pct):.1f}%)"
                    elif change < 0:
                        trend = "📉 **下降趋势**"
                        trend_desc = f"从 {first_val:.4f} 下降到 {last_val:.4f},跌幅 **{abs(change):.4f}**({abs(pct):.1f}%)"
                    else:
                        trend = "➡️ **基本平稳**"
                        trend_desc = f"基本维持在 {last_val:.4f} 附近"

                    summary_text = (
                        f"**{target_col} 未来 {n_steps} 步趋势总结**({time_range})\n\n"
                        f"**趋势方向**:{trend}\n\n"
                        f"{trend_desc}\n\n"
                        f"**预测区间**:{min(future_preds):.4f} ~ {max(future_preds):.4f}\n\n"
                        f"**预测均值**:{sum(future_preds)/len(future_preds):.4f}\n\n"
                        f"💡 *以上为模型自动预测结果,仅供参考。*"
                    )
            except Exception as e:
                forecast_text = f"趋势预测生成失败:{str(e)}"
                summary_text = "*趋势预测生成失败,请查看上方训练结果*"

        # ===== 生成结果包(zip) ===== 三层结构
        # 第一层:核心文件(必须保存,独立 try)
        zip_path = None
        ba_png_path = None
        ba_csv_path = None
        hist_png_path = None
        hist_csv_path = None
        th_to_plot = None

        if forecast_plot is not None and future_preds is not None and len(future_preds) > 0:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import zipfile
                import json

                # ── 核心文件 1: forecast.png ──
                forecast_png_path = str(output_dir / "forecast.png")
                if not Path(forecast_png_path).is_file():
                    fig = select_plot_function(
                        chart_requirement, hist, future_preds,
                        target_col, steps_hist, steps_fut, xtick_hist, xtick_fut, std_val, xlabel
                    )
                    fig.savefig(forecast_png_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    forecast_plot = forecast_png_path

                # ── 核心文件 2: forecast_data.csv ──
                all_steps = steps_hist + steps_fut
                all_vals = list(hist) + list(future_preds)
                types_ = ['历史'] * len(steps_hist) + ['预测'] * len(steps_fut)
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

                # ── 核心文件 3: metrics.json ──
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
                    'target': target_col_display if isinstance(target_col, list) else target_col,
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

            except Exception as e:
                print("核心文件保存失败: {}".format(e))
                # 核心文件失败,zip 仍然尝试打包(用已存在的文件)
                forecast_png_path = str(output_dir / "forecast.png")

            # ── 第二层:非核心文件(独立 try,失败不影响 ZIP) ──
            try:
                from matplotlib.figure import Figure

                # sklearn BA 图
                if predictor and predictor.is_fitted and predictor.task_type == "regression":
                    try:
                        split_idx = min(int(len(X_df) * 0.8), len(X_df) - 1)
                        X_test_ba = X_df.iloc[split_idx:].select_dtypes(include=[np.number])
                        y_test_ba = np.asarray(y_series.iloc[split_idx:]).flatten()
                        if len(y_test_ba) > 0:
                            X_scaled_ba = predictor.scaler.transform(X_test_ba.fillna(X_test_ba.median()))
                            y_pred_ba = predictor.model.predict(X_scaled_ba)
                            ba_fig = plot_bland_altman(y_test_ba, y_pred_ba, title=model_name)
                            ba_png_path = str(output_dir / "bland_altman.png")
                            ba_fig.savefig(ba_png_path, dpi=150, bbox_inches="tight")
                            plt.close(ba_fig)
                            ba_df = pd.DataFrame({"mean": list((y_pred_ba + y_test_ba) / 2), "diff": list(y_pred_ba - y_test_ba)})
                            ba_csv_path = str(output_dir / "bland_altman_data.csv")
                            ba_df.to_csv(ba_csv_path, index=False, encoding="utf-8-sig")
                    except Exception as e:
                        print("BA chart error: {}".format(e))

                # sklearn 训练历史
                if predictor and predictor.is_fitted and predictor.train_history:
                    th = predictor.train_history
                    n = len(th.get("loss", th.get("accuracy", [])))
                    if n > 0:
                        try:
                            hist_fig = Figure(figsize=(7, 4), dpi=150)
                            ax_h = hist_fig.add_subplot(111)
                            if "loss" in th:
                                ax_h.plot(th["epoch"], th["loss"], color="#E64B35", lw=2, marker="o", markersize=4)
                                ax_h.set_ylabel("Loss (RMSE)", fontsize=10)
                                ax_h.set_title("Training Loss per Epoch (" + model_name + ")", fontsize=11)
                            else:
                                ax_h.plot(th["epoch"], th["accuracy"], color="#4DBBD5", lw=2, marker="o", markersize=4)
                                ax_h.set_ylabel("Accuracy", fontsize=10)
                                ax_h.set_title("Training Accuracy per Epoch (" + model_name + ")", fontsize=11)
                            ax_h.set_xlabel("Epoch", fontsize=10)
                            ax_h.grid(True, alpha=0.3)
                            hist_fig.tight_layout()
                            hist_png_path = str(output_dir / "training_history.png")
                            hist_fig.savefig(hist_png_path, dpi=150, bbox_inches="tight")
                            plt.close(hist_fig)
                            pd.DataFrame.from_dict(th, orient='columns').to_csv(str(output_dir / "training_history.csv"), index=False, encoding="utf-8-sig")
                            hist_csv_path = str(output_dir / "training_history.csv")
                            th_to_plot = th
                        except Exception as e:
                            print("History chart error: {}".format(e))

                # LSTM / CNN1D / PatchTST 训练历史
                if lstm_pred and lstm_pred.is_fitted and hasattr(lstm_pred, 'train_history'):
                    th = lstm_pred.train_history
                    if th and th.get('epoch'):
                        try:
                            th_png = plot_training_history(th, target_col_display or target_col, output_dir)
                            if th_png and Path(th_png).is_file():
                                hist_png_path = th_png
                            pd.DataFrame.from_dict(th, orient='columns').to_csv(str(output_dir / "training_history.csv"), index=False, encoding="utf-8-sig")
                            hist_csv_path = str(output_dir / "training_history.csv")
                            th_to_plot = th
                        except Exception as e:
                            print("LSTM/CNN1D history error: {}".format(e))
            except Exception as e:
                print("非核心文件生成失败: {}".format(e))

            # ── 第三层:ZIP 创建(独立 try,失败不影响返回值) ──
            try:
                zip_name = f"ChronoML_{target_col_display or target_col}_{model_name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.zip"
                zip_path = str(output_dir / zip_name)
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # 核心文件
                    if Path(forecast_png_path).is_file():
                        zf.write(forecast_png_path, arcname="forecast.png")
                    if Path(output_dir / "forecast_data.csv").is_file():
                        zf.write(output_dir / "forecast_data.csv", arcname="forecast_data.csv")
                    if Path(output_dir / "metrics.json").is_file():
                        zf.write(output_dir / "metrics.json", arcname="metrics.json")
                    # 非核心文件(可选)
                    if ba_png_path and Path(ba_png_path).is_file():
                        zf.write(ba_png_path, arcname="bland_altman.png")
                    if ba_csv_path and Path(ba_csv_path).is_file():
                        zf.write(ba_csv_path, arcname="bland_altman_data.csv")
                    if hist_png_path and Path(hist_png_path).is_file():
                        zf.write(hist_png_path, arcname="training_history.png")
                    if hist_csv_path and Path(hist_csv_path).is_file():
                        zf.write(hist_csv_path, arcname="training_history.csv")
                prog(0.95, desc="打包完成")
            except Exception as e:
                print("ZIP 打包失败: {}".format(e))
                zip_path = None

        # 生成训练历史曲线图（用于 UI 显示）
        training_hist_plot = None
        th_for_plot = None
        if lstm_pred and lstm_pred.is_fitted and hasattr(lstm_pred, 'train_history'):
            th_for_plot = lstm_pred.train_history
        elif predictor and predictor.is_fitted and hasattr(predictor, 'train_history'):
            th_for_plot = predictor.train_history

        if th_for_plot and th_for_plot.get('epoch'):
            try:
                th_png_path = plot_training_history(
                    th_for_plot,
                    target_col_display or target_col or 'Target',
                    output_dir
                )
                if th_png_path and Path(th_png_path).is_file():
                    training_hist_plot = th_png_path
            except Exception as e:
                print("Training history plot error: {}".format(e))

        return msg, config, importance, forecast_plot, forecast_text, summary_text, zip_path, training_hist_plot

    def on_download(dummy, prog=gr.Progress()):
        """手动触发下载最新结果包"""
        import zipfile
        from pathlib import Path
        base_dir = Path(__file__).parent.resolve()
        outputs = base_dir / "outputs"
        if not outputs.exists():
            return ""
        zips = sorted(outputs.rglob("ChronoML_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if zips:
            return str(zips[0])
        return ""

    def on_predict(file):
        global predictor, lstm_pred

        # 修复:更严格的 guard,确保 predictor 和 lstm_pred 都是 None 或未训练时提前返回
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
                return "❌ 预测失败:模型返回空结果", None
            result_df = df.copy()
            result_df['预测值'] = pred
        elif lstm_ready:
            X = df.values.astype(np.float32)
            pred = lstm_pred.predict(X)
            if pred is None:
                print("LSTM 预测返回 None")
                return "❌ 预测失败:模型返回空结果", None
            result_df = df.copy()
            result_df['预测值'] = pred

        # 双重保险:result_df 仍未定义则返回错误
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
        outputs=[result_out, config_out, importance_out, forecast_plot, forecast_out, summary_out, download_file, training_history_plot]
    )
    download_btn.click(
        on_download,
        inputs=[download_file],
        outputs=[download_file]
    )
    predict_btn.click(on_predict, inputs=[predict_file], outputs=[predict_status, predict_out], queue=False)

    # ============================================================
    # 🧭 智能预测向导 Tab
    # ============================================================
    with gr.Tab("🧭 智能预测向导"):

        # --- 全局状态 ---
        wizard_state = gr.State({
            "step": 0,
            "file_path": None,
            "parse_config": None,
            "diagnostics": None,
            "df_parsed": None,
            "analysis": None,
            "answers": None,
            "recommendation": None,
            "final_config": None,
        })

        # --- Step 0: 文件上传 ---
        gr.Markdown("## 🧭 智能预测向导")
        gr.Markdown("*上传数据 → 系统分析 → 智能推荐 → 一键预测*")
        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                wiz_file_input = gr.File(
                    label="📂 上传 CSV 文件",
                    file_types=[".csv"],
                    height=80,

                )
                wiz_diagnostics_md = gr.Markdown("*上传后自动分析...*")

            with gr.Column(scale=1):
                # --- Step 1: CSV 解析确认 ---
                gr.Markdown("### 📋 Step 1: 确认数据解析方式")
                gr.Markdown("*系统已自动检测，可直接确认或手动调整*")

                wiz_has_header = gr.Radio(
                    label="① 文件有没有表头？",
                    choices=["系统自动检测（推荐）", "有表头（第一行是列名）", "无表头（手动指定列名）"],
                    value="系统自动检测（推荐）",
                    
                )
                wiz_date_col = gr.Radio(
                    label="② 哪列是日期/时间？",
                    choices=["系统自动检测（推荐）", "无日期列（纯数值索引）"],
                    value="系统自动检测（推荐）",
                    
                )
                wiz_date_format = gr.Radio(
                    label="③ 日期格式？",
                    choices=["系统自动检测（推荐）", "指定格式：%Y-%m-%d", "指定格式：%Y/%m/%d", "指定格式：%Y%m%d", "时间戳（数字）"],
                    value="系统自动检测（推荐）",
                    
                )
                wiz_separator = gr.Radio(
                    label="④ 分隔符是什么？",
                    choices=["系统自动检测（推荐）", "逗号", "分号", "Tab", "空格"],
                    value="系统自动检测（推荐）",
                    
                )
                wiz_missing_strategy = gr.Radio(
                    label="⑤ 缺失值如何处理？",
                    choices=["系统自动检测（推荐）", "删除含缺失的行", "用均值填充", "用前值填充（前向）", "不处理"],
                    value="系统自动检测（推荐）",
                    
                )
                wiz_confirm_parse_btn = gr.Button("✅ 确认并分析数据 →", variant="primary", size="lg")

        # --- Step 2: 数据概览 ---
        gr.Markdown("---")
        gr.Markdown("### 📊 Step 2: 数据概览与分析")
        wiz_data_overview = gr.Markdown("*确认解析后将显示数据概览...*")
        wiz_to_config_btn = gr.Button("✅ 下一步：配置预测 →", variant="primary", size="lg", interactive=False)

        # --- Step 3: 预测配置 ---
        gr.Markdown("---")
        gr.Markdown("### 🎯 Step 3: 配置预测需求")

        with gr.Row():
            with gr.Column(scale=1):
                wiz_target_col = gr.Dropdown(
                    label="① 目标列（预测什么）？",
                    choices=[],
                    allow_custom_value=True,
                    info="上传并解析数据后自动填充"
                )
                wiz_pred_len = gr.Radio(
                    label="② 预测多久以后？",
                    choices=["短期（7步）", "中期（30步）", "长期（90步）", "自定义"],
                    value="中期（30步）",

                )
                wiz_pred_len_custom = gr.Number(
                    label="自定义步数",
                    value=30,
                    visible=False,

                )
                wiz_external_factors = gr.CheckboxGroup(
                    label="③ 外部驱动因素（可多选）？",
                    choices=["湿度", "价格/成本", "节假日", "无外部因素（纯自变量）"],

                )
                wiz_priority = gr.Radio(
                    label="④ 优先准确率还是稳定性？",
                    choices=["准确率优先（允许波动）", "稳定性优先（减少极端误差）", "均衡模式"],
                    value="准确率优先（允许波动）"
                )
                wiz_need_explain = gr.Checkbox(
                    label="⑤ 显示模型解释（特征重要性）？",
                    value=False,

                )
                wiz_generate_btn = gr.Button("🧠 生成推荐方案 →", variant="primary", size="lg")

            with gr.Column(scale=1):
                wiz_recommendation_md = gr.Markdown("*推荐方案将显示在这里...*")
                wiz_model_slider_md = gr.Markdown("*可调参数...*")

                # 可调参数（根据推荐动态显示）
                wiz_seq_len = gr.Slider(label="seq_len（输入窗口）", minimum=12, maximum=512, value=48, step=4)
                wiz_pred_len_slider = gr.Slider(label="pred_len（预测步数）", minimum=3, maximum=512, value=24, step=1)
                wiz_epochs = gr.Slider(label="epochs（训练轮数）", minimum=5, maximum=500, value=30, step=5)
                wiz_lr = gr.Slider(label="learning_rate（学习率）", minimum=0.0001, maximum=0.01, value=0.001, step=0.0005)
                wiz_model_select = gr.Dropdown(
                    label="模型",
                    choices=["PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting"],
                    value="PatchTST"
                )

        # --- Step 4: 训练 + 结果 ---
        gr.Markdown("---")
        wiz_risk_warnings = gr.Markdown("*风险提示...*")
        with gr.Row():
            wiz_cancel_btn = gr.Button("← 上一步", variant="secondary")
            wiz_train_btn = gr.Button("🚀 开始预测", variant="primary", size="lg")

        gr.Markdown("---")
        gr.Markdown("### 📈 预测结果")
        wiz_result_out = gr.Textbox(label="训练结果", lines=6, interactive=False)
        wiz_forecast_plot = gr.Image(label="趋势预测图")
        wiz_forecast_text = gr.Textbox(label="预测值", lines=10, interactive=False)
        wiz_summary_out = gr.Markdown("*趋势总结...*")
        wiz_download_btn = gr.Button("📥 下载完整结果包", variant="secondary", size="lg")
        wiz_download_file = gr.File(label="点击下载 zip", interactive=False)

    # ============================================================
    # 向导事件函数
    # ============================================================

    def wiz_on_file_upload(file, state):
        """Step 0: 文件上传 → 自动检测"""
        if file is None:
            return {**state, "step": 0, "file_path": None, "diagnostics": None}, \
                "*请上传 CSV 文件*", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        # Gradio 6.x + gradio_client 场景: string path 直接传入，不走 FileData 解析
        if isinstance(file, str):
            file_path = file if Path(file).is_file() else None
        else:
            file_path = extract_file_path(file)
        if not file_path:
            return {**state, "step": 0}, "*❌ 文件路径无效*", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from src.utils.csv_parser import CSVParser

            parser = CSVParser()
            config, diagnostics = parser.auto_detect(file_path)

            # 更新 UI 控件
            header_choices = ["系统自动检测（推荐）", "有表头（第一行是列名）", "无表头（手动指定列名）"]
            header_default = "系统自动检测（推荐）"
            if diagnostics.get('has_header'):
                header_default = "有表头（第一行是列名）"
            else:
                header_default = "无表头（手动指定列名）"

            date_col_choices = ["系统自动检测（推荐）", "无日期列（纯数值索引）"]
            if diagnostics.get('detected_date_col'):
                date_col_choices.insert(0, f"第 [ {diagnostics['detected_date_col']} ] 列")

            sep_display = diagnostics.get('detected_separator', '逗号')
            sep_choices = ["系统自动检测（推荐）", "逗号", "分号", "Tab", "空格"]
            sep_map = {'逗号': '逗号', '分号': '分号', 'Tab': 'Tab', '空格': '空格'}
            if sep_display in sep_map.values():
                sep_choices = ["系统自动检测（推荐）"] + [sep_display]

            # 预览表格
            preview_rows = diagnostics.get('preview_rows', [])
            preview_df = pd.DataFrame(preview_rows[:50], columns=preview_rows[0] if preview_rows else None) if preview_rows else None

            diag_md = f"""**文件**: `{diagnostics['file_name']}`  
**大小**: {diagnostics['file_size_mb']} MB  
**预估行数**: ~{diagnostics.get('estimated_rows', '?')} 行 × {diagnostics.get('detected_cols', '?')} 列  
**分隔符**: {sep_display}  
**表头**: {diagnostics.get('detected_header', '?')}  
**日期列**: {diagnostics.get('detected_date_col', '未检测到')}  
**数值列**: {diagnostics.get('numeric_cols', [])}"""

            new_state = {
                **state,
                "step": 1,
                "file_path": file_path,
                "diagnostics": diagnostics,
                "parse_config": config.to_dict(),
            }
            _wizard_data['file_path'] = file_path
            _wizard_data['step'] = 1

            return new_state, diag_md, \
                gr.update(choices=header_choices, value=header_default), \
                gr.update(choices=date_col_choices), \
                gr.update(), \
                gr.update(choices=sep_choices), \
                gr.update(), \
                _preview_md(preview_df) if preview_df is not None else gr.update(visible=False)

        except Exception as e:
            import traceback; traceback.print_exc()
            return {**state, "step": 0}, f"*❌ 解析失败: {str(e)[:100]}*", \
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def wiz_on_confirm_parse(wiz_has_header, wiz_date_col, wiz_date_format,
                              wiz_separator, wiz_missing_strategy,
                              state):
        # gr.State 序列化可能丢失数据，优先从模块级变量读取
        file_path = (state.get("file_path") if state else None) or _wizard_data.get('file_path')
        """Step 1→2: 确认解析配置 → 解析 + 分析"""
        if not file_path:
            return "*❌ 请先上传文件*", gr.update(interactive=False), \
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from src.utils.csv_parser import CSVParser
            from src.utils.data_analyzer import DataAnalyzer

            # 构建 ParseConfig
            from src.utils.csv_parser import ParseConfig
            cfg = ParseConfig()
            cfg.mode = "manual"

            # 表头
            if wiz_has_header == "有表头（第一行是列名）":
                cfg.has_header = True
            elif wiz_has_header == "无表头（手动指定列名）":
                cfg.has_header = False
            else:
                cfg.has_header = state.get('diagnostics', {}).get('has_header', True)

            # 日期列
            if wiz_date_col and "第 [" in wiz_date_col:
                import re
                match = re.search(r'第 \[ (.+?) \]', wiz_date_col)
                if match:
                    cfg.date_col = match.group(1)
            elif wiz_date_col == "无日期列（纯数值索引）":
                cfg.date_col = None
            else:
                cfg.date_col = state.get('diagnostics', {}).get('detected_date_col')

            # 分隔符
            sep_map = {'逗号': ',', '分号': ';', 'Tab': '\t', '空格': ' '}
            if wiz_separator in sep_map:
                cfg.separator = sep_map[wiz_separator]
            else:
                cfg.separator = ','

            # 缺失值
            miss_map = {
                "删除含缺失的行": "drop",
                "用均值填充": "fill_mean",
                "用前值填充（前向）": "fill_forward",
                "不处理": "none",
            }
            if wiz_missing_strategy in miss_map:
                cfg.missing_strategy = miss_map[wiz_missing_strategy]
            else:
                cfg.missing_strategy = "auto"

            # 解析
            parser = CSVParser()
            df, parse_report = parser.parse(file_path, cfg)

            # 分析
            analyzer = DataAnalyzer()
            date_col = cfg.date_col if cfg.date_col and cfg.date_col != 'auto' else None
            analysis = analyzer.analyze(df, date_col=date_col)

            # 更新目标列下拉
            numeric_choices = list((analysis.column_stats or {}).keys())
            default_target = analysis.suggested_target or (numeric_choices[0] if numeric_choices else None)

            overview_md = f"""**✅ 解析成功**

| 项目 | 值 |
|------|-----|
| 数据量 | {analysis.n_samples:,} 行 × {analysis.n_features} 列 |
| 数值列 | {analysis.n_numeric_cols} 个 |
| 日期列 | {analysis.date_col or '无'} |
| 日期范围 | {analysis.date_range[0] if analysis.date_range else '?'} ~ {analysis.date_range[1] if analysis.date_range else '?'} |
| 时间粒度 | {analysis.time_step_unit or '未知'} |
| 季节性检测 | {analysis.seasonality_label or '未检测到'} |
| 缺失值 | {sum(analysis.missing_summary.values()) if analysis.missing_summary else 0} 个 |
| 推荐目标 | **{analysis.suggested_target}**（{analysis.suggested_target_reason or ''}）|
| 推荐 seq_len | {analysis.suggested_seq_len} |
| 推荐 pred_len | {analysis.suggested_pred_len} |

**⚠️ 警告**: {analysis.warnings[0] if analysis.warnings else '无'}
"""
            if not numeric_choices:
                overview_md += "\n\n*❌ 未检测到数值列，无法进行时序预测*"

            new_state = {
                **state,
                "step": 2,
                "parse_config": cfg.to_dict(),
            }
            # DataFrame 不走 gr.State 序列化，直接存模块级变量
            _wizard_data['df_parsed'] = df
            _wizard_data['analysis'] = analysis.to_dict()
            _wizard_data['step'] = 2

            return overview_md, gr.update(interactive=True), \
                gr.update(choices=numeric_choices, value=default_target), \
                gr.update(value="中期（30步）"), \
                gr.update(choices=["湿度", "价格/成本", "节假日", "无外部因素（纯自变量）"]), \
                gr.update(value=analysis.suggested_seq_len), \
                gr.update(value=analysis.suggested_pred_len), \
                gr.update(value="准确率优先（允许波动）")

        except Exception as e:
            import traceback; traceback.print_exc()
            return f"*❌ 解析/分析失败: {str(e)[:200]}*", gr.update(interactive=False), \
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def wiz_on_generate_recommendation(wiz_target_col, wiz_pred_len, wiz_pred_len_custom,
                                        wiz_external_factors, wiz_priority, wiz_need_explain,
                                        wiz_seq_len, state):
        """Step 2→3: 用户配置 → 生成推荐"""
        if not wiz_target_col:
            return "*❌ 请选择目标列*", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from src.utils.data_analyzer import DataAnalyzer, AnalysisResult
            from src.utils.recommendation_engine import RecommendationEngine

            # 重建 analysis 对象（从模块级变量读取，避免 gr.State 序列化丢失）
            analysis_dict = _wizard_data.get('analysis', {})
            analysis = AnalysisResult()
            for k, v in analysis_dict.items():
                setattr(analysis, k, v)

            # 解析预测步数
            if wiz_pred_len == "自定义":
                user_pred = int(wiz_pred_len_custom) if wiz_pred_len_custom else 30
            elif "7步" in wiz_pred_len:
                user_pred = 7
            elif "90步" in wiz_pred_len:
                user_pred = 90
            else:
                user_pred = 30

            # 解析外部因素
            ext_factors = [f for f in (wiz_external_factors or []) if f != "无外部因素（纯自变量）"]

            # 优先级
            if "稳定性" in wiz_priority:
                priority = "stability"
            elif "均衡" in wiz_priority:
                priority = "balanced"
            else:
                priority = "accuracy"

            user_answers = {
                'pred_len': user_pred,
                'external_factors': ext_factors,
                'priority': priority,
                'need_explain': wiz_need_explain,
            }

            # 推荐
            engine = RecommendationEngine()
            recommendation = engine.recommend(analysis, user_answers)

            # 推荐卡片
            risk_md = ""
            if recommendation.risk_warnings:
                risk_md = "**⚠️ 风险提示:**\n" + "\n".join(f"  • {w}" for w in recommendation.risk_warnings)
            else:
                risk_md = "**✅ 未检测到明显风险**"

            reason_md = f"""**🧠 智能推荐结果**

**推荐模型**: `{recommendation.model}`  
**推荐理由**: {recommendation.reason}  
**置信度**: {recommendation.confidence:.0%}

**推荐参数**:
| 参数 | 值 |
|------|-----|
| seq_len | {recommendation.seq_len} |
| pred_len | {recommendation.pred_len} |
| epochs | {recommendation.epochs} |
| learning_rate | {recommendation.learning_rate} |
| dropout | {recommendation.dropout} |

{risk_md}
"""

            new_state = {
                **state,
                "step": 3,
                "answers": user_answers,
                "recommendation": recommendation.to_dict(),
                "final_config": {
                    'target_col': wiz_target_col,
                    'model': recommendation.model,
                    'seq_len': recommendation.seq_len,
                    'pred_len': recommendation.pred_len,
                    'epochs': recommendation.epochs,
                    'learning_rate': recommendation.learning_rate,
                    'dropout': recommendation.dropout,
                    'hidden_size': recommendation.hidden_size,
                    'num_layers': recommendation.num_layers,
                    'd_model': recommendation.d_model,
                    'n_heads': recommendation.n_heads,
                    'n_layers_trans': recommendation.n_layers_trans,
                    'd_ff': recommendation.d_ff,
                    'patch_size': recommendation.patch_size,
                    'hidden_channels': recommendation.hidden_channels,
                    'batch_size': recommendation.batch_size,
                    'predict_mode': recommendation.predict_mode,
                }
            }

            max_seq = max(12, (analysis.n_samples or 36) // 2)
            # 不在这里设 maximum，只更新 value，组件自身 maximum 足够大（512）
            return reason_md, \
                gr.update(value=recommendation.seq_len, minimum=12), \
                gr.update(value=recommendation.pred_len, minimum=3), \
                gr.update(value=recommendation.epochs), \
                gr.update(value=recommendation.learning_rate), \
                gr.update(choices=[recommendation.model] + [m for m in ["PatchTST", "LSTM", "EnhancedCNN1D", "GradientBoosting"] if m != recommendation.model], value=recommendation.model), \
                new_state

        except Exception as e:
            import traceback; traceback.print_exc()
            return f"*❌ 推荐生成失败: {str(e)[:200]}*", \
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), state

    def wiz_on_train(wiz_model, wiz_seq_len, wiz_pred_len, wiz_epochs, wiz_lr,
                     wiz_target_col, wiz_external_factors, state):
        """Step 3→4: 开始训练"""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from src.utils.feature_engine import FeatureEngine, FeatureConfig
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import zipfile
            import json

            df = _wizard_data.get('df_parsed')
            if df is None:
                return "*❌ 数据无效，请重新上传*", None, None, None, gr.update()

            target_col = wiz_target_col
            # 验证目标列不能为空
            if not target_col or not isinstance(target_col, str) or target_col.strip() == "":
                return "*❌ 请选择目标列（预测什么）*", None, None, None, gr.update()
            seq_len = int(wiz_seq_len)
            pred_len = int(wiz_pred_len)
            epochs = int(wiz_epochs)
            lr = float(wiz_lr)
            model_name = wiz_model

            # 准备数据
            analysis_dict = _wizard_data.get('analysis', {}) or {}
            seasonality = analysis_dict.get('detected_seasonality')
            time_unit = analysis_dict.get('time_step_unit')
            date_col = analysis_dict.get('date_col')

            # 自动特征工程
            fe = FeatureEngine()
            feat_cfg = fe.suggest_config(seasonality, time_unit)
            feat_cfg.target_col = target_col
            df_enhanced, new_features = fe.build(df, target_col, date_col, seasonality, feat_cfg)

            # 准备 X, y
            numeric_cols = list(df_enhanced.select_dtypes(include=[np.number]).columns)
            y = df_enhanced[target_col].values.astype(np.float32)
            feature_cols = [c for c in numeric_cols if c != target_col]

            # lag/rolling 特征在前 N 行有 NaN，移除这些行；剩余 NaN 用 0 填充
            n_drop = max(7, min(30, len(df_enhanced) // 10))
            if feature_cols:
                X_df_feat = df_enhanced[feature_cols].iloc[n_drop:].fillna(0)
            else:
                X_df_feat = pd.DataFrame({'__target__': y[n_drop:]})
            X = X_df_feat.values.astype(np.float32)
            y = y[n_drop:]

            # 获取最终配置中的其他参数
            rec = (_wizard_data.get('recommendation') or
                   (state.get('recommendation', {}) if state else {}))
            hidden_size = rec.get('hidden_size', 64)
            num_layers = rec.get('num_layers', 2)
            dropout = rec.get('dropout', 0.2)
            batch_size = rec.get('batch_size', 32)
            d_model = rec.get('d_model', 128)
            n_heads = rec.get('n_heads', 4)
            n_layers_trans = rec.get('n_layers_trans', 3)
            d_ff = rec.get('d_ff', 256)
            patch_size = rec.get('patch_size', 16)
            hidden_channels = rec.get('hidden_channels', 64)

            msg_out = ""
            lstm_pred = None
            predictor = None
            future_preds = None
            th = None
            metrics = {}

            # 训练
            if model_name == "LSTM":
                from src.models.lstm_model import LSTMPredictor
                lstm_pred = LSTMPredictor()
                ok, msg = lstm_pred.train(X, y, seq_len=min(seq_len, max(5, len(X)//5)),
                                          hidden_size=hidden_size, num_layers=num_layers,
                                          epochs=min(epochs, 50), batch_size=batch_size,
                                          learning_rate=lr,
                                          target_col=target_col)
                if ok:
                    future_preds = lstm_pred.predict_future(X, steps=min(pred_len, 30))
                    th = lstm_pred.train_history
                    metrics = lstm_pred.metrics

            elif model_name == "PatchTST":
                from src.models.patchtst_model import PatchTSTPredictor
                lstm_pred = PatchTSTPredictor()
                # 安全参数：确保 seq_len + pred_len <= len(X) * 0.7
                max_total = max(5, int(len(X) * 0.7))
                safe_seq = min(seq_len, max(4, max_total // 2))
                safe_pred = min(pred_len, max(1, max_total - safe_seq))
                ok, msg = lstm_pred.train(X, y, seq_len=safe_seq, pred_len=safe_pred,
                                          d_model=d_model, n_heads=n_heads, n_layers=n_layers_trans,
                                          d_ff=d_ff, patch_size=max(4, patch_size),
                                          epochs=min(epochs, 30), batch_size=batch_size,
                                          learning_rate=lr,
                                          target_col=target_col)
                if ok:
                    future_preds = lstm_pred.predict_future(X, steps=min(pred_len, 30))
                    th = lstm_pred.train_history
                    metrics = lstm_pred.metrics

            elif model_name == "EnhancedCNN1D":
                from src.models.cnn1d_complex import EnhancedCNN1DPredictor
                lstm_pred = EnhancedCNN1DPredictor()
                auto_seq = min(seq_len, max(4, len(X)//4))
                auto_pred = min(pred_len, max(1, auto_seq//2))
                ok, msg = lstm_pred.train(X, y, seq_len=auto_seq, pred_len=auto_pred,
                                          hidden_channels=hidden_channels,
                                          epochs=min(epochs, 50), batch_size=batch_size,
                                          learning_rate=lr, dropout=dropout)
                if ok:
                    future_preds = lstm_pred.predict_future(X, steps=min(pred_len, 30))
                    th = lstm_pred.train_history
                    metrics = lstm_pred.metrics

            elif model_name == "GradientBoosting":
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                m = GradientBoostingRegressor(n_estimators=min(50, epochs), max_depth=3, learning_rate=lr, random_state=42)
                m.fit(X_tr, y_tr)
                from sklearn.metrics import mean_squared_error, r2_score
                y_pred = m.predict(X_te)
                metrics = {
                    'RMSE': float(np.sqrt(mean_squared_error(y_te, y_pred))),
                    'R2': float(r2_score(y_te, y_pred)),
                    'MAE': float(np.mean(np.abs(y_te - y_pred))),
                    'model': 'GradientBoosting'
                }
                future_preds = m.predict(X_scaled[-1].reshape(1, -1))
                future_preds = np.array([float(future_preds[0])] * min(pred_len, 30))
                msg = f"**GradientBoosting** 训练完成：R²={metrics['R2']:.4f}"

                # 构造 training history（模拟 staged_predict）
                n_est = min(50, epochs)
                th = {'epoch': list(range(1, n_est + 1)),
                      'train_loss': [], 'val_loss': [],
                      'train_mae': [], 'val_mae': [],
                      'train_r2': [], 'val_r2': []}
                for i in range(1, n_est + 1):
                    m_i = GradientBoostingRegressor(n_estimators=i, max_depth=3, learning_rate=lr, random_state=42)
                    m_i.fit(X_tr, y_tr)
                    tr_pred = m_i.predict(X_tr)
                    te_pred = m_i.predict(X_te)
                    th['train_loss'].append(float(np.sqrt(mean_squared_error(y_tr, tr_pred))))
                    th['val_loss'].append(float(np.sqrt(mean_squared_error(y_te, te_pred))))
                    th['train_mae'].append(float(np.mean(np.abs(y_tr - tr_pred))))
                    th['val_mae'].append(float(np.mean(np.abs(y_te - te_pred))))
                    ss_res = np.sum((y_tr - tr_pred) ** 2)
                    ss_tot = np.sum((y_tr - np.mean(y_tr)) ** 2)
                    th['train_r2'].append(float(1 - ss_res / (ss_tot + 1e-8)))
                    ss_res_v = np.sum((y_te - te_pred) ** 2)
                    ss_tot_v = np.sum((y_te - np.mean(y_te)) ** 2)
                    th['val_r2'].append(float(1 - ss_res_v / (ss_tot_v + 1e-8)))

            if msg and future_preds is not None:
                msg_out = f"{msg}\n\n**指标**: RMSE={metrics.get('RMSE', 'N/A'):.4f}, R²={metrics.get('R2', 'N/A'):.4f}, MAE={metrics.get('MAE', 'N/A'):.4f}"
            else:
                msg_out = msg or "训练失败"

            # 绘图
            plot_path = None
            hist_vals = None
            steps_h = None
            steps_f = None
            if future_preds is not None and len(future_preds) > 0:
                last_n = min(100, len(y))
                hist_vals = y[-last_n:]
                steps_h = list(range(last_n))
                steps_f = list(range(last_n, last_n + len(future_preds)))

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(steps_h, hist_vals, color='#3B82F6', lw=2, label='历史')
                ax.plot(steps_f, future_preds, color='#FF6B2B', lw=2, ls='--', label='预测')
                ax.axvline(last_n - 0.5, color='gray', ls=':', lw=1.5)
                ax.set_xlabel('时间步')
                ax.set_ylabel(target_col)
                ax.set_title(f'{target_col} - {model_name}')
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()

                output_dir = Path(__file__).parent / "outputs" / "wizard"
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_path = str(output_dir / f"wizard_{target_col}_{model_name}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)  # 关闭 figure 释放内存

            # 预测值文本
            fc_lines = [f"**未来 {len(future_preds) if future_preds is not None else 0} 步预测值 ({target_col}):**"]
            if future_preds is not None:
                for i, v in enumerate(future_preds[:20]):
                    fc_lines.append(f"  第 {i+1} 步: **{float(v):.4f}**")
                if len(future_preds) > 20:
                    fc_lines.append(f"  ... (共 {len(future_preds)} 步)")

            fc_text = "\n".join(fc_lines)

            # 打包 ZIP（提前初始化变量）
            zip_path = None
            download_zip_path = None
            if plot_path and future_preds is not None:
                try:
                    output_dir = Path(__file__).parent / "outputs" / "wizard"
                    zip_name = f"ChronoML_{target_col}_{model_name}_wizard.zip"
                    zip_path = str(output_dir / zip_name)

                    # 预测 CSV
                    fc_csv = str(output_dir / "forecast_data.csv")
                    if hist_vals is not None and steps_h is not None and steps_f is not None:
                        pd.DataFrame({
                            'step': steps_h + steps_f,
                            'type': ['历史']*len(steps_h) + ['预测']*len(steps_f),
                            target_col: list(hist_vals) + [float(v) for v in future_preds]
                        }).to_csv(fc_csv, index=False, encoding='utf-8-sig')
                    else:
                        # 如果没有历史数据，只保存预测值
                        pd.DataFrame({
                            'step': list(range(len(future_preds))),
                            'type': ['预测']*len(future_preds),
                            target_col: [float(v) for v in future_preds]
                        }).to_csv(fc_csv, index=False, encoding='utf-8-sig')

                    # metrics JSON
                    mt_json = str(output_dir / "metrics.json")
                    with open(mt_json, 'w', encoding='utf-8') as f:
                        json.dump({'model': model_name, 'target': target_col, 'metrics': metrics,
                                   'seq_len': seq_len, 'pred_len': pred_len,
                                   'new_features': new_features[:10]}, f, indent=2)

                    # history CSV
                    hist_csv = str(output_dir / "training_history.csv")
                    if th and th.get('epoch'):
                        pd.DataFrame.from_dict(th, orient='columns').to_csv(hist_csv, index=False, encoding='utf-8-sig')

                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(plot_path, "forecast.png")
                        zf.write(fc_csv, "forecast_data.csv")
                        zf.write(mt_json, "metrics.json")
                        hist_p = str(output_dir / "training_history.csv")
                        if Path(hist_p).is_file():
                            zf.write(hist_p, "training_history.csv")

                    logger.info(f"Wizard ZIP 已打包: {zip_path}")
                    
                    # 复制到用户下载文件夹
                    import shutil
                    downloads_dir = Path.home() / "Downloads"
                    download_zip_path = str(downloads_dir / zip_name)
                    shutil.copy2(zip_path, download_zip_path)
                    logger.info(f"ZIP 已复制到下载文件夹: {download_zip_path}")
                    
                except Exception as e:
                    logger.warning(f"ZIP 打包失败: {e}")

            # 趋势总结（追加下载路径信息）
            if future_preds is not None and len(future_preds) > 1:
                first_v = float(future_preds[0])
                last_v = float(future_preds[-1])
                change = last_v - first_v
                pct = change / abs(first_v) * 100 if first_v != 0 else 0
                if change > 0:
                    trend = "📈 **上升趋势**"
                elif change < 0:
                    trend = "📉 **下降趋势**"
                else:
                    trend = "➡️ **基本平稳**"
                summary = f"{trend}，从 {first_v:.4f} 到 {last_v:.4f}（{'+' if pct >= 0 else ''}{pct:.1f}%）"
                
                # 追加下载路径信息
                if download_zip_path:
                    summary += f"\n\n✅ **结果包已保存到下载文件夹：**\n`{download_zip_path}`"
            else:
                summary = "*趋势分析完成*"
                if download_zip_path:
                    summary += f"\n\n✅ **结果包已保存到：** `{download_zip_path}`"

            return msg_out, \
                plot_path if plot_path else None, \
                fc_text, \
                summary, \
                gr.update(value=zip_path) if zip_path else gr.update(value=None)

        except Exception as e:
            import traceback; traceback.print_exc()
            return f"*❌ 训练异常: {str(e)[:200]}*", None, None, None, gr.update(value=None)


    # ============================================================
    # 绑定向导事件
    # ============================================================

    def wiz_show_custom_pred_len(wiz_pred_len):
        return gr.update(visible=("自定义" in wiz_pred_len))

    wiz_pred_len.change(wiz_show_custom_pred_len, inputs=[wiz_pred_len], outputs=[wiz_pred_len_custom])

    wiz_file_input.change(
        wiz_on_file_upload,
        inputs=[wiz_file_input, wizard_state],
        outputs=[wizard_state, wiz_diagnostics_md, wiz_has_header, wiz_date_col,
                  wiz_date_format, wiz_separator, wiz_missing_strategy, wiz_data_overview],
        queue=False
    )

    wiz_confirm_parse_btn.click(
        wiz_on_confirm_parse,
        inputs=[wiz_has_header, wiz_date_col, wiz_date_format, wiz_separator,
                 wiz_missing_strategy, wizard_state],
        outputs=[wiz_data_overview, wiz_to_config_btn, wiz_target_col,
                  wiz_pred_len, wiz_external_factors, wiz_seq_len,
                  wiz_pred_len_slider, wiz_priority],
        queue=False
    )

    wiz_to_config_btn.click(
        lambda s: (gr.update(interactive=True) if s.get('step', 0) >= 2 else gr.update()),
        inputs=[wizard_state],
        outputs=[wiz_generate_btn],
        queue=False
    )

    wiz_generate_btn.click(
        wiz_on_generate_recommendation,
        inputs=[wiz_target_col, wiz_pred_len, wiz_pred_len_custom,
                 wiz_external_factors, wiz_priority, wiz_need_explain,
                 wiz_seq_len, wizard_state],
        outputs=[wiz_recommendation_md, wiz_seq_len, wiz_pred_len_slider,
                  wiz_epochs, wiz_lr, wiz_model_select, wizard_state],
        queue=False
    )

    wiz_train_btn.click(
        wiz_on_train,
        inputs=[wiz_model_select, wiz_seq_len, wiz_pred_len_slider, wiz_epochs, wiz_lr,
                 wiz_target_col, wiz_external_factors, wizard_state],
        outputs=[wiz_result_out, wiz_forecast_plot, wiz_forecast_text,
                  wiz_summary_out, wiz_download_file],
        queue=False
    )

    wiz_download_btn.click(
        lambda zip_file: gr.update(value=zip_file) if zip_file else gr.update(value=None),
        inputs=[wiz_download_file],
        outputs=[wiz_download_file],
        queue=False
    )


if __name__ == "__main__":
    # Railway / HuggingFace Spaces 用 PORT 环境变量
    import os
    port = int(os.environ.get("PORT", 7861))
    print("=" * 60)
    print("  ChronoML Web 版 v1.5 已启动!")
    print(f"  访问地址: http://127.0.0.1:{port}")
    print("  同一局域网内的手机/电脑都可以访问")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        show_error=True
    )
