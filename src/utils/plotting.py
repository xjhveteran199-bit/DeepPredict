"""
PublicationPlotter - 论文级可视化工具
支持 IEEE / Nature / Science / Cell 等主流期刊样式
一键导出高 DPI 图片 + LaTeX 表格

使用示例:
    from src.utils.plotting import PublicationPlotter, JournalStyle

    plotter = PublicationPlotter(style=JournalStyle.IEEE)
    plotter.plot_prediction(y_true, y_pred, y_pred_lower, y_pred_upper,
                            title="CNN1D Prediction on Sensor Data",
                            save_path="fig1_prediction.png")
    plotter.plot_metrics_table(metrics_dict, save_path="table1_metrics.tex")
    plotter.plot_residuals(y_true, y_pred, save_path="fig2_residuals.png")
    plotter.export_all(export_dir="./paper_figures", dpi=300)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# 常用地质/工程期刊配色（色盲友好）
_COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'magenta': '#EE3377',
    'grey': '#BBBBBB',
    'dark': '#333333',
    'light_blue': '#66BBEE',
    'light_orange': '#EE8866',
}

_FIGURE_WIDTH_IEEE = 3.5    # 单栏 inch
_FIGURE_WIDTH_2COL = 7.0   # 双栏 inch
_FONTSIZE_AXES = 9
_FONTSIZE_TITLE = 10
_FONTSIZE_LEGEND = 8
_FONTSIZE_TICK = 8


class JournalStyle:
    """支持的期刊样式预设"""
    IEEE = "ieee"
    NATURE = "nature"
    SCIENCE = "science"
    CELL = "cell"
    BLACK_WHITE = "bw"  # 纯黑白（打印友好）


def _apply_style(style: str, fig: plt.Figure, ax: plt.Axes):
    """应用期刊样式"""
    if style == JournalStyle.BLACK_WHITE:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color('black')
        return

    # 通用 Science/Nature 样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.tick_params(labelsize=_FONTSIZE_TICK, width=0.8)
    ax.xaxis.label.set_fontsize(_FONTSIZE_AXES)
    ax.yaxis.label.set_fontsize(_FONTSIZE_AXES)
    ax.title.set_fontsize(_FONTSIZE_TITLE)
    fig.patch.set_facecolor('white')


def _get_rcParams(style: str):
    """配置 matplotlib 全局样式"""
    if style == JournalStyle.IEEE:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times'],
            'font.size': 9,
            'axes.labelsize': 9,
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
        })
    elif style in (JournalStyle.NATURE, JournalStyle.SCIENCE, JournalStyle.CELL):
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 9,
            'axes.labelsize': 9,
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'grid.linewidth': 0.5,
        })
    else:  # 默认
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 9,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'figure.dpi': 150,
            'savefig.dpi': 300,
        })


class PublicationPlotter:
    """
    论文级可视化工具

    支持的图表类型:
    - plot_prediction()      预测时序图（含置信区间）
    - plot_metrics_table()   指标汇总表（PNG + LaTeX）
    - plot_residuals()       残差分析（4 子图）
    - plot_scatter()         实际 vs 预测散点图
    - plot_loss_curve()      训练损失曲线
    - plot_multi_model()     多模型对比
    - plot_attention()       Attention 热力图（Transformer）
    - plot_frequency()       频域分析（FFT 频谱）
    - plot_cross_validation() 时序交叉验证滚动预测
    - export_all()           一键导出所有图表
    """

    def __init__(
        self,
        style: str = JournalStyle.IEEE,
        figure_width: float = _FIGURE_WIDTH_IEEE,
        style_name: str = "DeepPredict"
    ):
        self.style = style
        self.figure_width = figure_width
        self.style_name = style_name
        self.figure_height = figure_width * 0.618  # 黄金比例
        self._saved_figures: Dict[str, plt.Figure] = {}
        _get_rcParams(style)

    # ───────────────────────────────────────────────────────────────
    # 核心图表
    # ───────────────────────────────────────────────────────────────

    def plot_prediction(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_lower: Optional[np.ndarray] = None,
        y_pred_upper: Optional[np.ndarray] = None,
        time_index: Optional[np.ndarray] = None,
        title: str = "Prediction vs Actual",
        xlabel: str = "Time Step",
        ylabel: str = "Value",
        labels: Optional[Dict[str, str]] = None,
        highlight_region: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show_metrics: bool = True,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        预测时序图 - 论文中最常用的主图

        Args:
            y_true: 真实值
            y_pred: 预测值
            y_pred_lower / y_pred_upper: 95% 置信区间（可选）
            time_index: 横轴标签（可选，默认 0,1,2,...）
            title: 图标题
            labels: dict，键 'true'/'pred'/'ci' 的显示名
            highlight_region: (start, end) 高亮预测区域
            save_path: 保存路径，支持 .png / .pdf / .svg
            show_metrics: 是否在图中显示 R²/RMSE/MAE 文本
        """
        if labels is None:
            labels = {'true': 'Actual', 'pred': 'Predicted', 'ci': '95% CI'}

        n = len(y_true)
        if time_index is None:
            time_index = np.arange(n)
        if figsize is None:
            figsize = (self.figure_width, self.figure_width * 0.65)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        # 置信区间填充
        if y_pred_lower is not None and y_pred_upper is not None:
            ax.fill_between(
                time_index, y_pred_lower, y_pred_upper,
                alpha=0.2, color=_COLORS['light_blue'],
                label=labels.get('ci', '95% CI')
            )

        # 真实值线
        ax.plot(
            time_index, y_true,
            color=_COLORS['blue'], linewidth=1.2,
            label=labels.get('true', 'Actual'), alpha=0.9
        )

        # 预测值线
        ax.plot(
            time_index, y_pred,
            color=_COLORS['orange'], linewidth=1.2, linestyle='--',
            label=labels.get('pred', 'Predicted'), alpha=0.9
        )

        # 高亮预测区域
        if highlight_region is not None:
            start, end = highlight_region
            ax.axvspan(start, end, alpha=0.08, color='green')
            ax.text(start + (end - start) / 2, ax.get_ylim()[1] * 0.98,
                    'Forecast Zone', ha='center', fontsize=7,
                    color='green', style='italic')

        # 内嵌指标文本
        if show_metrics:
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)
            mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

            textstr = (f'$R^2$ = {r2:.4f}\n'
                       f'RMSE = {rmse:.4f}\n'
                       f'MAE = {mae:.4f}\n'
                       f'MAPE = {mape:.2f}%')
            props = dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='gray', alpha=0.85, linewidth=0.8)
            ax.text(0.97, 0.03, textstr, transform=ax.transAxes,
                    fontsize=7, verticalalignment='bottom',
                    horizontalalignment='right', bbox=props,
                    family='monospace')

        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.legend(loc='upper left', framealpha=0.9,
                  edgecolor='gray', fontsize=_FONTSIZE_LEGEND)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.grid(True, alpha=0.25)

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time_index: Optional[np.ndarray] = None,
        title: str = "Residual Analysis",
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        残差分析 4 子图 - 论文标准配置:
        (1) 残差 vs 时间
        (2) 残差直方图
        (3) Q-Q 图
        (4) 实际 vs 预测散点
        """
        residuals = y_true - y_pred

        if figsize is None:
            figsize = (self.figure_width * 1.4, self.figure_width * 1.0)

        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        fig.suptitle(title, fontweight='bold', y=1.01, fontsize=10)

        # (1) 残差时序
        ax1 = axes[0, 0]
        if time_index is None:
            time_index = np.arange(len(residuals))
        ax1.plot(time_index, residuals, color=_COLORS['blue'], linewidth=0.8, alpha=0.8)
        ax1.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax1.fill_between(time_index, residuals, 0, alpha=0.15, color=_COLORS['blue'])
        ax1.set_xlabel('Time Step', fontsize=8)
        ax1.set_ylabel('Residual', fontsize=8)
        ax1.set_title('(a) Residuals over Time', fontsize=8)
        ax1.grid(True, alpha=0.25)

        # (2) 残差直方图
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=30, color=_COLORS['teal'], alpha=0.7,
                 edgecolor='white', linewidth=0.5, density=True)
        # 正态拟合曲线
        mu, sigma = np.mean(residuals), np.std(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x_range, (1 / (sigma * np.sqrt(2 * np.pi))) *
                 np.exp(-0.5 * ((x_range - mu) / sigma) ** 2),
                 color='red', linewidth=1.5, label='Normal Fit')
        ax2.set_xlabel('Residual', fontsize=8)
        ax2.set_ylabel('Density', fontsize=8)
        ax2.set_title(f'(b) Histogram (μ={mu:.3f}, σ={sigma:.3f})', fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.25)

        # (3) Q-Q 图
        ax3 = axes[1, 0]
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.get_lines()[0].set_markerfacecolor(_COLORS['blue'])
        ax3.get_lines()[0].set_markersize(3)
        ax3.get_lines()[0].set_alpha(0.6)
        ax3.get_lines()[1].set_color('red')
        ax3.get_lines()[1].set_linewidth(1.2)
        ax3.set_title('(c) Q-Q Plot', fontsize=8)
        ax3.grid(True, alpha=0.25)

        # (4) 实际 vs 预测
        ax4 = axes[1, 1]
        ax4.scatter(y_true, y_pred, alpha=0.4, s=10, color=_COLORS['blue'])
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val],
                 'r--', linewidth=1.5, label='Perfect Fit')
        ax4.set_xlabel('Actual', fontsize=8)
        ax4.set_ylabel('Predicted', fontsize=8)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)
        ax4.set_title(f'(d) Actual vs Predicted ($R^2$={r2:.4f})', fontsize=8)
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.25)
        ax4.set_aspect('equal', adjustable='box')

        _apply_style(self.style, fig, ax1)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_metrics_table(
        self,
        metrics: Dict[str, Dict[str, float]],
        model_names: Optional[List[str]] = None,
        highlight_best: bool = True,
        save_path: Optional[str] = None,
        title: str = "Model Performance Comparison",
        caption: Optional[str] = None,
        export_latex: bool = True,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        指标汇总表 - 可直接插入论文

        Args:
            metrics: {'ModelName': {'R2': 0.85, 'RMSE': 0.12, ...}}
            model_names: 指定顺序
            highlight_best: 高亮每列最优值
            save_path: .png 保存路径
            export_latex: 同时导出 .tex 文件
        """
        # 提取所有指标名
        all_metric_names = set()
        for m in metrics.values():
            all_metric_names.update(m.keys())
        metric_names = sorted(all_metric_names)

        # 优先级排序：R2 放最前
        priority = ['R2', 'R²', 'RMSE', 'MAE', 'MAPE', 'MSE', 'Accuracy', 'Precision', 'Recall', 'F1']
        metric_names = sorted(metric_names, key=lambda x: (
            priority.index(x) if x in priority else len(priority), x
        ))

        if model_names is None:
            model_names = list(metrics.keys())

        n_cols = len(metric_names) + 1
        n_rows = len(model_names)

        cell_text = []
        for model in model_names:
            row = [model]
            for metric in metric_names:
                val = metrics[model].get(metric, None)
                if val is None:
                    row.append('—')
                elif isinstance(val, float):
                    row.append(f'{val:.4f}')
                else:
                    row.append(str(val))
            cell_text.append(row)

        # 找出每列最优值
        best_indices = {}
        for j, metric in enumerate(metric_names):
            metric_vals = []
            for i, model in enumerate(model_names):
                v = metrics[model].get(metric)
                metric_vals.append(v if isinstance(v, (int, float)) else None)
            numeric_vals = [(i, v) for i, v in enumerate(metric_vals) if v is not None]
            if numeric_vals:
                # R2/MAPE 类指标最大化，其他最小化
                if metric in ('R2', 'R²', 'Accuracy', 'Precision', 'Recall', 'F1'):
                    best_idx = max(numeric_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(numeric_vals, key=lambda x: x[1])[0]
                best_indices[j + 1] = best_idx

        # 绘制表格
        fig, ax = plt.subplots(figsize=(self.figure_width * 1.2, 0.25 * n_rows + 0.5))
        ax.axis('off')
        ax.axis('tight')

        table = ax.table(
            cellText=cell_text,
            colLabels=['Model'] + metric_names,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        # 表头样式
        for j in range(n_cols):
            cell = table[0, j]
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')

        # 高亮最优值
        if highlight_best:
            for j, i_best in best_indices.items():
                cell = table[i_best + 1, j]
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(fontweight='bold', color='#1B5E20')

        # 斑马纹
        for i in range(1, n_rows + 1):
            for j in range(n_cols):
                if i % 2 == 0 and (j not in best_indices or best_indices[j] != i - 1):
                    table[i, j].set_facecolor('#F5F5F5')

        ax.set_title(title, fontweight='bold', pad=12, fontsize=10)

        if caption:
            fig.text(0.5, -0.02, caption, ha='center', fontsize=7,
                    style='italic', wrap=True)

        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        # LaTeX 导出
        if export_latex and save_path:
            tex_path = Path(save_path).with_suffix('.tex')
            self._export_latex_table(model_names, metric_names, cell_text,
                                    best_indices, tex_path, caption or title)

        return fig

    def _export_latex_table(
        self, model_names, metric_names, cell_text,
        best_indices, tex_path: Path, caption: str
    ):
        """导出标准 LaTeX 三线表"""
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{tab:{tex_path.stem}}}',
            r'\begin{tabular}{l' + 'c' * len(metric_names) + '}',
            r'\toprule',
            ' & '.join(['\\textbf{Model}'] +
                       [f'\\textbf{{{m}}}' for m in metric_names]) + r' \\',
            r'\midrule',
        ]
        for i, row in enumerate(cell_text):
            formatted = []
            for j, cell in enumerate(row):
                if j == 0:
                    formatted.append(cell)
                elif (j) in best_indices and best_indices[j] == i:
                    formatted.append(f'\\textbf{{{cell}}}')
                else:
                    formatted.append(cell)
            lines.append(' & '.join(formatted) + r' \\')
        lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text('\n'.join(lines), encoding='utf-8')

    def plot_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted",
        save_path: Optional[str] = None,
        color_by: Optional[np.ndarray] = None,
        color_map: str = 'viridis',
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """实际 vs 预测散点图，含 R² 标注"""
        if figsize is None:
            figsize = (self.figure_width, self.figure_width)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        sc = ax.scatter(
            y_true, y_pred,
            c=color_by, cmap=color_map,
            alpha=0.5, s=15, edgecolors='none'
        )

        # 完美拟合线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=1.5, label='Perfect Fit', zorder=5)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)

        textstr = (f'$R^2$ = {r2:.4f}\n'
                   f'RMSE = {rmse:.4f}\n'
                   f'MAE = {mae:.4f}')
        props = dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='gray', alpha=0.9)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=props, family='monospace')

        ax.set_xlabel('Actual Value', labelpad=3)
        ax.set_ylabel('Predicted Value', labelpad=3)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='box')

        if color_by is not None:
            cbar = plt.colorbar(sc, ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=7)

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_loss_curve(
        self,
        train_losses: List[float],
        val_losses: List[float],
        best_epoch: Optional[int] = None,
        title: str = "Training and Validation Loss",
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        log_scale: bool = False,
        dpi: int = 300,
    ) -> plt.Figure:
        """改进的训练损失曲线"""
        if figsize is None:
            figsize = (self.figure_width, self.figure_width * 0.55)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        epochs = np.arange(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, color=_COLORS['blue'], linewidth=1.5,
                label='Train', alpha=0.85)
        ax.plot(epochs, val_losses, color=_COLORS['orange'], linewidth=1.5,
                label='Validation', alpha=0.85)

        if best_epoch is not None:
            ax.axvline(best_epoch, color='green', linestyle='--',
                      linewidth=1, alpha=0.7, label=f'Best Epoch ({best_epoch})')
            ax.scatter([best_epoch], [val_losses[best_epoch - 1]],
                      color='green', s=40, zorder=5)

        # 最优 epoch 标注
        best_val_epoch = int(np.argmin(val_losses)) + 1
        best_val = float(np.min(val_losses))
        ax.annotate(
            f'Best\nEpoch {best_val_epoch}\nVal={best_val:.4f}',
            xy=(best_val_epoch, best_val),
            xytext=(best_val_epoch + len(epochs) * 0.12, best_val * 1.15),
            fontsize=7, arrowprops=dict(arrowstyle='->', color='green', alpha=0.6),
            color='green'
        )

        ax.set_xlabel('Epoch', labelpad=3)
        ax.set_ylabel('Loss (MSE)', labelpad=3)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=_FONTSIZE_LEGEND)
        ax.grid(True, alpha=0.25)

        if log_scale:
            ax.set_yscale('log')

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_multi_model(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        model_names: Optional[List[str]] = None,
        time_index: Optional[np.ndarray] = None,
        title: str = "Multi-Model Comparison",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show_r2: bool = True,
        n_models_show: int = 4,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        多模型预测对比图 - 论文对比实验主图

        Args:
            results: {'ModelName': {'y_true': ..., 'y_pred': ...}}
            n_models_show: 最多显示几个模型（超过则分组）
        """
        if model_names is None:
            model_names = list(results.keys())

        n_models = len(model_names)
        if n_models > n_models_show:
            # 分组绘制
            fig_list = []
            for i in range(0, n_models, n_models_show):
                sub_models = model_names[i:i + n_models_show]
                sub_results = {k: results[k] for k in sub_models}
                fig = self._plot_multi_model_group(
                    sub_results, sub_models, time_index,
                    title=f"{title} ({i // n_models_show + 1})",
                    ylabel=ylabel, show_r2=show_r2,
                    figsize=figsize, dpi=dpi
                )
                fig_list.append(fig)
            return fig_list

        fig = self._plot_multi_model_group(
            results, model_names, time_index,
            title=title, ylabel=ylabel, show_r2=show_r2,
            figsize=figsize, dpi=dpi
        )

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def _plot_multi_model_group(
        self, results, model_names, time_index,
        title, ylabel, show_r2, figsize, dpi
    ) -> plt.Figure:
        n = len(next(iter(results.values()))['y_true'])
        if time_index is None:
            time_index = np.arange(n)

        fig, ax = plt.subplots(figsize=figsize or
                               (self.figure_width * 1.3, self.figure_width * 0.7),
                               constrained_layout=True)

        color_cycle = [_COLORS['blue'], _COLORS['orange'], _COLORS['teal'],
                       _COLORS['magenta']]

        # 真实值（加粗）
        y_true = next(iter(results.values()))['y_true']
        ax.plot(time_index, y_true, color='black', linewidth=2.0,
                label='Actual', alpha=0.9, zorder=10)

        for idx, model_name in enumerate(model_names):
            y_pred = results[model_name]['y_pred']
            color = color_cycle[idx % len(color_cycle)]
            r2 = results[model_name].get('r2')
            label = model_name
            if show_r2 and r2 is not None:
                label += f' ($R^2$={r2:.3f})'
            ax.plot(time_index, y_pred, color=color, linewidth=1.2,
                    linestyle='--', label=label, alpha=0.85)

        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel('Time Step', labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.legend(loc='best', framealpha=0.9, fontsize=_FONTSIZE_LEGEND,
                  ncol=2, columnspacing=1.0)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')
        return fig

    def plot_attention(
        self,
        attention_weights: np.ndarray,
        time_index: Optional[np.ndarray] = None,
        title: str = "Attention Heatmap",
        xlabel: str = "Key Position",
        ylabel: str = "Query Position",
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = 'Blues',
        dpi: int = 300,
    ) -> plt.Figure:
        """Attention 热力图 - Transformer/PatchTST 可解释性"""
        if figsize is None:
            figsize = (self.figure_width, self.figure_width * 0.85)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        seq_len = attention_weights.shape[0]
        if time_index is None:
            time_index = np.arange(seq_len)

        im = ax.imshow(attention_weights, aspect='auto', cmap=cmap,
                       interpolation='nearest')
        ax.set_xlabel(xlabel, labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.set_title(title, fontweight='bold', pad=8)

        ax.set_xticks(np.linspace(0, seq_len - 1, min(8, seq_len)).astype(int))
        ax.set_xticklabels([f'{time_index[i]:.0f}'
                           for i in ax.get_xticks()], fontsize=7)
        ax.set_yticks(np.linspace(0, seq_len - 1, min(8, seq_len)).astype(int))
        ax.set_yticklabels([f'{time_index[i]:.0f}'
                           for i in ax.get_yticks()], fontsize=7)

        cbar = plt.colorbar(im, ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('Attention Weight', fontsize=8)

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_frequency(
        self,
        signal: np.ndarray,
        sample_rate: float = 1.0,
        title: str = "Frequency Spectrum",
        xlabel: str = "Frequency (Hz)",
        ylabel: str = "Amplitude",
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """FFT 频谱图 - 传感器数据分析常用"""
        from scipy.fft import fft, fftfreq

        if figsize is None:
            figsize = (self.figure_width, self.figure_width * 0.5)

        n = len(signal)
        freqs = fftfreq(n, 1 / sample_rate)
        fft_vals = np.abs(fft(signal))
        positive_mask = freqs >= 0

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.plot(freqs[positive_mask], fft_vals[positive_mask],
                color=_COLORS['blue'], linewidth=0.9, alpha=0.8)
        ax.fill_between(freqs[positive_mask], fft_vals[positive_mask],
                        alpha=0.15, color=_COLORS['blue'])
        ax.set_xlabel(xlabel, labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, freqs[positive_mask].max())

        # 标注主峰
        peak_idx = np.argmax(fft_vals[positive_mask])
        peak_freq = freqs[positive_mask][peak_idx]
        if peak_freq > 0:
            ax.annotate(f'Peak: {peak_freq:.3f}Hz',
                       xy=(peak_freq, fft_vals[positive_mask][peak_idx]),
                       xytext=(peak_freq + 0.05, fft_vals[positive_mask][peak_idx] * 0.9),
                       fontsize=7, arrowprops=dict(arrowstyle='->', color='red', alpha=0.6),
                       color='red')

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_cross_validation(
        self,
        cv_results: Dict[int, Dict[str, np.ndarray]],
        title: str = "Time Series Cross-Validation Rolling Prediction",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        时序交叉验证滚动预测可视化

        Args:
            cv_results: {fold_id: {'y_true': ..., 'y_pred': ..., 'time_index': ...}}
        """
        if figsize is None:
            figsize = (self.figure_width * 1.4, self.figure_width * 0.8)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        colors = [_COLORS['blue'], _COLORS['orange'], _COLORS['teal'],
                 _COLORS['magenta'], _COLORS['grey']]

        all_time = []
        all_true = []
        r2_list = []

        for fold_id, result in cv_results.items():
            y_true = result['y_true']
            y_pred = result['y_pred']
            time_idx = result.get('time_index', np.arange(len(y_true)))

            color = colors[fold_id % len(colors)]
            offset = fold_id * 0.5  # 错开绘制避免重叠

            ax.plot(time_idx, y_true + offset, color=color, linewidth=1.2,
                    alpha=0.8, label=f'Fold {fold_id+1} Actual')
            ax.plot(time_idx, y_pred + offset, color=color, linewidth=1.0,
                    linestyle='--', alpha=0.7, label=f'Fold {fold_id+1} Pred')

            all_time.extend(time_idx)
            all_true.extend(y_true)

            r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)
            r2_list.append(r2)

        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel('Time Step', labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.legend(loc='best', fontsize=6, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.25)

        # 内嵌各折 R²
        avg_r2 = np.mean(r2_list)
        textstr = f'Mean $R^2$ = {avg_r2:.4f}\nFolds: {", ".join([f"{r:.3f}" for r in r2_list])}'
        props = dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='gray', alpha=0.85)
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes,
                fontsize=7, verticalalignment='top', bbox=props, family='monospace')

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_forecast_with_ci(
        self,
        y_history: np.ndarray,
        y_forecast: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        forecast_index: np.ndarray,
        history_index: Optional[np.ndarray] = None,
        title: str = "Time Series Forecast with Confidence Intervals",
        ylabel: str = "Value",
        confidence: float = 0.95,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        预测图 - 历史 + 预测 + 置信区间，最适合论文使用

        Args:
            y_history: 历史数据（用于展示上下文）
            y_forecast: 预测值
            ci_lower / ci_upper: 置信区间上下界
            forecast_index: 预测段的 x 轴
            history_index: 历史段的 x 轴（可选）
        """
        if figsize is None:
            figsize = (self.figure_width * 1.4, self.figure_width * 0.6)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        if history_index is None:
            history_index = np.arange(-len(y_history), 0)

        # 历史数据
        ax.plot(history_index, y_history, color=_COLORS['blue'],
                linewidth=1.5, label='Historical Data', alpha=0.9)

        # 预测分界线
        ax.axvline(history_index[-1], color='gray', linestyle=':',
                   linewidth=1, alpha=0.7)
        ax.text(history_index[-1], ax.get_ylim()[1] * 0.95,
                'Forecast →', fontsize=7, color='gray',
                ha='right', style='italic')

        # 置信区间
        ci_alpha = 0.15 if confidence >= 0.95 else 0.25
        ax.fill_between(forecast_index, ci_lower, ci_upper,
                        alpha=ci_alpha, color=_COLORS['light_orange'],
                        label=f'{confidence:.0%} Confidence Interval')

        # 预测值
        ax.plot(forecast_index, y_forecast, color=_COLORS['orange'],
                linewidth=1.5, linestyle='--', label='Forecast', alpha=0.9)

        # 标注预测终点
        ax.scatter([forecast_index[-1]], [y_forecast[-1]],
                   color=_COLORS['orange'], s=30, zorder=5)
        ax.annotate(f'{y_forecast[-1]:.2f}',
                   xy=(forecast_index[-1], y_forecast[-1]),
                   xytext=(forecast_index[-1] + 0.5, y_forecast[-1] * 1.02),
                   fontsize=7, color=_COLORS['orange'])

        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel('Time Step', labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=_FONTSIZE_LEGEND)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        title: str = "Feature Importance",
        top_k: int = 15,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        color: str = _COLORS['blue'],
        dpi: int = 300,
    ) -> plt.Figure:
        """特征重要性条形图"""
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        sorted_features = sorted_features[:top_k]

        if figsize is None:
            figsize = (self.figure_width, self.figure_width * 0.6)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        names = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=color, alpha=0.8, height=0.6)

        # 数值标签
        for bar, val in zip(bars, values):
            ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', labelpad=3)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.25, axis='x')

        _apply_style(self.style, fig, ax)
        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    # ───────────────────────────────────────────────────────────────
    # 导出工具
    # ───────────────────────────────────────────────────────────────

    def export_all(
        self,
        export_dir: str = "./paper_figures",
        dpi: int = 300,
        formats: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        train_losses: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None,
        model_name: str = "model",
    ) -> Dict[str, str]:
        """
        一键导出论文所需全套图表

        自动生成:
        - fig1_prediction.png  (预测时序图)
        - fig2_residuals.png    (残差分析)
        - fig3_scatter.png      (散点图)
        - fig4_loss.png         (损失曲线)
        - table1_metrics.png + .tex (指标表)

        Args:
            export_dir: 导出目录
            dpi: 图片分辨率（论文通常 300）
            formats: 额外格式 ['pdf', 'svg']
            metrics: 模型指标字典
            y_true / y_pred: 预测数据
            train_losses / val_losses: 损失数据

        Returns:
            {'fig_name': 'saved_path', ...}
        """
        if formats is None:
            formats = []
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        def _save(fig, name, extra_formats=None):
            path_png = out_dir / f"{name}.png"
            fig.savefig(path_png, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            saved[name] = str(path_png)
            for fmt in (extra_formats or []):
                path_fmt = out_dir / f"{name}.{fmt}"
                fig.savefig(path_fmt, dpi=dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                saved[f"{name}.{fmt}"] = str(path_fmt)
            plt.close(fig)

        if y_true is not None and y_pred is not None:
            fig1 = self.plot_prediction(y_true, y_pred, title=f"{model_name} Prediction",
                                       save_path=None)
            _save(fig1, "fig1_prediction", formats)

            fig2 = self.plot_residuals(y_true, y_pred,
                                       title=f"{model_name} Residual Analysis",
                                       save_path=None)
            _save(fig2, "fig2_residuals", formats)

            fig3 = self.plot_scatter(y_true, y_pred,
                                     title=f"{model_name} Actual vs Predicted",
                                     save_path=None)
            _save(fig3, "fig3_scatter", formats)

        if train_losses is not None and val_losses is not None:
            fig4 = self.plot_loss_curve(train_losses, val_losses,
                                        title=f"{model_name} Training Loss",
                                        save_path=None)
            _save(fig4, "fig4_loss", formats)

        if metrics is not None:
            fig5 = self.plot_metrics_table(metrics, title="Model Performance Comparison",
                                           save_path=None)
            _save(fig5, "table1_metrics", formats)

        return saved

    def get_figure(self, name: str) -> Optional[plt.Figure]:
        return self._saved_figures.get(name)


# ─────────────────────────────────────────────────────────────────
# 快捷函数（非面向对象方式调用）
# ─────────────────────────────────────────────────────────────────

def quick_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    style: str = JournalStyle.IEEE,
    save_path: str = "figure.png",
    title: str = "Prediction vs Actual",
    **kwargs
) -> plt.Figure:
    """一行代码生成论文级预测图"""
    plotter = PublicationPlotter(style=style)
    fig = plotter.plot_prediction(y_true, y_pred, title=title,
                                  save_path=save_path, **kwargs)
    return fig

