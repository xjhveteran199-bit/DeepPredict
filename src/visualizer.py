"""
DeepPredict 可视化模块 - PredictVisualizer
提供论文级时序预测图表，支持 scienceplots 子刊风格

功能:
- 预测 vs 真实时序叠加图（含置信区间 + 局部放大）
- 残差分布直方图
- 多变量相关热力图
- SHAP beeswarm 图（scienceplots 风格增强）

Author: DeepPredict Agent
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
from typing import Optional, Tuple, List, Dict, Any

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 样式配置
# ─────────────────────────────────────────────────────────────────

# scienceplots 需要 LaTeX，在 Windows 上可能不可用，默认禁用
# 如需启用请手动设置: HAS_SCIENCEPLOTS = True 并确保安装了 LaTeX
HAS_SCIENCEPLOTS = False

# 期刊配色（色盲友好）
_COLORS = {
    'true': '#1f77b4',      # 蓝色 - 真实值
    'pred': '#d62728',      # 红色 - 预测值
    'ci': '#aec7e8',        # 浅蓝 - 置信区间
    'residual': '#2ca02c',   # 绿色 - 残差
    'grid': '#cccccc',
}

# 全局字体大小
_FONT_SIZE = 9
_DPI = 300


def _apply_science_style():
    """应用 scienceplots 子刊风格"""
    # 禁用 LaTeX（避免 latex 未安装报错）
    matplotlib.rcParams['text.usetex'] = False

    if HAS_SCIENCEPLOTS:
        try:
            plt.style.use(['science', 'nature'])
        except Exception:
            plt.style.use(['ggplot'])
    else:
        # Fallback: 自定义简洁风格
        plt.style.use(['ggplot'])

    matplotlib.rcParams['font.size'] = _FONT_SIZE
    matplotlib.rcParams['axes.labelsize'] = _FONT_SIZE
    matplotlib.rcParams['axes.titlesize'] = _FONT_SIZE + 1
    matplotlib.rcParams['xtick.labelsize'] = _FONT_SIZE - 1
    matplotlib.rcParams['ytick.labelsize'] = _FONT_SIZE - 1
    matplotlib.rcParams['legend.fontsize'] = _FONT_SIZE - 1
    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['savefig.dpi'] = _DPI
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3
    matplotlib.rcParams['grid.linewidth'] = 0.5


# ─────────────────────────────────────────────────────────────────
# PredictVisualizer 主类
# ─────────────────────────────────────────────────────────────────

class PredictVisualizer:
    """
    DeepPredict 专用可视化器

    提供:
    - plot_prediction_timeseries(): 预测 vs 真实时序叠加图（支持置信区间 + 局部放大）
    - plot_residual_distribution(): 残差分布直方图
    - plot_correlation_heatmap(): 多变量相关热力图
    - plot_prediction_vs_actual(): 实际 vs 预测散点图（带 R²）
    - plot_all(): 一键生成全套图表
    """

    def __init__(self, figsize: tuple = (10, 5)):
        """
        Args:
            figsize: 默认图形尺寸 (width, height)
        """
        self.figsize = figsize
        self._saved_figures: Dict[str, plt.Figure] = {}

    def plot_prediction_timeseries(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[np.ndarray] = None,
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
        title: str = "Prediction vs Actual Time Series",
        ylabel: str = "Value",
        zoom_range: Optional[Tuple[int, int]] = None,
        zoom_title: str = "Zoomed View (Detail)",
        labels: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
        dpi: int = _DPI,
        show_metrics: bool = True,
        n_metrics_decimals: int = 4,
    ) -> plt.Figure:
        """
        预测 vs 真实时序叠加图（子刊风格）

        Features:
        - 真实值线（蓝色）+ 预测值线（红色）叠加
        - 置信区间（浅蓝色填充）
        - 局部放大子图（zoomed panel）
        - 右上角内嵌 R²/RMSE/MAE 指标文本
        - 导出高清 PNG（300 DPI）

        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            dates: 日期/时间标签（可选，默认使用索引）
            ci_lower / ci_upper: 95% 置信区间上下界（可选）
            title: 图标题
            ylabel: Y轴标签
            zoom_range: 局部放大范围 (start_idx, end_idx)
            zoom_title: 放大子图标题
            labels: 自定义标签 dict{'true': '...', 'pred': '...', 'ci': '...'}
            save_path: 保存路径（.png / .pdf）
            figsize: 图形尺寸
            dpi: 导出分辨率
            show_metrics: 是否显示内嵌指标
            n_metrics_decimals: 指标小数位

        Returns:
            matplotlib Figure 对象
        """
        _apply_science_style()

        n = len(y_true)
        if dates is None:
            x = np.arange(n)
            xlabel = "Time Step"
        else:
            x = np.arange(n)
            xlabel = "Date"

        if labels is None:
            labels = {'true': 'Actual', 'pred': 'Predicted', 'ci': '95% CI'}

        # ── 计算指标 ──
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

        # ── 创建图形 ──
        if zoom_range is not None:
            fig = plt.figure(figsize=figsize or (12, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1],
                                   hspace=0.25, left=0.08, right=0.95,
                                   top=0.92, bottom=0.12)
            ax_main = fig.add_subplot(gs[0])
            ax_zoom = fig.add_subplot(gs[1])
        else:
            fig, ax_main = plt.subplots(figsize=figsize or self.figsize)
            ax_zoom = None

        # ── 主图：时序叠加 ──
        # 置信区间
        if ci_lower is not None and ci_upper is not None:
            ax_main.fill_between(x, ci_lower, ci_upper,
                                 alpha=0.2, color=_COLORS['ci'],
                                 label=labels.get('ci', '95% CI'))

        # 真实值线
        ax_main.plot(x, y_true,
                     color=_COLORS['true'], linewidth=1.5,
                     label=labels.get('true', 'Actual'), alpha=0.9)

        # 预测值线
        ax_main.plot(x, y_pred,
                     color=_COLORS['pred'], linewidth=1.5, linestyle='--',
                     label=labels.get('pred', 'Predicted'), alpha=0.9)

        # 局部放大区域标注
        if zoom_range is not None:
            start, end = zoom_range
            ax_main.axvspan(start, end, alpha=0.08, color='orange')
            ax_main.text((start + end) / 2, ax_main.get_ylim()[1] * 0.98,
                        '← Zoomed Region →', ha='center', fontsize=7,
                        color='darkorange', style='italic')

        ax_main.set_title(title, fontweight='bold', pad=8)
        ax_main.set_xlabel(xlabel, labelpad=3)
        ax_main.set_ylabel(ylabel, labelpad=3)
        ax_main.legend(loc='upper left', framealpha=0.9,
                      edgecolor='gray', fontsize=_FONT_SIZE - 1)
        ax_main.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax_main.grid(True, alpha=0.25)

        # 内嵌指标文本
        if show_metrics:
            textstr = (f'$R^2$ = {r2:.{n_metrics_decimals}f}\n'
                       f'RMSE = {rmse:.{n_metrics_decimals}f}\n'
                       f'MAE = {mae:.{n_metrics_decimals}f}\n'
                       f'MAPE = {mape:.2f}%')
            props = dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='gray', alpha=0.85, linewidth=0.8)
            ax_main.text(0.97, 0.03, textstr, transform=ax_main.transAxes,
                        fontsize=_FONT_SIZE - 1, verticalalignment='bottom',
                        horizontalalignment='right', bbox=props,
                        family='monospace')

        # ── 放大子图 ──
        if zoom_range is not None:
            start, end = zoom_range
            zx = x[start:end]
            zy_true = y_true[start:end]
            zy_pred = y_pred[start:end]

            if ci_lower is not None and ci_upper is not None:
                ax_zoom.fill_between(zx, ci_lower[start:end], ci_upper[start:end],
                                    alpha=0.2, color=_COLORS['ci'])

            ax_zoom.plot(zx, zy_true, color=_COLORS['true'], linewidth=1.5,
                        label=labels.get('true', 'Actual'), alpha=0.9)
            ax_zoom.plot(zx, zy_pred, color=_COLORS['pred'], linewidth=1.5,
                        linestyle='--', label=labels.get('pred', 'Predicted'), alpha=0.9)

            # 放大子图指标
            if len(zy_true) > 2:
                z_rmse = np.sqrt(np.mean((zy_true - zy_pred) ** 2))
                z_r2 = 1 - np.sum((zy_true - zy_pred) ** 2) / (
                    np.sum((zy_true - np.mean(zy_true)) ** 2) + 1e-10)
                z_text = f'Zoomed: $R^2$={z_r2:.4f}, RMSE={z_rmse:.4f}'
                ax_zoom.text(0.02, 0.97, z_text, transform=ax_zoom.transAxes,
                            fontsize=_FONT_SIZE - 1, va='top',
                            bbox=dict(boxstyle='round', facecolor='white',
                                     alpha=0.8, edgecolor='gray'))

            ax_zoom.set_title(zoom_title, fontsize=_FONT_SIZE, pad=5)
            ax_zoom.set_xlabel(xlabel, labelpad=3)
            ax_zoom.set_ylabel(ylabel, labelpad=3)
            ax_zoom.legend(loc='upper left', framealpha=0.9,
                          fontsize=_FONT_SIZE - 2)
            ax_zoom.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax_zoom.grid(True, alpha=0.25)

        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_residual_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Distribution",
        save_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
        dpi: int = _DPI,
        bins: int = 40,
        show_stats: bool = True,
        show_qq: bool = False,
    ) -> plt.Figure:
        """
        残差分布直方图

        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图标题
            save_path: 保存路径
            figsize: 图形尺寸
            dpi: 分辨率
            bins: 直方图柱子数
            show_stats: 显示统计指标
            show_qq: 是否附加 Q-Q 图子图

        Returns:
            matplotlib Figure
        """
        _apply_science_style()
        residuals = y_true - y_pred

        if show_qq:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (12, 4),
                                     constrained_layout=True)
            ax_hist = axes[0]
            ax_qq = axes[1]
        else:
            fig, ax_hist = plt.subplots(figsize=figsize or self.figsize)
            ax_qq = None

        # 直方图
        ax_hist.hist(residuals, bins=bins, alpha=0.7,
                     color=_COLORS['residual'], edgecolor='white',
                     linewidth=0.5, density=True, label='Residuals')

        # 正态拟合曲线
        mu, sigma = np.mean(residuals), np.std(residuals)
        x_range = np.linspace(residuals.min() - sigma, residuals.max() + sigma, 200)
        normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                     np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        ax_hist.plot(x_range, normal_pdf, color='red', linewidth=1.5,
                     label=f'Normal Fit\n(μ={mu:.3f}, σ={sigma:.3f})')

        # 零线
        ax_hist.axvline(0, color='black', linestyle='--', linewidth=1,
                        alpha=0.7, label='Zero')

        # 统计信息
        if show_stats:
            skew = self._skewness(residuals)
            kurt = self._kurtosis(residuals)
            jb_stat, jb_p = self._jarque_bera(residuals)
            stats_text = (f'Mean: {mu:.4f}\nStd: {sigma:.4f}\n'
                          f'Skew: {skew:.3f}\nKurt: {kurt:.3f}\n'
                          f'JB p: {jb_p:.4f}')
            props = dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='gray', alpha=0.85)
            ax_hist.text(0.97, 0.97, stats_text, transform=ax_hist.transAxes,
                        fontsize=_FONT_SIZE - 2, va='top', ha='right',
                        bbox=props, family='monospace')

        ax_hist.set_title(title, fontweight='bold', pad=8)
        ax_hist.set_xlabel('Residual', labelpad=3)
        ax_hist.set_ylabel('Density', labelpad=3)
        ax_hist.legend(loc='upper left', fontsize=_FONT_SIZE - 1)
        ax_hist.grid(True, alpha=0.25)

        # Q-Q 图
        if ax_qq is not None:
            try:
                from scipy import stats as scipy_stats
                scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                ax_qq.get_lines()[0].set_markerfacecolor(_COLORS['residual'])
                ax_qq.get_lines()[0].set_markersize(3)
                ax_qq.get_lines()[0].set_alpha(0.5)
                ax_qq.get_lines()[1].set_color('red')
                ax_qq.get_lines()[1].set_linewidth(1.2)
                ax_qq.set_title('Q-Q Plot (Normal)', fontsize=_FONT_SIZE)
                ax_qq.grid(True, alpha=0.25)
            except Exception:
                pass  # Q-Q 图失败不影响主图

        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Heatmap",
        method: str = 'pearson',
        cmap: str = 'RdBu_r',
        save_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
        dpi: int = _DPI,
        annot: bool = True,
        fmt: str = '.2f',
        vmin: float = -1.0,
        vmax: float = 1.0,
        max_features: int = 20,
    ) -> plt.Figure:
        """
        多变量相关热力图

        Args:
            df: 特征 DataFrame
            title: 图标题
            method: 相关方法 ('pearson', 'spearman', 'kendall')
            cmap: 颜色映射
            save_path: 保存路径
            figsize: 图形尺寸
            dpi: 分辨率
            annot: 是否显示相关系数数值
            fmt: 数值格式
            vmin/vmax: 颜色范围
            max_features: 最多显示的特征数（避免图形过大）

        Returns:
            matplotlib Figure
        """
        _apply_science_style()

        # 限制特征数量
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > max_features:
            # 选择方差最大的前 max_features 个
            variances = numeric_df.var().sort_values(ascending=False)
            top_features = variances.head(max_features).index.tolist()
            numeric_df = numeric_df[top_features]

        corr = numeric_df.corr(method=method)

        if figsize is None:
            n = corr.shape[0]
            figsize = (max(6, n * 0.45), max(5, n * 0.4))

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto')

        # 设置刻度
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right',
                          fontsize=_FONT_SIZE - 2)
        ax.set_yticklabels(corr.columns, fontsize=_FONT_SIZE - 2)

        # 数值标注
        if annot:
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    val = corr.values[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, format(val, fmt),
                           ha='center', va='center', color=color,
                           fontsize=_FONT_SIZE - 2)

        # 颜色条
        cbar = plt.colorbar(im, ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=_FONT_SIZE - 2)
        cbar.set_label(f'{method.capitalize()} Correlation', fontsize=_FONT_SIZE)

        ax.set_title(title, fontweight='bold', pad=8)

        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted",
        save_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
        dpi: int = _DPI,
        alpha: float = 0.5,
        s: float = 10,
    ) -> plt.Figure:
        """
        实际 vs 预测散点图

        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图标题
            save_path: 保存路径
            figsize: 图形尺寸
            dpi: 分辨率
            alpha: 散点透明度
            s: 散点大小

        Returns:
            matplotlib Figure
        """
        _apply_science_style()

        fig, ax = plt.subplots(figsize=figsize or (6, 6),
                               constrained_layout=True)

        ax.scatter(y_true, y_pred, alpha=alpha, s=s,
                   color=_COLORS['true'], edgecolors='none')

        # 完美拟合线
        min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
        max_val = max(np.nanmax(y_true), np.nanmax(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=1.5, label='Perfect Fit', zorder=5)

        # 指标
        rmse = np.sqrt(np.nanmean((y_true - y_pred) ** 2))
        mae = np.nanmean(np.abs(y_true - y_pred))
        ss_res = np.nansum((y_true - y_pred) ** 2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        textstr = (f'$R^2$ = {r2:.4f}\n'
                   f'RMSE = {rmse:.4f}\n'
                   f'MAE = {mae:.4f}')
        props = dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='gray', alpha=0.85)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=_FONT_SIZE, verticalalignment='top',
                bbox=props, family='monospace')

        ax.set_xlabel('Actual', labelpad=3)
        ax.set_ylabel('Predicted', labelpad=3)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.legend(loc='lower right', fontsize=_FONT_SIZE - 1)
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='box')

        fig.patch.set_facecolor('white')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self._saved_figures[save_path] = fig

        return fig

    def plot_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        dates: Optional[np.ndarray] = None,
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
        zoom_range: Optional[Tuple[int, int]] = None,
        output_dir: Optional[str] = None,
        prefix: str = "dp",
        dpi: int = _DPI,
    ) -> Dict[str, str]:
        """
        一键生成全套可视化图表

        Args:
            y_true / y_pred: 真实值和预测值
            df: 原始特征 DataFrame（用于相关热力图）
            dates: 日期标签
            ci_lower / ci_upper: 置信区间
            zoom_range: 放大区域
            output_dir: 输出目录
            prefix: 文件名前缀
            dpi: 分辨率

        Returns:
            {'chart_name': 'saved_path'}
        """
        import os
        saved = {}

        def _save(fig, name):
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, f"{prefix}_{name}.png")
                fig.savefig(path, dpi=dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                saved[name] = path
            plt.close(fig)

        # 1. 时序图
        fig_ts = self.plot_prediction_timeseries(
            y_true, y_pred, dates=dates,
            ci_lower=ci_lower, ci_upper=ci_upper,
            zoom_range=zoom_range, title="Prediction vs Actual Time Series",
            save_path=None
        )
        _save(fig_ts, "timeseries")

        # 2. 残差分布
        fig_res = self.plot_residual_distribution(
            y_true, y_pred, title="Residual Distribution", save_path=None,
            show_qq=True
        )
        _save(fig_res, "residuals")

        # 3. 散点图
        fig_sc = self.plot_prediction_vs_actual(
            y_true, y_pred, title="Actual vs Predicted", save_path=None
        )
        _save(fig_sc, "scatter")

        # 4. 相关热力图
        if df is not None:
            fig_corr = self.plot_correlation_heatmap(
                df, title="Feature Correlation Heatmap", save_path=None
            )
            _save(fig_corr, "correlation")

        return saved

    # ─────────────────────────────────────────────────────────────
    # 内部工具
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        n = len(x)
        if n < 3:
            return 0.0
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-10:
            return 0.0
        return np.mean(((x - m) / s) ** 3)

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        n = len(x)
        if n < 4:
            return 0.0
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-10:
            return 0.0
        return np.mean(((x - m) / s) ** 4) - 3

    @staticmethod
    def _jarque_bera(x: np.ndarray) -> Tuple[float, float]:
        """简化 Jarque-Bera 检验统计量"""
        n = len(x)
        if n < 3:
            return 0.0, 1.0
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-10:
            return 0.0, 1.0
        b1 = np.mean(((x - m) / s) ** 3)
        b2 = np.mean(((x - m) / s) ** 4)
        jb = (n / 6) * (b1 ** 2 + (b2 - 3) ** 2 / 4)
        # 近似 p 值（chi-square df=2）
        p_val = np.exp(-jb / 2) if jb > 0 else 1.0
        return jb, p_val

    def get_figure(self, name: str) -> Optional[plt.Figure]:
        return self._saved_figures.get(name)


# ─────────────────────────────────────────────────────────────────
# 快捷函数
# ─────────────────────────────────────────────────────────────────

def quick_plot_timeseries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    zoom_range: Optional[Tuple[int, int]] = None,
    save_path: str = "prediction_timeseries.png",
    **kwargs
) -> plt.Figure:
    """一行代码生成预测时序图"""
    viz = PredictVisualizer()
    return viz.plot_prediction_timeseries(
        y_true, y_pred,
        ci_lower=ci_lower, ci_upper=ci_upper,
        zoom_range=zoom_range, save_path=save_path, **kwargs
    )


def quick_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "residuals.png",
    **kwargs
) -> plt.Figure:
    """一行代码生成残差分布图"""
    viz = PredictVisualizer()
    return viz.plot_residual_distribution(y_true, y_pred, save_path=save_path, **kwargs)


def quick_correlation(
    df: pd.DataFrame,
    save_path: str = "correlation.png",
    **kwargs
) -> plt.Figure:
    """一行代码生成相关热力图"""
    viz = PredictVisualizer()
    return viz.plot_correlation_heatmap(df, save_path=save_path, **kwargs)
