"""
论文图表生成示例
一键生成 publication-quality 图表

运行方法:
    cd DeepPredict
    python examples/paper_figures_demo.py

生成图表:
    ./paper_figures/
    ├── fig1_prediction.png     # 预测时序图（含指标）
    ├── fig2_residuals.png      # 残差分析（4子图）
    ├── fig3_scatter.png        # 实际 vs 预测散点
    ├── fig4_loss.png           # 训练损失曲线
    ├── table1_metrics.png       # 指标汇总表
    └── table1_metrics.tex       # LaTeX 三线表（直接粘贴）
"""

import numpy as np
from src.utils.plotting import PublicationPlotter, JournalStyle

# ── 模拟数据 ──────────────────────────────────────────────
np.random.seed(42)
n = 300
time = np.arange(n)
# 真实信号：正弦波 + 噪声
y_true = 5 * np.sin(2 * np.pi * time / 50) + 2 * np.sin(2 * np.pi * time / 20) + np.random.randn(n) * 0.3
# 模拟预测：加入偏移和噪声
y_pred = y_true + np.random.randn(n) * 0.25
# 置信区间
ci = 1.96 * 0.3
y_lower = y_pred - ci
y_upper = y_pred + ci

# ── 模拟指标 ──────────────────────────────────────────────
metrics = {
    'CNN1D-V4':    {'R2': 0.8472, 'RMSE': 0.3125, 'MAE': 0.2418, 'MAPE': 5.83},
    'LSTM':        {'R2': 0.7913, 'RMSE': 0.3891, 'MAE': 0.3012, 'MAPE': 7.14},
    'PatchTST':    {'R2': 0.8234, 'RMSE': 0.3567, 'MAE': 0.2789, 'MAPE': 6.52},
    'GRU':         {'R2': 0.7621, 'RMSE': 0.4102, 'MAE': 0.3187, 'MAPE': 7.68},
}

# ── 模拟损失曲线 ──────────────────────────────────────────
epochs = 80
train_losses = [2.5 * np.exp(-e / 20) + 0.1 + np.random.randn() * 0.05 for e in range(epochs)]
val_losses = [2.6 * np.exp(-e / 18) + 0.12 + np.random.randn() * 0.08 for e in range(epochs)]

# ── 选择期刊样式 ──────────────────────────────────────────
# 可选: 'ieee' | 'nature' | 'science' | 'bw'（黑白打印友好）
STYLE = 'ieee'
EXPORT_DIR = './paper_figures'

plotter = PublicationPlotter(style=STYLE)
print(f"使用样式: {STYLE}")

# ── 1. 预测时序图（含置信区间）────────────────────────────
fig1 = plotter.plot_prediction(
    y_true, y_pred, y_lower, y_upper, time_index=time,
    title="CNN1D-V4 Prediction on Sensor Time Series",
    xlabel="Time Step (s)", ylabel="Sensor Reading (mV)",
    labels={'true': 'Actual', 'pred': 'Predicted', 'ci': '95% CI'},
    save_path=f"{EXPORT_DIR}/fig1_prediction.png",
    dpi=300
)
print("✅ fig1_prediction.png")

# ── 2. 残差分析（4 子图）──────────────────────────────────
fig2 = plotter.plot_residuals(
    y_true, y_pred, time_index=time,
    title="CNN1D-V4 Residual Analysis",
    save_path=f"{EXPORT_DIR}/fig2_residuals.png", dpi=300
)
print("✅ fig2_residuals.png")

# ── 3. 散点图 ───────────────────────────────────────────
fig3 = plotter.scatter(
    y_true, y_pred,
    title="CNN1D-V4 Actual vs Predicted",
    save_path=f"{EXPORT_DIR}/fig3_scatter.png", dpi=300
)
print("✅ fig3_scatter.png")

# ── 4. 损失曲线 ─────────────────────────────────────────
fig4 = plotter.plot_loss_curve(
    train_losses, val_losses,
    title="CNN1D-V4 Training Loss",
    save_path=f"{EXPORT_DIR}/fig4_loss.png", dpi=300
)
print("✅ fig4_loss.png")

# ── 5. 多模型对比 ───────────────────────────────────────
multi_results = {
    name: {'y_true': y_true, 'y_pred': y_true + np.random.randn(n) * 0.3, 'r2': m['R2']}
    for name, m in metrics.items()
}
fig5 = plotter.plot_multi_model(
    multi_results, time_index=time,
    title="Multi-Model Prediction Comparison",
    ylabel="Sensor Reading (mV)",
    save_path=f"{EXPORT_DIR}/fig5_multi_model.png", dpi=300
)
print("✅ fig5_multi_model.png")

# ── 6. 指标汇总表（PNG + LaTeX）──────────────────────────
fig6 = plotter.plot_metrics_table(
    metrics,
    title="Model Performance Comparison",
    caption="Performance metrics on test set. Best values in bold.",
    save_path=f"{EXPORT_DIR}/table1_metrics.png",
    dpi=300, export_latex=True
)
print("✅ table1_metrics.png + .tex")

# ── 7. 一键导出全套 ──────────────────────────────────────
saved = plotter.export_all(
    export_dir=EXPORT_DIR, dpi=300,
    formats=['pdf'],  # 同时导出 PDF（矢量格式，期刊首选）
    metrics=metrics,
    y_true=y_true, y_pred=y_pred,
    train_losses=train_losses, val_losses=val_losses,
    model_name="CNN1D-V4",
)
print(f"\n📊 全套图表已导出至: {EXPORT_DIR}/")
for name, path in saved.items():
    print(f"   {name}")
