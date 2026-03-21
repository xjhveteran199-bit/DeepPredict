# Changelog

All notable changes to DeepPredict will be documented in this file.

## [1.0.0] - 2026-03-21

### Added
- PyQt5 桌面应用主界面
- CSV 数据导入与预览
- 自然语言需求描述解析
- 自动任务类型识别（回归/分类/时序）
- 自动模型适配（LSTM/GradientBoosting/RandomForest/XGBoost）
- LSTM 深度学习时序预测（PyTorch）
- 训练指标评估（R²/RMSE/MAE/准确率/F1）
- 特征重要性分析
- 模型保存与加载
- Windows 单文件 exe 打包（PyInstaller）
- 桌面快捷方式自动创建

### Features
- 拖拽式 CSV 导入
- 实时训练日志显示
- 预测结果表格展示
- 特征重要性 Top10 展示

### Technical Stack
- PyQt5 (GUI)
- pandas + numpy (Data Processing)
- scikit-learn (Machine Learning)
- PyTorch (Deep Learning - LSTM)
- matplotlib + pyqtgraph (Visualization)
- PyInstaller (Packaging)
