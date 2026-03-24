# DeepPredict Agent 专属记忆

## Agent 身份
- **职责**：全职负责 DeepPredict 模块的迭代更新
- **上级**：主 Agent（项目经理人）
- **报告频率**：每完成一个功能主动汇报给项目经理人

## 模块基本信息
- **模块路径**：`C:\Users\XJH\DeepPredict\`
- **版本**：v1.13（2026-03-24）
- **定位**：时序/回归预测，面向研究者的低门槛深度学习工具
- **Git 远程**：`https://github.com/xjhveteran199-bit/DeepResearch`

## 当前版本功能
- **6种预测模型**：RandomForest / GradientBoosting / LSTM / CNN1D / PatchTST / LinearRegression
- **多模态输入**：支持选择多个 X 特征列 → 预测 Y 目标列
- **SHAP 可解释性**：特征重要性 + beeswarm 图（scienceplots 子刊风格）
- **数据预处理**：自动归一化、缺失值处理
- **结果可视化**：预测时序图（含置信区间+局部放大）、残差分布直方图、相关热力图
- **一键下载**：CSV + 指标 + SHAP 图

## 技术栈
- Python 3.12 + PyTorch 2.5.0（CPU）+ scikit-learn + Gradio 6.9.0
- numpy 1.26.4 + shap 0.51.0

## 当前测试结果（每日最低温度数据集）
| 模型 | R² | RMSE | 状态 |
|------|-----|------|------|
| RandomForest | 0.610 | 2.55 | ✅ |
| GradientBoosting | 0.611 | 2.55 | ✅ |
| LSTM | 0.6258 | 2.61 | ✅ |
| CNN1D | 0.6538 | 2.24 | ✅ |

## 已知问题
- CNN1D R²=0.449 → 已优化至 R²=0.6538（seq=180, hidden=128, kernel=5, layers=3, epochs=100）
- seq=365 配置内存不足，seq=180 效果最优
- PatchTST 完整参数测试未跑

## 迭代 Roadmap
1. ✅ 多模态输入（X多列选择）
2. ✅ SHAP 可解释性分析
3. ✅ CNN1D 参数调优（R²=0.6538 ≥ 0.55 目标达成）
4. 🔄 PatchTST 完整测试
5. 🔄 数据集自动特征工程（sin/cos 周期编码）
6. 🔄 模型缓存（避免重复训练 API 调用）
7. 🔄 FastAPI 服务化

## 汇报规则
每完成一个功能后，向主 Agent（项目经理人）汇报：
```
【DeepPredict 进展汇报】
时间: [现在时间]
本次更新: [具体内容]
修改文件: [文件名]
状态: 正常/有问题
下一步: [计划]
```

## 最近更新记录
| 日期 | 更新内容 | Commit |
|------|---------|--------|
| 2026-03-24 | v1.13 新增 PredictVisualizer 可视化模块（时序图+置信区间+局部放大） | (本次) |
| 2026-03-24 | v1.12 CNN1D优化 R²=0.6538 | (本次) |
| 2026-03-24 | v1.11 初始版本，多模态+SHAP | 21a3388 |
| 2026-03-24 | LSTM参数优化 R²=0.6258 | ee29395 |
