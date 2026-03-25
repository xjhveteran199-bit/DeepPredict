# DeepPredict Agent 专属记忆

## Agent 身份
- **职责**：全职负责 DeepPredict 模块的迭代更新
- **上级**：主 Agent（项目经理人）
- **报告频率**：每完成一个功能主动汇报给项目经理人

## 模块基本信息
- **模块路径**：`C:\Users\XJH\DeepResearch\DeepPredict\`（已迁移整合）
- **原独立项目**：`C:\Users\XJH\DeepPredict\`（已归档）
- **版本**：v1.14（2026-03-25）
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

## CNN1D 参数调优结果（2026-03-24）
- **目标**: R² ≥ 0.55
- **最优配置**: seq_len=180, epochs=100, hidden_channels=128, kernel_size=5
- **R²**: 0.6538 ✅ (超出目标)
- **RMSE**: 2.24
- **测试数据集**: dp_temperature.csv (3650 样本)

## 当前测试结果（每日最低温度数据集）
| 模型 | R² | RMSE | 状态 |
|------|-----|------|------|
| RandomForest | 0.610 | 2.55 | ✅ |
| GradientBoosting | 0.611 | 2.55 | ✅ |
| LSTM | 0.6258 | 2.61 | ✅ |
| CNN1D | 0.6538 | 2.24 | ✅ 已优化 |

## 在线数据调试结果（Airline Passengers 数据集）
| 模型 | R² | RMSE | 数据集 | 状态 |
|------|-----|------|--------|------|
| RandomForest | 0.9648 | 16.89 | Airline Passengers (144行) | ✅ |
> **在线数据调试结论**：PredictVisualizer 图表自动生成正常，模型泛化能力强（R²=0.9648）。网络数据下载验证通过。

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

## PM 复盘教训（2026-03-25）

### 🔴 教训1：验证修复时必须确认文件路径
**错误**：只验证了 `deep_research_web.py`（运行中服务），但修复在 `DeepPredict/deeppredict_web.py`（独立 App）。两个文件是分开的！

**正确做法**：验证修复时，必须先确认修复是在**运行中的网站服务文件**里，测试也必须对同一文件进行。独立 App 和网站服务是**两套代码**。

### 🔴 教训2：级联 Bug 要追踪所有状态不一致路径
**错误**：把下载崩溃归因于"P0 返回值长度问题"，但实际是 `is_fitted` 在异常时未正确设置，导致后续下载调用 `predict()` 时 model 为 None。

**正确做法**：分析级联 bug 时，追踪所有可能导致状态不一致的路径，包括异常处理路径中的 flag 设置，不能只看返回值数量。

### 🔴 教训3：重启服务后必须确认版本一致性
**错误**：报告 PatchTST 报错 `No module named 'models.patchtst_model'`，但命令行测试导入正常。问题是测试时用的是旧版服务器。

**正确做法**：重启网站后，确认运行中的服务版本与本地文件一致，再做验证测试。

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
| 2026-03-25 | PM 复盘教训记入：验证文件路径/级联Bug追踪/版本一致性 | - |
| 2026-03-25 | DeepPredict 迁移整合至 DeepResearch/DeepPredict/，Commit: bb43bd0 | bb43bd0 |
| 2026-03-24 | CNN1D 参数调优完成：seq_len=180, epochs=100, hidden=128, kernel=5, R²=0.6538 ≥ 0.55 | (已提交) |
| 2026-03-24 | 在线数据调试通过：Airline Passengers 数据集 RandomForest R²=0.9648，PredictVisualizer 图表自动生成正常 | (待提交) |
| 2026-03-24 | v1.13 新增 PredictVisualizer 可视化模块（时序图+置信区间+局部放大） | (本次) |
| 2026-03-24 | v1.12 CNN1D优化 R²=0.6538 | (本次) |
| 2026-03-24 | v1.11 初始版本，多模态+SHAP | 21a3388 |
| 2026-03-24 | LSTM参数优化 R²=0.6258 | ee29395 |
