# ChronoML Agent 专属记忆

## Agent 身份
- **职责**：全职负责 ChronoML（时序预测）模块的迭代更新
- **上级**：主 Agent（项目经理人）
- **报告频率**：每完成一个功能主动汇报给项目经理人

## 模块基本信息
- **独立项目路径**：`C:\Users\XJH\ChronoML\`（已从 DeepPredict 改名）
- **GitHub**：`https://github.com/xjhveteran199-bit/ChronoML`
- **版本**：v1.04
- **定位**：面向研究者的零门槛时序预测工具，支持多种 ML/DL 模型

## 当前版本功能
- **7种预测模型**：RandomForest / GradientBoosting / LSTM / CNN1D / PatchTST / LinearRegression / LogisticRegression
- **多模态输入**：支持选择多个 X 特征列 → 预测 Y 目标列
- **SHAP 可解释性**：特征重要性 + beeswarm 图（scienceplots 子刊风格）
- **数据预处理**：自动归一化、缺失值处理、日期特征提取（sin/cos 周期编码）
- **结果可视化**：预测时序图（含置信区间+局部放大）、残差分布图
- **一键下载**：CSV + 指标 + PNG 图 + ZIP 完整包

## 技术栈
- Python 3.12 + PyTorch 2.5.0（CPU）+ scikit-learn + Gradio 6.9.0
- numpy 1.26.4 + shap 0.51.0 + matplotlib

## 预测性能（最新测试）
| 数据集 | 样本数 | CNN1D R² |
|--------|--------|----------|
| 每日最低温度 | 3650 | 0.5065 ✅ |
| Airline Passengers | 144 | 0.9648 ✅ |

## 已知问题
- CNN1D 对小数据集可能不稳定（需更多 epoch）
- PatchTST 完整参数测试未跑
- seq=365 配置内存不足，seq=180 效果最优

## 迭代 Roadmap
1. ✅ 多模态输入（X多列选择）
2. ✅ SHAP 可解释性分析
3. ✅ CNN1D 参数调优（R²=0.6538 ≥ 0.55 目标达成）
4. 🔄 PatchTST 完整测试
5. 🔄 数据集自动特征工程（sin/cos 周期编码）
6. 🔄 模型缓存（避免重复训练）
7. 🔄 FastAPI 服务化

## 命名历史
- 原名：DeepPredict（独立项目 → 集成到 DeepResearch）
- 新名：ChronoML（2026-03-26 从 DeepPredict 改名拆分独立上架）

## 汇报规则
每完成一个功能后，向主 Agent（项目经理人）汇报：
```
【ChronoML 进展汇报】
时间: [现在时间]
本次更新: [具体内容]
修改文件: [文件名]
状态: 正常/有问题
下一步: [计划]
```

## 最近更新记录
| 日期 | 更新内容 | Commit |
|------|---------|--------|
| 2026-03-26 | 从 DeepPredict 改名为 ChronoML，GitHub + 本地同步 | 52e1e9e |
| 2026-03-25 | DeepPredict v1.04 最终版（含 CNN1D v4 / RevIN Bug 修复 / LSTM 稳定性） | - |
