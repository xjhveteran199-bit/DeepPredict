# 更新日志 / Changelog

## v1.05 — 2026-03-23

### 新增
- **数据解耦器（DataDecoupler）**：`src/data/data_decoupler.py`
  - 自动识别列类型：日期(numeric)、类别(categorical)、数值(numeric)、文本(text)
  - 日期列 → 数值 ordinal 编码 + 标准化
  - 类别列 → LabelEncoder 编码
  - 数值列 → Z-score 标准化
  - 上传 CSV 后显示每列的类型分析
- **增强 1D-CNN（EnhancedCNN1D）**：`src/models/cnn1d_complex.py`
  - Multi-Scale CNN：并行3分支（kernel=3/5/7），分别捕捉短/中/长期模式
  - Residual Block × 2：稳定深层训练
  - SE Channel Attention：自动关注重要特征通道
  - 直接多步预测输出，适合多变量复杂数据

### 修复
- PatchTST `predict_future()` 缺少 `X` 参数报错问题
- Sklearn 模型新增 `predict_future()` 滚动预测方法

---

## v1.04 — 2026-03-23

- **数据解耦**：上传 CSV 后自动识别并分离不同类型数据（类别列/数值列/日期列），分别预处理后再合并输入模型
- **1D-CNN**：新增 1D 卷积神经网络模型，专门处理复杂时序数据（多变量、不规则采样、混合格式），比 LSTM 更好地捕捉局部模式

---

## v1.03 — 2026-03-23

### 新增
- **X 自变量选择**：用户上传 CSV 后可自主选择 X 列（时间/自变量列）和 Y 列（目标列）
- **支付宝收款系统**：国内用户可扫码购买积分，无需 Stripe
- **积分系统**：注册送 100 积分，按次扣积分（PatchTST 20分 / LSTM 15分 / GradientBoosting 5分）
- **FastAPI 后端**：`/api/*` REST 接口，含用户管理、积分、支付、Webhook
- **NanoBnana 风格首页**：暗色科技风，定价 ¥7 / ¥28 / ¥49
- **Docker 部署支持**：Dockerfile + docker-compose，一键部署

### 修复
- LSTM predict_future 参数名修正（`n_future` → `steps`）
- CNN1D 小数据集保护（seq_len < 16 时自动调整）
- 模块 import 路径问题

### 变更
- Gradio 界面步骤编号更新：① X列 → ② Y列 → ③ 模式 → ④ 模型 → ⑤ 需求描述

---

## v1.02 — 2026-03-22

- 初始版本
- 支持 PatchTST / LSTM / GradientBoosting / RandomForest
- 智能数据分析 + 自动模型推荐
