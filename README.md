# 🧠 DeepPredict - 零门槛深度学习预测工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**无需编程基础，无需机器学习背景。** 上传 CSV，选择目标列，3 步完成 AI 预测。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🤖 **AI 自动选模型** | 系统自动分析数据，推荐最优模型（PatchTST / LSTM / GradientBoosting） |
| 📊 **零代码操作** | 告别 Jupyter Notebook，点点鼠标完成预测 |
| 🔒 **数据本地处理** | 数据不上传服务器，保护隐私安全 |
| 📱 **支持手机访问** | 响应式设计，随时随地使用 |
| 🚀 **快速出结果** | 几分钟内完成训练和预测 |

---

## 📥 安装

### 环境要求
- Python 3.8+
- Windows / macOS / Linux

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/xjhveteran199-bit/DeepPredict.git
cd DeepPredict

# 2. 创建虚拟环境（推荐）
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 启动 Web 版
python deeppredict_web.py
```

安装完成后，打开浏览器访问：**http://localhost:7860**

---

## 🎯 使用方法

### 1. 导入数据
上传你的 CSV 文件，系统自动分析数据结构。

### 2. 选择目标列
选择你要预测的列（Y），系统根据数据情况推荐最优模型。

### 3. 开始训练
点击「开始训练」，等待几分钟查看预测结果。

---

## 📊 支持的模型

| 模型 | 类型 | 适用场景 |
|------|------|----------|
| **PatchTST** | Transformer | 长序列时序预测（≥200条数据） |
| **LSTM** | 深度学习 | 中短期时序预测（≥100条数据） |
| **GradientBoosting** | 集成学习 | 分类任务、回归任务 |
| **RandomForest** | 集成学习 | 分类任务、回归任务 |

---

## 💡 适用场景

- 🧬 **生物医学**：细胞培养数据、药物反应预测
- 📈 **市场分析**：销售预测、流量预测
- 🌡️ **环境科学**：空气质量、气候变化预测
- 🏭 **工业生产**：设备故障预测、质量控制
- 📚 **学术研究**：时序数据分析与预测

---

## 🔧 目录结构

```
DeepPredict/
├── deeppredict_web.py     # Web 界面主程序
├── src/
│   ├── models/            # 预测模型
│   │   ├── patchtst_model.py    # PatchTST Transformer
│   │   ├── lstm_model.py        # LSTM 模型
│   │   └── ...
│   └── core/              # 核心模块
│       └── task_router.py       # 任务路由
├── test_data/            # 测试数据
├── requirements.txt      # 依赖列表
└── README.md
```

---

## 💰 定价

| 版本 | 价格 | 说明 |
|------|------|------|
| 个人版 | **免费** | 适合学习和研究 |
| 预测服务 | **¥5/次** | 单次预测，不限数据量 |
| 批量服务 | **¥99/月** | 无限次预测 + 优先模型 |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📧 联系

- **GitHub**: [xjhveteran199-bit/DeepPredict](https://github.com/xjhveteran199-bit/DeepPredict)
- **邮箱**: contact@deeppredict.ai

---

*让 AI 预测触手可及*
