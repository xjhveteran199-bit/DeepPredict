# ====== 多阶段构建 ======
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（先装 CPU-only torch 节省体积）
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 复制 requirements.txt 并安装依赖
COPY requirements.txt .
# 过滤掉 torch（已单独装）和 gradio（通过 pip install gradio 装最新）
RUN grep -v "^torch" requirements.txt > req_filtered.txt || cp requirements.txt req_filtered.txt
RUN pip install --no-cache-dir -r req_filtered.txt

# ====== 生产镜像 ======
FROM python:3.12-slim

WORKDIR /app

# 运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制源代码
COPY . .

# 环境变量
ENV PORT=8000
ENV PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

EXPOSE 8000 7860

# 启动（通过 Gunicorn 运行 FastAPI，Gradio 在子进程）
CMD ["sh", "-c", "python -c \"from web.backend.models.database import SQLModel, engine; SQLModel.metadata.create_all(engine)\" && uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 & python deeppredict_web.py & wait"]
