FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖（GradIO 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 预装 CPU-only PyTorch（体积小很多）
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 复制依赖文件
COPY requirements.txt .

# 避免装 GPU 版 torch（如果 requirements里有的话先过滤掉）
RUN grep -v "^torch" requirements.txt > requirements_filtered.txt || cp requirements.txt requirements_filtered.txt
RUN pip install --no-cache-dir -r requirements_filtered.txt

# 装 Gradio（如果 requirements 里没有）
RUN pip install --no-cache-dir gradio

# 复制源代码
COPY . .

# 暴露端口
EXPOSE 7860

# 启动
ENV PORT=7860
CMD ["python", "deeppredict_web.py"]
