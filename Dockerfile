# 使用官方 PyTorch 2.3.0 + CUDA 12.1 的稳定镜像
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 安装必要的系统底层依赖
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 启动 Git LFS
RUN git lfs install

# 【重要优化】将大模型权重固化在镜像内，彻底消灭 Serverless 扩容时的冷启动下载耗时
RUN mkdir -p /models/FitDiT && \
    git clone https://huggingface.co/BoyuanJiang/FitDiT /models/FitDiT

# 复制并按照精准版本安装 Python 库
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 将 RunPod 推理处理核心移入容器
COPY handler.py .

# 配置环境变量
ENV PYTHONPATH="/app:${PYTHONPATH}"

# 绑定 RunPod 启动命令
CMD [ "python", "-u", "handler.py" ]
