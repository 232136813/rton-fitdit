# 选用兼容 RTX 4090 / A100 等高端算力架构的 CUDA 12 生产级 PyTorch 基础镜像
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 设置容器内无交互前端，防止阻塞构建
ENV DEBIAN_FRONTEND=noninteractive

# 安装核心系统级图像解码扩展包
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 确立容器内的工作主目录
WORKDIR /app

# 预先复制依赖，利用 Docker 层缓存机制加速后续构建
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 预建模型挂载占位点，保证后续 RunPod 云盘直接对接到此处
RUN mkdir -p /models

# 将编写完成的业务处理逻辑代码复制到容器中
COPY handler.py /app/handler.py

# 设置启动命令，RunPod Serverless 会在拉取容器后自动执行此脚本监听外部请求
CMD [ "python", "-u", "/app/handler.py" ]
