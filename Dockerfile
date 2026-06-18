# 选用稳定的深度学习底座镜像
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 安装操作系统级别的基础库和动态链接库（图像处理所需）
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 初始化 Git Large File Storage (LFS)
RUN git lfs install

# 下载权重文件并将其缓存至镜像层中（此处需要几分钟，请保持网络畅通）
RUN mkdir -p /models/FitDiT && \
    git clone https://huggingface.co/BoyuanJiang/FitDiT /models/FitDiT

# 复制并安装 Python 包依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 将处理代码移入容器
COPY handler.py .

# 设定 Python 执行环境变量
ENV PYTHONPATH="/app:${PYTHONPATH}"

# 绑定 RunPod 入口命令
CMD [ "python", "-u", "handler.py" ]
