FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. 复制并安装完全体依赖
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. 软链接网盘中的 handler.py 到本地工作目录
# 这样容器启动运行 python /app/handler.py 时，能直接执行网盘里的最新监听代码

COPY handler.py /app/handler.py

# 3. 创建云盘固定挂载点
RUN mkdir -p /runpod-volume


CMD [ "python", "-u", "/app/handler.py" ]
