FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制并建立固定的多方依赖关系
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 创建云盘专属挂载点占位符
RUN mkdir -p /runpod-volume

# 核心改变：全选复制您当前项目包含的全部 GitHub 源码脚本到工作目录
COPY . /app/

CMD [ "python", "-u", "/app/handler.py" ]
