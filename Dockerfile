FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. 复制并安装您锁定的依赖包
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. ⚡【终极修复】移除了拼接中的隐形空格，精准拉取官方开源算法脚本与模块
RUN git clone --depth 1 https://github.com/BoyuanJiang/FitDiT.git /tmp/fitdit_src && \
    cp /tmp/fitdit_src/gradio_sd3.py /app/gradio_sd3.py && \
    cp -r /tmp/fitdit_src/src /app/src && \
    rm -rf /tmp/fitdit_src

# 创建云盘专属固定挂载点
RUN mkdir -p /runpod-volume

# 3. 复制您项目本地编写的 handler.py 覆盖进去
COPY handler.py /app/handler.py

CMD [ "python", "-u", "/app/handler.py" ]
