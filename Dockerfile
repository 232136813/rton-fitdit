FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统基础图像库
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⚡【终极闭环】强行补齐最后一只地雷画图包 "matplotlib"，让姿态提取组件 dwpose 顺利闭环！
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "einops>=0.7.0" \
    "opencv-python>=4.8.0" \
    "scikit-image>=0.21.0" \
    "timm>=0.9.0" \
    "omegaconf>=2.3.0" \
    "huggingface_hub>=0.20.0,<0.24.0" \
    "diffusers>=0.28.0,<0.29.0" \
    "transformers>=4.40.0,<4.45.0" \
    "accelerate>=0.29.0" \
    "peft>=0.10.0" \
    "safetensors>=0.4.0" \
    "Pillow>=10.3.0" \
    "sentencepiece" \
    "protobuf" \
    "onnxruntime-gpu==1.19.0" \
    "scipy" \
    "matplotlib"

# 实体复制你的 handler.py
COPY handler.py /app/handler.py

# 创建云盘挂载目录占位
RUN mkdir -p /runpod-volume

CMD [ "python", "-u", "/app/handler.py" ]
