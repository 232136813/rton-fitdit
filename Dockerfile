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

# ⚡【终极版本锁】锁死 onnxruntime-gpu==1.19.0，完美对齐基础镜像中的 CUDA 12.4，拒绝 CUDA 13 报错！
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
    "scipy"

# 实体复制你的 handler.py
COPY handler.py /app/handler.py

# 创建云盘挂载目录占位
RUN mkdir -p /runpod-volume

CMD [ "python", "-u", "/app/handler.py" ]
