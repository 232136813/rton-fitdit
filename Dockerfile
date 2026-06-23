FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System libraries for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies matching FitDiT official environment
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "diffusers==0.31.0" \
    "transformers==4.44.0" \
    "accelerate>=0.31.0" \
    "safetensors>=0.4.0" \
    "huggingface_hub>=0.23.0" \
    "einops>=0.7.0" \
    "opencv-python-headless>=4.8.0" \
    "Pillow>=10.3.0" \
    "onnxruntime-gpu==1.20.1" \
    "scipy" \
    "scikit-image>=0.21.0" \
    "sentencepiece" \
    "protobuf" \
    "peft>=0.11.0" \
    "matplotlib"

# Copy application code
COPY handler.py /app/handler.py
COPY download_weights.py /app/download_weights.py

# Network volume mount point for model weights
RUN mkdir -p /runpod-volume


CMD ["python", "-u", "/app/handler.py"]