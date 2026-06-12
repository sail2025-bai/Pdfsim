# syntax=docker/dockerfile:1.4
#
# PDF 图纸相似性查找 API - BGE多模态版
#
# CPU版torch + Visualized-BGE 768维图文联合嵌入
# 镜像约1.2GB（torch CPU ~800MB + BGE权重 ~400MB）
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
        git wget \
 && rm -rf /var/lib/apt/lists/*

# 先装torch CPU版（小镜像，不拉CUDA）
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 装FlagEmbedding + visual_bge
RUN git clone https://github.com/FlagOpen/FlagEmbedding.git /tmp/FlagEmbedding \
 && cd /tmp/FlagEmbedding \
 && pip install -e . \
 && pip install -e research/visual_bge \
 && rm -rf /tmp/FlagEmbedding/.git

# 安装其余依赖
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 预下载BGE模型权重（避免首次请求时下载）
RUN python -c "\
from visual_bge.modeling import Visualized_BGE; \
print('Pre-downloading BGE model weights...'); \
model = Visualized_BGE(model_name_bge='BAAI/bge-base-en-v1.5'); \
model.eval(); \
print('BGE model ready'); \
" || echo "WARN: BGE model pre-download failed, will download on first request"

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/data/uploads /app/data/index /app/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
