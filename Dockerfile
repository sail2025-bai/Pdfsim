# syntax=docker/dockerfile:1.4
#
# PDF 图纸相似性查找 API - Dockerfile
#
# 注意：torch 体积大，生产环境可改 cpu-only 版
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖：PyMuPDF / opencv-headless 运行时
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# torch CPU-only 源（可选，显著减小镜像体积）
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu || \
    pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data/uploads /app/data/index /app/data/thumbnails

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
