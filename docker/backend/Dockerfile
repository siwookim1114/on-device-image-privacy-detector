# =============================================================================
# Backend Dockerfile — On-Device Image Privacy Detector
# =============================================================================
#
# Base: python:3.11-slim  (matches lab_env Python version)
#
# Layer order is deliberate:
#   1. OS system deps  — rarely changes
#   2. Heavy ML deps   — changes occasionally, expensive to rebuild
#   3. Lightweight API deps (backend/requirements.txt) — changes often
#   4. App source code — changes constantly
#
# ML models (MTCNN, EasyOCR, YOLOv8n, FaceNet, MobileSAM) are NOT baked into
# the image. They download to /root/.cache on first run and persist in the
# named Docker volume `ml_model_cache` declared in docker-compose.yml.
# This keeps the image size small (~3 GB) and avoids re-downloading on rebuild.
# =============================================================================

FROM python:3.11-slim AS base

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      # OpenCV (headless) native libs
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      # python-magic
      libmagic1 \
      # git — needed by pip to install mobile-sam from GitHub
      git \
      # health-check curl
      curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage: heavy-ml
# Install PyTorch (CPU build) and the large CV/ML libraries separately so
# Docker cache preserves this ~2.5 GB layer across code-only changes.
# ---------------------------------------------------------------------------
FROM base AS heavy-ml

# CPU-only torch — the container has no Metal GPU
# Pin to the same major versions used in requirements.txt (2.x / 0.x)
RUN pip install --no-cache-dir \
      torch==2.9.1 \
      torchvision==0.24.1 \
      --index-url https://download.pytorch.org/whl/cpu

# ML libraries that are large or have complex native deps
RUN pip install --no-cache-dir \
      facenet-pytorch==2.5.3 \
      easyocr==1.7.2 \
      ultralytics==8.4.0 \
      einops==0.8.2 \
      timm==1.0.24 \
      transformers==5.2.0

# MobileSAM — installed from GitHub source
RUN pip install --no-cache-dir \
      "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git"

# LangChain + agent framework (medium-sized, groups well here)
RUN pip install --no-cache-dir \
      langchain==1.2.10 \
      langchain-core==1.2.13 \
      langchain-openai==1.1.10 \
      langchain-ollama==1.0.1 \
      langgraph==1.0.8

# Remaining pipeline deps
RUN pip install --no-cache-dir \
      PyYAML==6.0.2 \
      Pillow==12.1.0 \
      opencv-python-headless==4.11.0.86 \
      numpy==2.4.2 \
      pydantic==2.12.5 \
      pymongo==4.16.0 \
      cryptography==46.0.3 \
      loguru==0.7.3 \
      tqdm==4.67.1 \
      rich==14.2.0

# ---------------------------------------------------------------------------
# Stage: final — app image
# ---------------------------------------------------------------------------
FROM heavy-ml AS final

WORKDIR /app

# Install lightweight FastAPI / web deps (changes frequently — own layer)
COPY backend/requirements.txt /tmp/backend_requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend_requirements.txt

# ---------------------------------------------------------------------------
# Copy application source
# Only the directories the backend actually imports from at runtime.
# The Dockerfile sits in docker/backend/ so context root is the repo root.
# ---------------------------------------------------------------------------
COPY agents/       /app/agents/
COPY utils/        /app/utils/
COPY backend/      /app/backend/
# configs and yolov8n.pt are bind-mounted at runtime (see docker-compose.yml)

# Create data directories (compose bind-mounts will overlay these at runtime)
RUN mkdir -p \
      /app/data/tmp_uploads \
      /app/data/full_pipeline_results \
      /app/data/face_db \
      /app/data/provenance_logs \
      /app/logs

# Make the repo root the Python path so `from agents.xxx` and `from utils.xxx`
# resolve correctly without installing the project as a package.
ENV PYTHONPATH="/app"

# Uvicorn entry point — matches backend/main.py module path
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
