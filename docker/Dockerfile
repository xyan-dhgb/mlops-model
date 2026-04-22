# ============================================================
# Dockerfile  –  ISIC 2024 Multimodal Base Image
# ============================================================
# Base image for GPU-accelerated training (EfficientNetB3 + TF 2.x)
# Used as parent by Dockerfile.training, .preprocessing, .serving
# ============================================================

FROM tensorflow/tensorflow:2.15.0-gpu

LABEL maintainer="isic2024-mlops"
LABEL description="ISIC 2024 Multimodal Skin Lesion Classifier – Base"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages (shared across all stages) ───────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/  src/
COPY MLflow_signature/ MLflow_signature/
COPY scripts/ scripts/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV TF_CPP_MIN_LOG_LEVEL=2

CMD ["python", "scripts/run_train.py"]
